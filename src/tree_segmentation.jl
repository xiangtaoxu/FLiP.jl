"""
Individual tree segmentation pipeline.

Operates on a point cloud that already carries above-ground height (`:AGH`).
The public entry point is [`tree_segmentation`](@ref); the file is laid out
orchestrator-first, helpers below in call order.

# Pipeline

1. Filter to above-ground points (`:AGH > tree_nearground_agh_threshold`).
2. Discover connected components via coordinate-only union-find (no graph),
   to bound the per-component memory of all later steps.
3. Process each component independently
   ([`_process_single_connected_component`](@ref)):
   3a. Build a radius graph for the component.
   3b. Label Non-Branching Segments (NBS) via
       [`label_non_branching_segments`](@ref) — greedy expansion from
       near-ground seeds, linearity-constrained.
   3c. Build a skeleton cloud + skeleton graph
       ([`create_skeleton_cloud`](@ref)) — one vertex per NBS-node.
   3d. Assemble NBS into trees ([`assemble_segments`](@ref)) — grow from
       near-ground NBS along the skeleton graph using Rule A (new branch)
       or Rule B (merge into existing NBS).
4. Concatenate per-component skeletons into one cloud + graph.
5. Cross-component orphan-NBS rescue
   ([`process_orphan_segments`](@ref)).
6. Re-order tree IDs by descending point count (largest tree → 1).
7. Stamp `:nbs_id`, `:node_id`, `:tree_id`, `:tree_nbs_id` onto the cloud
   and optionally write a CloudCompare-readable skeleton OBJ.
"""

# ── Entry point ───────────────────────────────────────────────────

"""
    tree_segmentation(pc::PointCloud; cfg::FLiPConfig=_CFG) -> NamedTuple

Run the individual-tree segmentation workflow on a point cloud already
carrying `:AGH` (above-ground height).

Per-component processing keeps peak memory at O(N_largest_component × avg_degree)
rather than O(N_total × avg_degree). Components are discovered by a lightweight
coordinate-only union-find before any graph is built.

# Returns

`NamedTuple` with fields:
- `filtered_cloud::PointCloud` — the above-ground points with `:nbs_id`,
  `:node_id`, `:tree_id`, `:tree_nbs_id` attached
- `pc_output::PointCloud` — alias of `filtered_cloud` (back-compat)
- `skeleton_cloud::PointCloud` — concatenated per-component skeleton (one
  vertex per NBS node), carrying `:node_id` and `:n_points`
- `n_components::Int` — number of valid connected components processed
- `neighbor_radius::Float64` — the radius used for component discovery /
  per-component graph build

See the file docstring for the seven-step pipeline outline.
"""
function tree_segmentation(pc::PointCloud; cfg::FLiPConfig=_CFG)
    hasattribute(pc, :AGH) || throw(ArgumentError(
        "tree_segmentation requires AGH attribute on input point cloud"))

    agh = getattribute(pc, :AGH)

    # ── Step 1: filter to above-ground points ─────────────────────
    threshold = cfg.tree_nearground_agh_threshold
    nearground_idx = findall(i -> isfinite(float(agh[i])) && float(agh[i]) > threshold,
                             eachindex(agh))
    empty_result = (
        filtered_cloud  = pc[1:0],
        pc_output       = pc[1:0],
        skeleton_cloud  = PointCloud(zeros(Float64, 0, 3), Dict{Symbol,Vector}()),
        n_components    = 0,
        neighbor_radius = 0.0,
    )
    isempty(nearground_idx) && return empty_result

    pc_filtered     = pc[nearground_idx]
    coords_filtered = coordinates(pc_filtered)
    agh_filtered    = float.(getattribute(pc_filtered, :AGH))
    N = size(coords_filtered, 1)

    neighbor_radius = cfg.tree_neighbor_radius > 0 ?
                      cfg.tree_neighbor_radius :
                      2.0 * cfg.pipeline_subsample_res
    neighbor_radius > 0 || throw(ArgumentError("tree neighbor radius must be > 0"))

    # ── Step 2: discover components via coordinate-only union-find ──
    cc_labels = connected_component_labels(coords_filtered, neighbor_radius,
                                           cfg.tree_min_nbs_size)

    # Build component → indices dispatch in one O(N) pass (was K× findall),
    # then release the per-point label vector (~ 8N bytes).
    cc_indices_dict = Dict{Int, Vector{Int}}()
    @inbounds for i in eachindex(cc_labels)
        cc_labels[i] > 0 || continue
        push!(get!(cc_indices_dict, cc_labels[i], Int[]), i)
    end
    cc_labels = Int[]   # free

    unique_ccs   = sort!(collect(keys(cc_indices_dict)))
    n_components = length(unique_ccs)
    @info "[tree_segmentation] $N filtered points → $n_components connected components" min_cc=cfg.tree_min_nbs_size

    n_components == 0 && return empty_result

    # ── Step 3: process each component independently ──────────────
    global_nbs_id      = zeros(Int32, N)
    global_node_id     = zeros(Int32, N)
    global_tree_id     = zeros(Int32, N)
    global_tree_nbs_id = zeros(Int32, N)

    all_skel_coords    = Matrix{eltype(coords_filtered)}[]
    all_skel_attrs     = Dict{Symbol,Vector}[]
    all_skel_edges     = Tuple{Int,Int}[]
    skel_vertex_offset = 0
    nbs_offset         = Int32(0)
    node_offset        = Int32(0)
    tree_offset        = Int32(0)

    for (ci, cc_id) in enumerate(unique_ccs)
        cc_indices = cc_indices_dict[cc_id]
        cc_n = length(cc_indices)
        @info "[tree_segmentation]   component $ci/$n_components: $cc_n points"

        cc_coords = coords_filtered[cc_indices, :]
        cc_agh    = agh_filtered[cc_indices]

        res = _process_single_connected_component(cc_coords, cc_agh, neighbor_radius; cfg=cfg)

        # Map local → global labels with running offsets
        local_nbs_max  = Int32(maximum(res.nbs_id;  init=0))
        local_node_max = Int32(maximum(res.node_id; init=0))
        local_tree_max = Int32(maximum(res.tree_id; init=0))
        @inbounds for (li, gi) in enumerate(cc_indices)
            nid  = res.nbs_id[li]
            global_nbs_id[gi]     = nid  > 0 ? nid  + nbs_offset  : Int32(0)
            noid = res.node_id[li]
            global_node_id[gi]    = noid > 0 ? noid + node_offset : Int32(0)
            tid  = res.tree_id[li]
            global_tree_id[gi]    = tid  > 0 ? tid  + tree_offset : Int32(0)
            tnid = res.tree_nbs_id[li]
            global_tree_nbs_id[gi] = tnid > 0 ? tnid + nbs_offset  : Int32(0)
        end

        # Accumulate skeleton cloud + edges with vertex offset
        skel_cloud = res.skeleton_cloud
        if npoints(skel_cloud) > 0
            push!(all_skel_coords, coordinates(skel_cloud))
            push!(all_skel_attrs, _all_attributes(skel_cloud))
            for e in Graphs.edges(res.graph_skeleton)
                push!(all_skel_edges, (src(e) + skel_vertex_offset,
                                       dst(e) + skel_vertex_offset))
            end
            skel_vertex_offset += npoints(skel_cloud)
        end

        nbs_offset  += local_nbs_max
        node_offset += local_node_max
        tree_offset += local_tree_max
    end

    empty!(cc_indices_dict)   # release after per-component loop

    # ── Step 4: merge skeletons across components ─────────────────
    if isempty(all_skel_coords)
        merged_skel       = PointCloud(zeros(Float64, 0, 3), Dict{Symbol,Vector}())
        merged_skel_graph = SimpleGraph{Int}(0)
    else
        merged_skel       = merge_pointclouds(all_skel_coords, all_skel_attrs)
        merged_skel_graph = SimpleGraph{Int}(skel_vertex_offset)
        for (u, v) in all_skel_edges
            add_edge!(merged_skel_graph, u, v)
        end
    end
    # Release per-component skeleton accumulators before orphan rescue allocates
    empty!(all_skel_coords); empty!(all_skel_attrs); empty!(all_skel_edges)

    # ── Step 5: cross-component orphan-NBS rescue ─────────────────
    process_orphan_segments(coords_filtered, global_nbs_id,
                            global_tree_id, global_tree_nbs_id; cfg=cfg)

    # ── Step 6: reorder tree_id by descending point count ─────────
    global_tree_id = relabel_by_occurrence(global_tree_id; positive_only=true, T_out=Int32)
    tree_offset    = Int32(maximum(global_tree_id; init=0))

    # ── Step 7: attach attributes + write skeleton OBJ ────────────
    setattribute!(pc_filtered, :nbs_id,      global_nbs_id)
    setattribute!(pc_filtered, :node_id,     global_node_id)
    setattribute!(pc_filtered, :tree_id,     global_tree_id)
    setattribute!(pc_filtered, :tree_nbs_id, global_tree_nbs_id)

    if !isempty(cfg.pipeline_output_dir)
        obj_path = joinpath(expanduser(cfg.pipeline_output_dir),
                            "$(cfg.pipeline_output_prefix)skeleton_graph.obj")
        _write_polyline_obj(obj_path, coordinates(merged_skel), merged_skel_graph)
        @info "[tree_segmentation] wrote: $obj_path"
    end

    @info "[tree_segmentation] done" n_components=n_components n_trees=tree_offset n_nbs=nbs_offset

    return (
        filtered_cloud  = pc_filtered,
        pc_output       = pc_filtered,
        skeleton_cloud  = merged_skel,
        n_components    = Int(n_components),
        neighbor_radius = neighbor_radius,
    )
end

# ── Per-component processing ──────────────────────────────────────

"""
    _process_single_connected_component(cc_coords, cc_agh, neighbor_radius; cfg)
        -> NamedTuple

Run the NBS → skeleton → assembly chain for one component. All four sub-stages
share the same per-component radius graph. Returns NamedTuple fields:
`nbs_id`, `node_id`, `tree_id`, `tree_nbs_id`, `skeleton_cloud`, `graph_skeleton`.
"""
function _process_single_connected_component(cc_coords::AbstractMatrix{<:Real},
                                             cc_agh::AbstractVector{<:Real},
                                             neighbor_radius::Real;
                                             cfg::FLiPConfig)
    g_res    = build_radius_graph(cc_coords, neighbor_radius)
    nbs_res  = label_non_branching_segments(g_res.graph, cc_coords, cc_agh; cfg=cfg)
    skel_res = create_skeleton_cloud(g_res.graph, cc_coords, nbs_res.node_id)
    asm_res  = assemble_segments(g_res.graph, cc_coords,
                                 nbs_res.nbs_id, nbs_res.node_id, cc_agh,
                                 skel_res.graph_skeleton, skel_res.skeleton_cloud;
                                 cfg=cfg)
    return (nbs_id         = nbs_res.nbs_id,
            node_id        = nbs_res.node_id,
            tree_id        = asm_res.tree_id,
            tree_nbs_id    = asm_res.tree_nbs_id,
            skeleton_cloud = skel_res.skeleton_cloud,
            graph_skeleton = skel_res.graph_skeleton)
end

# ── Step 3b: NBS labeling ─────────────────────────────────────────

"""
    label_non_branching_segments(graph, points, agh_values; cfg) -> NamedTuple

Segment every vertex of `graph` into non-branching segments (NBS) by greedy
neighborhood expansion. Connected components smaller than `cfg.tree_min_nbs_size`
are discarded upfront (label 0). Valid segments are relabeled by descending
size (largest segment → label 1).

Returns `(nbs_id::Vector{Int32}, node_id::Vector{Int32})`.
"""
function label_non_branching_segments(
    graph::SimpleGraph{Int},
    points::AbstractMatrix{<:Real},
    agh_values::AbstractVector{<:Real};
    cfg::FLiPConfig = _CFG,
)
    N = nv(graph)
    size(points, 1) == N     || throw(ArgumentError("graph vertex count must match number of points"))
    length(agh_values) == N  || throw(ArgumentError("agh_values length must match graph vertex count"))

    min_segment_size       = cfg.tree_min_nbs_size
    neighbor_distance      = cfg.tree_nbs_neighbor_distance
    nearground_agh_ceiling = cfg.tree_nearground_agh_threshold + 2.0 * cfg.pipeline_subsample_res

    global_nbs_id  = zeros(Int, N)
    global_node_id = zeros(Int, N)
    gsws           = GreedySearchWorkspace(N)
    unlabeled_mask = trues(N)

    # Pre-pass: discard isolated clusters smaller than min_segment_size (O(V+E)).
    for comp in connected_components(graph)
        length(comp) >= min_segment_size && continue
        @inbounds for v in comp
            global_nbs_id[v]  = -1
            unlabeled_mask[v] = false
        end
    end

    # Pre-allocated buffers for _find_seed_clusters (reused across iterations)
    seed_candidate_buf = sizehint!(Int[], 1024)
    frontier_buf       = sizehint!(Int[], 1024)   # rejected frontier CCs from greedy searches

    # Pre-sort by ascending z for fallback seed selection.
    z_sorted = sortperm(view(points, :, 3); rev=false)
    z_cursor = Ref(1)

    next_id          = 1
    next_global_node = 1
    n_labeled_total  = count(!, unlabeled_mask)   # already labeled (discarded small CCs)
    last_pct_report  = 0
    t_nbs_start      = time()

    while true
        seed_clusters = _find_seed_clusters(
            points, agh_values, nearground_agh_ceiling,
            unlabeled_mask, graph,
            gsws.cc_ws;
            is_first_iteration = (next_id == 1),
            z_sorted       = z_sorted,
            z_cursor       = z_cursor,
            frontier_buf   = frontier_buf,
            candidate_buf  = seed_candidate_buf,
        )
        isempty(seed_clusters) && break   # no unlabeled points remain

        for start_vertices in seed_clusters
            # Skip if any point was already claimed by an earlier search this round
            any_claimed = false
            @inbounds for v in start_vertices
                if !unlabeled_mask[v]
                    any_claimed = true
                    break
                end
            end
            any_claimed && continue

            result = greedy_neighborhood_search(
                graph, start_vertices, neighbor_distance;
                vertex_mask          = unlabeled_mask,
                workspace            = gsws,
                points               = points,
                linearity_angle_deg  = Float64(cfg.tree_linearity_angle_deg),
                min_frontier_cc_size = Int(cfg.tree_frontier_min_cc_size),
            )
            labeled_idx = result.vertices
            node_ids    = result.node_ids
            n_labeled   = length(labeled_idx)

            # Collect rejected frontier CCs as seeds for future iterations
            append!(frontier_buf, result.rejected_frontier)

            if n_labeled < min_segment_size
                @inbounds for v in labeled_idx
                    global_nbs_id[v]  = -1
                    unlabeled_mask[v] = false
                end
            else
                offset = next_global_node - 1
                @inbounds for i in 1:n_labeled
                    v = labeled_idx[i]
                    global_nbs_id[v]  = next_id
                    global_node_id[v] = node_ids[i] + offset
                    unlabeled_mask[v] = false
                end
                next_global_node += result.max_node_id
                next_id          += 1
            end

            n_labeled_total += n_labeled
        end

        pct = round(Int, 100.0 * n_labeled_total / N)
        if pct >= last_pct_report + 5
            last_pct_report = pct - (pct % 5)
            @info "NBS labeling progress" pct=last_pct_report nbs_count=next_id-1 elapsed_s=round(time()-t_nbs_start, digits=1)
        end
    end

    # Relabel valid segments by descending size; -1 (discarded) and 0 (unprocessed)
    # both fall through `positive_only=true` to 0.
    nbs_relabeled = relabel_by_occurrence(global_nbs_id; positive_only=true, T_out=Int32)
    return (nbs_id = nbs_relabeled, node_id = Int32.(global_node_id))
end

"""
    _find_seed_clusters(points, agh_values, nearground_agh_ceiling, unlabeled_mask,
                        graph, cc_workspace;
                        is_first_iteration, z_sorted, z_cursor,
                        frontier_buf, candidate_buf) -> Vector{Vector{Int}}

Find seed clusters for `label_non_branching_segments`. Returns a list of vertex-index
vectors sorted by descending cluster size (first iteration) or ascending z (subsequent).

**First iteration**: collects all unlabeled vertices with AGH ≤ `nearground_agh_ceiling`,
splits them into connected components via `connected_component_subset!`, and returns
them ranked largest-first.

**Subsequent iterations**: draws seeds from `frontier_buf` — rejected frontier CCs
accumulated from prior `greedy_neighborhood_search` calls. Filters out already-labeled
vertices, sorts the remainder by ascending z, and returns each as a single-point seed
cluster. When `frontier_buf` is exhausted, falls back to the next lowest-z unlabeled
point via `z_cursor` (O(1) amortised). Returns an empty vector when no unlabeled points
remain (signals termination).
"""
function _find_seed_clusters(
    points::AbstractMatrix{<:Real},
    agh_values::AbstractVector{<:Real},
    nearground_agh_ceiling::Float64,
    unlabeled_mask::BitVector,
    graph::SimpleGraph{Int},
    cc_workspace::ConnectedComponentSubsetWorkspace;
    is_first_iteration::Bool,
    z_sorted::Vector{Int},
    z_cursor::Ref{Int},
    frontier_buf::Vector{Int},
    candidate_buf::Vector{Int},
)
    empty!(candidate_buf)

    if is_first_iteration
        # Collect all unlabeled near-ground vertices
        @inbounds for v in eachindex(unlabeled_mask)
            unlabeled_mask[v] || continue
            float(agh_values[v]) <= nearground_agh_ceiling || continue
            push!(candidate_buf, v)
        end
        if !isempty(candidate_buf)
            labels = connected_component_subset!(cc_workspace, graph, candidate_buf)
            return group_indices_by_label(candidate_buf, labels)
        end
    else
        # Draw seeds from the rejected-frontier buffer.
        # Filter out vertices that have been labeled since they were added.
        @inbounds for v in frontier_buf
            unlabeled_mask[v] && push!(candidate_buf, v)
        end
        empty!(frontier_buf)

        if !isempty(candidate_buf)
            # Sort by ascending z so growth proceeds upward from branch points
            sort!(candidate_buf; by = v -> @inbounds(points[v, 3]))
            return Vector{Int}[Int[v] for v in candidate_buf]
        end
    end

    # Fallback: advance z_cursor to next unlabeled point (O(1) amortised)
    while z_cursor[] <= length(z_sorted)
        v = z_sorted[z_cursor[]]
        z_cursor[] += 1
        if unlabeled_mask[v]
            return Vector{Int}[Int[v]]
        end
    end

    # No unlabeled points remain — signal termination
    return Vector{Vector{Int}}()
end

# ── Step 3c: skeleton construction ────────────────────────────────

"""
    create_skeleton_cloud(graph, coords_filtered, node_id; template_pc=nothing)
        -> NamedTuple{(:skeleton_cloud, :graph_skeleton)}

Build a skeleton point cloud and skeleton graph from NBS node assignments produced
by [`label_non_branching_segments`](@ref).

Each non-zero node label `n` in `node_id` is represented in the skeleton by the
centroid of all points assigned to that node. The returned `skeleton_cloud` carries
per-point attributes `:node_id` (original node label) and `:n_points` (number of raw
points contributing to the centroid).

The skeleton graph has one vertex per node. An edge is inserted between nodes A and B
for every point-level edge in `graph` that crosses the A/B node boundary; the edge
counts are tracked for completeness but no MST pruning is applied.
"""
function create_skeleton_cloud(
    graph::SimpleGraph{Int},
    coords_filtered::AbstractMatrix{<:Real},
    node_id::AbstractVector{<:Integer};
    template_pc::Union{Nothing, PointCloud} = nothing,
)
    N = nv(graph)
    size(coords_filtered, 1) == N || throw(ArgumentError("graph vertex count must match coords_filtered rows"))
    length(node_id) == N          || throw(ArgumentError("node_id length must match graph vertex count"))

    max_node = Int(maximum(node_id; init=0))
    if max_node == 0
        empty_pc = PointCloud(zeros(Float64, 0, 3), Dict{Symbol,Vector}())
        return (skeleton_cloud=empty_pc, graph_skeleton=SimpleGraph{Int}(0))
    end

    # Per-node coordinate sums and counts
    node_sum = zeros(Float64, max_node, 3)
    node_cnt = zeros(Int, max_node)
    @inbounds for i in 1:N
        nid = Int(node_id[i])
        nid > 0 || continue
        node_sum[nid, 1] += Float64(coords_filtered[i, 1])
        node_sum[nid, 2] += Float64(coords_filtered[i, 2])
        node_sum[nid, 3] += Float64(coords_filtered[i, 3])
        node_cnt[nid] += 1
    end

    # Keep only nodes with at least one point; build remap original → skeleton vertex
    valid_nodes  = [n for n in 1:max_node if node_cnt[n] > 0]
    n_nodes      = length(valid_nodes)
    node_to_skel = zeros(Int, max_node)
    for (si, n) in enumerate(valid_nodes)
        node_to_skel[n] = si
    end

    skel_coords = zeros(Float64, n_nodes, 3)
    skel_npts   = zeros(Int32, n_nodes)
    skel_nids   = zeros(Int32, n_nodes)
    for (si, n) in enumerate(valid_nodes)
        c = node_cnt[n]
        skel_coords[si, 1] = node_sum[n, 1] / c
        skel_coords[si, 2] = node_sum[n, 2] / c
        skel_coords[si, 3] = node_sum[n, 3] / c
        skel_npts[si] = Int32(c)
        skel_nids[si] = Int32(n)
    end

    skel_pc = PointCloud(skel_coords, Dict{Symbol,Vector}())
    setattribute!(skel_pc, :node_id,  skel_nids)
    setattribute!(skel_pc, :n_points, skel_npts)

    # Count cross-node edge connections (informational; no MST pruning)
    edge_counts = Dict{Tuple{Int,Int}, Int}()
    @inbounds for e in Graphs.edges(graph)
        u  = src(e); v = dst(e)
        nA = Int(node_id[u]); nB = Int(node_id[v])
        (nA > 0 && nB > 0 && nA != nB) || continue
        siA = node_to_skel[nA]; siB = node_to_skel[nB]
        (siA > 0 && siB > 0) || continue
        key = siA < siB ? (siA, siB) : (siB, siA)
        edge_counts[key] = get(edge_counts, key, 0) + 1
    end

    if isempty(edge_counts)
        return (skeleton_cloud=skel_pc, graph_skeleton=SimpleGraph{Int}(n_nodes))
    end

    graph_skeleton = SimpleGraph{Int}(n_nodes)
    for ((u, v), _) in edge_counts
        add_edge!(graph_skeleton, u, v)
    end

    return (skeleton_cloud=skel_pc, graph_skeleton=graph_skeleton)
end

# ── Step 3d: assembly ─────────────────────────────────────────────

"""
    assemble_segments(graph, coords, nbs_id, node_id, agh_values,
                      graph_skeleton, skeleton_cloud; cfg) -> NamedTuple

Assemble non-branching segments into individual trees by iteratively growing
from near-ground seed NBS outward through the skeleton graph.

# Arguments
- `graph::SimpleGraph{Int}`: point-level radius graph
- `coords::AbstractMatrix{<:Real}`: N×3 point coordinates
- `nbs_id::AbstractVector{<:Integer}`: per-point NBS label (0=discarded, 1..k)
- `node_id::AbstractVector{<:Integer}`: per-point node label (globally unique)
- `agh_values::AbstractVector{<:Real}`: per-point above-ground height
- `graph_skeleton::SimpleGraph{Int}`: skeleton graph (one vertex per node)
- `skeleton_cloud::PointCloud`: skeleton point cloud with `:node_id` attribute
- `cfg::FLiPConfig`: uses `tree_nearground_agh_threshold`, `tree_assembly_merge_threshold`

# Returns
`(tree_nbs_id::Vector{Int32}, tree_id::Vector{Int32})` where `tree_id` is the
per-point tree assignment (0=unassigned) and `tree_nbs_id` is the per-point NBS
label re-ordered continuously by descending size.
"""
function assemble_segments(
    graph::SimpleGraph{Int},
    coords::AbstractMatrix{<:Real},
    nbs_id::AbstractVector{<:Integer},
    node_id::AbstractVector{<:Integer},
    agh_values::AbstractVector{<:Real},
    graph_skeleton::SimpleGraph{Int},
    skeleton_cloud::PointCloud;
    cfg::FLiPConfig = _CFG,
)
    N = size(coords, 1)
    length(nbs_id) == N      || throw(ArgumentError("nbs_id length must match number of points"))
    length(node_id) == N     || throw(ArgumentError("node_id length must match number of points"))
    length(agh_values) == N  || throw(ArgumentError("agh_values length must match number of points"))

    # Use the same near-ground ceiling as label_non_branching_segments:
    # threshold + 2× subsample resolution to account for discretisation
    nearground_ceiling = cfg.tree_nearground_agh_threshold + 2.0 * cfg.pipeline_subsample_res
    merge_threshold    = cfg.tree_assembly_merge_threshold

    tree_id     = zeros(Int32, N)
    tree_nbs_id = Int32.(copy(nbs_id))

    # ── Step 4.0: precomputations ────────────────────────────────
    # (a) nbs_id → point indices
    nbs_points = Dict{Int, Vector{Int}}()
    @inbounds for i in 1:N
        nid = Int(nbs_id[i])
        nid > 0 || continue
        push!(get!(nbs_points, nid, Int[]), i)
    end

    isempty(nbs_points) && return (tree_nbs_id = tree_nbs_id, tree_id = tree_id)

    # (b) node_id → skeleton vertex index
    skel_node_ids = getattribute(skeleton_cloud, :node_id)
    n_skel = nv(graph_skeleton)
    node_to_skel = Dict{Int, Int}()
    sizehint!(node_to_skel, n_skel)
    for si in 1:n_skel
        node_to_skel[Int(skel_node_ids[si])] = si
    end

    # (c) skeleton vertex → NBS label
    skel_to_nbs = zeros(Int, n_skel)
    @inbounds for i in 1:N
        nid = Int(node_id[i])
        nid > 0 || continue
        sid = Int(nbs_id[i])
        sid > 0 || continue
        sv = get(node_to_skel, nid, 0)
        sv > 0 && (skel_to_nbs[sv] = sid)
    end

    # (d) NBS → skeleton vertex set
    nbs_skel_nodes = Dict{Int, Vector{Int}}()
    for sv in 1:n_skel
        nlab = skel_to_nbs[sv]
        nlab > 0 || continue
        push!(get!(nbs_skel_nodes, nlab, Int[]), sv)
    end

    # (e) NBS↔NBS adjacency via point-level graph (connection counts)
    nbs_adj = Dict{Int, Dict{Int, Int}}()
    @inbounds for e in Graphs.edges(graph)
        a = Int(nbs_id[src(e)])
        b = Int(nbs_id[dst(e)])
        (a > 0 && b > 0 && a != b) || continue
        inner_a = get!(nbs_adj, a, Dict{Int, Int}())
        inner_a[b] = get(inner_a, b, 0) + 1
        inner_b = get!(nbs_adj, b, Dict{Int, Int}())
        inner_b[a] = get(inner_b, a, 0) + 1
    end

    # ── Step 4.1: seed trees from near-ground NBS ────────────────
    seed_res = _seed_trees_from_nearground!(tree_id, nbs_points, agh_values, nearground_ceiling)
    nbs_tree     = seed_res.nbs_tree
    assigned_nbs = seed_res.assigned_nbs
    next_tree_id = seed_res.next_tree_id   # currently unused after extraction; kept for future hooks

    @info "Assembly: seeded $(next_tree_id - 1) trees from near-ground NBS"

    # ── Step 4.2: iterative growth via skeleton graph ────────────
    iteration = 0
    while true
        iteration += 1

        # 4.2a: for each assigned NBS, scan its skeleton neighbors for unassigned NBS
        frontier_info = Dict{Int, Dict{Int32, Int}}()   # frontier_nbs → Dict(tree_id → connection_count)
        for assigned_k in assigned_nbs
            skel_nodes_k = get(nbs_skel_nodes, assigned_k, Int[])
            for sv in skel_nodes_k
                for sn in Graphs.neighbors(graph_skeleton, sv)
                    neighbor_nbs = skel_to_nbs[sn]
                    (neighbor_nbs > 0 && !(neighbor_nbs in assigned_nbs)) || continue
                    tid = nbs_tree[assigned_k]
                    info = get!(frontier_info, neighbor_nbs, Dict{Int32, Int}())
                    # Tie-break by point-level connection count
                    adj_inner = get(nbs_adj, neighbor_nbs, Dict{Int, Int}())
                    conn_count = get(adj_inner, assigned_k, 0)
                    info[tid] = get(info, tid, 0) + conn_count
                end
            end
        end

        isempty(frontier_info) && break   # no frontier → terminate

        # Sort frontier NBS by number of points (large → small)
        frontier_sorted = sort!(collect(keys(frontier_info));
                                by = k -> -length(get(nbs_points, k, Int[])))

        n_assigned_this_round = 0

        for k in frontier_sorted
            k in assigned_nbs && continue   # may have been assigned earlier this round

            skel_nodes_k     = get(nbs_skel_nodes, k, Int[])
            tree_connections = frontier_info[k]
            n_total_nodes    = length(skel_nodes_k)

            # Count how many skeleton nodes in this NBS touch an already-assigned NBS
            n_nodes_with_tree_conn = 0
            for sv in skel_nodes_k
                for sn in Graphs.neighbors(graph_skeleton, sv)
                    nbr_nbs = skel_to_nbs[sn]
                    if nbr_nbs != k && nbr_nbs in assigned_nbs
                        n_nodes_with_tree_conn += 1
                        break   # count each node once
                    end
                end
            end

            # Best tree = most total point-level connections
            best_tree  = Int32(0)
            best_count = 0
            for (tid, cnt) in tree_connections
                if cnt > best_count
                    best_count = cnt
                    best_tree  = tid
                end
            end
            best_tree == 0 && continue   # shouldn't happen for a valid frontier

            frac_connected = n_total_nodes > 0 ? n_nodes_with_tree_conn / n_total_nodes : 0.0

            if frac_connected <= merge_threshold
                # Rule A: mostly internal nodes → assign as a new branch of best_tree
                nbs_tree[k] = best_tree
                push!(assigned_nbs, k)
                pts = get(nbs_points, k, Int[])
                @inbounds for i in pts
                    tree_id[i] = best_tree
                end
                n_assigned_this_round += 1
            else
                # Rule B: straddles existing tree NBS → merge into the closest one
                # by skeleton-edge count.
                skel_neighbor_counts = Dict{Int, Int}()
                for sv in skel_nodes_k
                    for sn in Graphs.neighbors(graph_skeleton, sv)
                        nbr_nbs = skel_to_nbs[sn]
                        (nbr_nbs > 0 && nbr_nbs != k) || continue
                        skel_neighbor_counts[nbr_nbs] = get(skel_neighbor_counts, nbr_nbs, 0) + 1
                    end
                end

                target_nbs   = 0
                target_count = 0
                for (nbr_nbs, cnt) in skel_neighbor_counts
                    if cnt > target_count
                        target_count = cnt
                        target_nbs   = nbr_nbs
                    end
                end

                if target_nbs > 0
                    target_tid = get(nbs_tree, target_nbs, Int32(0))
                    if target_tid == Int32(-1)
                        target_tid = Int32(0)   # target was a split NBS; treat as unassigned
                    end
                    pts = get(nbs_points, k, Int[])
                    @inbounds for i in pts
                        tree_id[i]     = target_tid
                        tree_nbs_id[i] = Int32(target_nbs)
                    end
                    nbs_tree[k] = target_tid
                    push!(assigned_nbs, k)
                    n_assigned_this_round += 1
                end
            end
        end

        n_assigned_this_round == 0 && break

        @info "Assembly iteration $iteration: assigned $n_assigned_this_round NBS" total_assigned=length(assigned_nbs)
    end

    # ── Step 4.3: re-order tree_nbs_id by descending group size ──────
    # Within this assemble_segments call, every non-zero tree_nbs_id value belongs
    # to exactly one tree: original NBS labels are unique 1..K, Rule A leaves them
    # untouched, and Rule B adopts the target NBS's (already-unique) label. So
    # keying by tree_nbs_id alone is equivalent to keying by (tree_id, tree_nbs_id)
    # and lets us delegate to `relabel_by_occurrence`.
    @inbounds for i in 1:N
        tree_id[i] == 0 && (tree_nbs_id[i] = Int32(0))
    end
    tree_nbs_id = relabel_by_occurrence(tree_nbs_id; positive_only=true, T_out=Int32)

    n_trees = length(unique(tid for tid in tree_id if tid > 0))
    n_assigned_pts = count(>(Int32(0)), tree_id)
    @info "Assembly complete" n_trees n_assigned_points=n_assigned_pts total_points=N iterations=iteration

    return (tree_nbs_id = tree_nbs_id, tree_id = tree_id)
end

"""
    _seed_trees_from_nearground!(tree_id, nbs_points, agh_values, nearground_ceiling)
        -> NamedTuple

For each NBS whose minimum AGH is at or below `nearground_ceiling`, assign a fresh
tree id (1, 2, …) and mutate `tree_id` to that value for every point in the NBS.
Returns `(nbs_tree::Dict{Int,Int32}, assigned_nbs::Set{Int}, next_tree_id::Int32)`.
"""
function _seed_trees_from_nearground!(tree_id::AbstractVector{Int32},
                                      nbs_points::AbstractDict{Int,Vector{Int}},
                                      agh_values::AbstractVector{<:Real},
                                      nearground_ceiling::Real)
    next_tree_id = Int32(1)
    nbs_tree     = Dict{Int, Int32}()
    assigned_nbs = Set{Int}()
    ceiling_f64  = Float64(nearground_ceiling)

    for (k, pts) in nbs_points
        min_agh = Inf
        @inbounds for i in pts
            v = Float64(agh_values[i])
            v < min_agh && (min_agh = v)
        end
        if min_agh <= ceiling_f64
            nbs_tree[k] = next_tree_id
            push!(assigned_nbs, k)
            @inbounds for i in pts
                tree_id[i] = next_tree_id
            end
            next_tree_id += Int32(1)
        end
    end

    return (nbs_tree=nbs_tree, assigned_nbs=assigned_nbs, next_tree_id=next_tree_id)
end

# ── Step 5: orphan NBS rescue ─────────────────────────────────────

"""
    process_orphan_segments(coords, nbs_id, tree_id, tree_nbs_id; cfg) -> nothing

Rescue orphan NBS that the per-component assembly never assigned to a tree.
Builds a coarse NBS-level graph from two sources:
  1. Radius graph among orphan points (`tree_assembly_occlusion_tolerance`)
     — orphan↔orphan edges
  2. KDTree search from orphan points to already-assigned points
     — orphan↔tree edges

then iteratively propagates `tree_id` through the coarse graph. Mutates
`tree_id` and `tree_nbs_id` in place.
"""
function process_orphan_segments(
    coords::AbstractMatrix{<:Real},
    nbs_id::AbstractVector{<:Integer},
    tree_id::AbstractVector{Int32},
    tree_nbs_id::AbstractVector{Int32};
    cfg::FLiPConfig = _CFG,
)
    N = size(coords, 1)
    occlusion_tol = cfg.tree_assembly_occlusion_tolerance
    occlusion_tol > 0 || return nothing

    # Identify orphan NBS (nbs_id > 0, all points unassigned)
    orphan_nbs_points = Dict{Int, Vector{Int}}()
    @inbounds for i in 1:N
        nid = Int(nbs_id[i])
        nid > 0 && tree_id[i] == 0 || continue
        push!(get!(orphan_nbs_points, nid, Int[]), i)
    end
    # Remove NBS that have any assigned points (partially assigned ≠ orphan)
    @inbounds for i in 1:N
        nid = Int(nbs_id[i])
        nid > 0 && tree_id[i] > 0 && delete!(orphan_nbs_points, nid)
    end

    isempty(orphan_nbs_points) && return nothing

    orphan_nbs_list = collect(keys(orphan_nbs_points))
    n_orphan_nbs    = length(orphan_nbs_list)

    # Collect orphan point indices
    orphan_pt_idx = Int[]
    for pts in values(orphan_nbs_points)
        append!(orphan_pt_idx, pts)
    end
    n_orphan_pts = length(orphan_pt_idx)

    # ── Source 1: radius graph among orphan points ───────────────
    @info "[tree_segmentation] orphan rescue: building radius graph for $n_orphan_pts orphan points (r=$occlusion_tol m)"
    orphan_coords = coords[orphan_pt_idx, :]
    orphan_graph  = build_radius_graph(orphan_coords, occlusion_tol).graph

    orphan_nbs_of_pt = zeros(Int, n_orphan_pts)
    @inbounds for (j, i) in enumerate(orphan_pt_idx)
        orphan_nbs_of_pt[j] = Int(nbs_id[i])
    end

    coarse_adj = Dict{Int, Dict{Int, Int}}()   # nbs_a → nbs_b → count
    @inbounds for e in Graphs.edges(orphan_graph)
        a = orphan_nbs_of_pt[src(e)]
        b = orphan_nbs_of_pt[dst(e)]
        (a > 0 && b > 0 && a != b) || continue
        inner_a = get!(coarse_adj, a, Dict{Int, Int}())
        inner_a[b] = get(inner_a, b, 0) + 1
        inner_b = get!(coarse_adj, b, Dict{Int, Int}())
        inner_b[a] = get(inner_b, a, 0) + 1
    end

    # ── Source 2: KDTree search from orphan points to assigned points ─
    assigned_idx = Int[]
    @inbounds for i in 1:N
        tree_id[i] > 0 && push!(assigned_idx, i)
    end

    orphan_to_tree_nbs = Dict{Int, Dict{Int32, Dict{Int32, Int}}}()  # orph_nbs → tid → tnid → count

    if !isempty(assigned_idx)
        assigned_3xM = Matrix{eltype(coords)}(undef, 3, length(assigned_idx))
        @inbounds for (j, i) in enumerate(assigned_idx)
            assigned_3xM[1, j] = coords[i, 1]
            assigned_3xM[2, j] = coords[i, 2]
            assigned_3xM[3, j] = coords[i, 3]
        end
        kdtree = KDTree(assigned_3xM)
        @info "[tree_segmentation] orphan rescue: KDTree query for orphan→tree connections"

        # Single pass through orphan points populates BOTH `coarse_adj` (sentinel -tid)
        # AND `orphan_to_tree_nbs`. Previously this was two passes with duplicate
        # `inrange` calls — fused for ~2× fewer KDTree queries on the orphan set.
        for (orph_nbs, orph_pts) in orphan_nbs_points
            tnbs_by_tid = Dict{Int32, Dict{Int32, Int}}()
            for i in orph_pts
                query = SVector{3, Float64}(coords[i, 1], coords[i, 2], coords[i, 3])
                hits = inrange(kdtree, query, occlusion_tol)
                for j in hits
                    aid  = assigned_idx[j]
                    tid  = tree_id[aid]
                    tnid = tree_nbs_id[aid]
                    tid > 0 || continue
                    inner = get!(coarse_adj, orph_nbs, Dict{Int,Int}())
                    inner[-tid] = get(inner, -tid, 0) + 1
                    tnid > 0 || continue
                    tnbs_inner = get!(tnbs_by_tid, tid, Dict{Int32, Int}())
                    tnbs_inner[tnid] = get(tnbs_inner, tnid, 0) + 1
                end
            end
            isempty(tnbs_by_tid) || (orphan_to_tree_nbs[orph_nbs] = tnbs_by_tid)
        end
        # `kdtree` and `assigned_3xM` go out of scope at the `end` of this `if`.
    end

    # Release Step-5 intermediates before the propagation loop allocates new dicts.
    orphan_graph     = SimpleGraph{Int}(0)
    orphan_coords    = zeros(eltype(coords), 0, 3)
    empty!(orphan_nbs_of_pt)
    empty!(orphan_pt_idx)
    empty!(assigned_idx)

    # ── Iterative propagation on coarse graph ─────────────────────
    orphan_tree_id  = Dict{Int, Int32}()
    orphan_assigned = Set{Int}()

    orphan_iteration = 0
    while true
        orphan_iteration += 1
        n_rescued = 0

        unassigned = filter(k -> !(k in orphan_assigned), orphan_nbs_list)
        sort!(unassigned; by = k -> -length(orphan_nbs_points[k]))

        for orph_nbs in unassigned
            adj = get(coarse_adj, orph_nbs, Dict{Int, Int}())
            isempty(adj) && continue

            votes = Dict{Int32, Int}()
            for (nbr, cnt) in adj
                if nbr < 0
                    votes[Int32(-nbr)] = get(votes, Int32(-nbr), 0) + cnt
                elseif nbr in orphan_assigned
                    tid = get(orphan_tree_id, nbr, Int32(0))
                    tid > 0 || continue
                    votes[tid] = get(votes, tid, 0) + cnt
                end
            end
            isempty(votes) && continue

            best_tid = Int32(0)
            best_cnt = 0
            for (tid, cnt) in votes
                if cnt > best_cnt
                    best_cnt = cnt
                    best_tid = tid
                end
            end

            orphan_tree_id[orph_nbs] = best_tid
            push!(orphan_assigned, orph_nbs)
            n_rescued += 1
        end

        n_rescued == 0 && break
        @info "[tree_segmentation] orphan rescue iteration $orphan_iteration: rescued $n_rescued NBS"
    end

    # Apply orphan assignments to point-level arrays.
    # `next_fresh_tnbs` allocates globally-unique labels for orphans whose `best_tid`
    # has no nearby NBS (rescued purely via orphan→orphan propagation).
    next_fresh_tnbs = Int32(maximum(tree_nbs_id; init = Int32(0))) + Int32(1)

    for (orph_nbs, best_tid) in orphan_tree_id
        best_tid > 0 || continue

        tnbs_by_tid = get(orphan_to_tree_nbs, orph_nbs, Dict{Int32, Dict{Int32, Int}}())
        tnbs_counts = get(tnbs_by_tid, best_tid, Dict{Int32, Int}())

        best_tnbs     = Int32(0)
        best_tnbs_cnt = 0
        for (tnid, cnt) in tnbs_counts
            if cnt > best_tnbs_cnt
                best_tnbs_cnt = cnt
                best_tnbs     = tnid
            end
        end

        if best_tnbs == 0
            best_tnbs = next_fresh_tnbs
            next_fresh_tnbs += Int32(1)
        end

        for i in orphan_nbs_points[orph_nbs]
            tree_id[i]     = best_tid
            tree_nbs_id[i] = best_tnbs
        end
    end

    return nothing
end

# ── Step 7: skeleton OBJ writer ───────────────────────────────────

"""
    _write_polyline_obj(path, coords, graph) -> nothing

Write a skeleton graph as an OBJ polyline file readable by CloudCompare.
Every edge is written as a separate `l u v` statement (1-indexed vertices).
"""
function _write_polyline_obj(path::AbstractString, coords::AbstractMatrix{<:Real},
                             graph::SimpleGraph{Int})
    open(path, "w") do io
        println(io, "# FLiP.jl skeleton graph")
        n = size(coords, 1)
        for i in 1:n
            println(io, "v $(Float64(coords[i, 1])) $(Float64(coords[i, 2])) $(Float64(coords[i, 3]))")
        end
        for e in Graphs.edges(graph)
            println(io, "l $(src(e)) $(dst(e))")
        end
    end
    return nothing
end
