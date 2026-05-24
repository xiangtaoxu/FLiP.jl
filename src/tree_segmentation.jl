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

    # Build component → indices dispatch in one O(N) pass. `connected_component_labels`
    # already ranks valid components as dense 1..K_cc (and 0 for below-min-size), so a
    # Vector indexed by component id is the natural container — no hashing.
    K_cc = Int(maximum(cc_labels; init=0))
    cc_indices_by_id = [Int[] for _ in 1:K_cc]
    @inbounds for i in eachindex(cc_labels)
        lab = cc_labels[i]
        lab > 0 && push!(cc_indices_by_id[lab], i)
    end
    cc_labels = Int[]   # free per-point label vector (~ 8N bytes)

    n_components = K_cc
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

    for (ci, cc_indices) in enumerate(cc_indices_by_id)
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

    empty!(cc_indices_by_id)   # release after per-component loop

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
    g_res    = build_radius_graph(cc_coords, neighbor_radius; weights=false)
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

    K_nbs    = Int(maximum(nbs_id;  init=0))
    max_node = Int(maximum(node_id; init=0))
    K_nbs == 0 && return (tree_nbs_id = tree_nbs_id, tree_id = tree_id)

    # ── Step 4.0: precomputations ────────────────────────────────
    info = _init_assembly_info(graph, nbs_id, node_id, graph_skeleton, skeleton_cloud,
                               K_nbs, max_node)
    nbs_points     = info.nbs_points
    skel_to_nbs    = info.skel_to_nbs
    nbs_skel_nodes = info.nbs_skel_nodes
    nbs_adj        = info.nbs_adj

    # ── Step 4.1: seed trees from near-ground NBS ────────────────
    seed_res = _seed_trees_from_nearground!(tree_id, nbs_points, agh_values, nearground_ceiling)
    nbs_tree     = seed_res.nbs_tree       # Vector{Int32}  (0 = unassigned)
    assigned_nbs = seed_res.assigned_nbs   # BitVector
    next_tree_id = seed_res.next_tree_id

    @info "Assembly: seeded $(next_tree_id - 1) trees from near-ground NBS"

    # ── Step 4.2: iterative growth via skeleton graph ────────────
    # `frontier_info` is built once and maintained INCREMENTALLY: each time an NBS
    # transitions to assigned we add its contributions to its unassigned skeleton
    # neighbors and remove it from the frontier set. Updates from within an iteration
    # are DEFERRED and applied after the iteration to match the original
    # rebuild-each-iteration semantics (later k's in the same round don't see earlier
    # k's contributions, matching the snapshot-and-process model).
    frontier_info = Dict{Int, Dict{Int32, Int}}()   # frontier_nbs → Dict(tree_id → connection_count)
    for k in 1:K_nbs
        assigned_nbs[k] || continue
        _update_frontier_after_assign!(frontier_info, k, nbs_tree[k],
                                       nbs_skel_nodes, skel_to_nbs, assigned_nbs,
                                       nbs_adj, graph_skeleton)
    end

    pending_assignments = Tuple{Int,Int32}[]
    iteration = 0
    while !isempty(frontier_info)
        iteration += 1

        # Sort frontier NBS by number of points (large → small)
        frontier_sorted = sort!(collect(keys(frontier_info));
                                by = k -> -length(nbs_points[k]))
        empty!(pending_assignments)
        n_assigned_this_round = 0

        for k in frontier_sorted
            assigned_nbs[k] && continue   # may have been assigned earlier this round

            skel_nodes_k     = nbs_skel_nodes[k]
            tree_connections = frontier_info[k]
            n_total_nodes    = length(skel_nodes_k)

            # Fused single pass over skel_nodes_k × skeleton neighbors that computes
            # BOTH `n_nodes_with_tree_conn` (for Rule A/B branch test) and
            # `skel_neighbor_counts` (for Rule B target selection). Was two separate
            # passes before — same total cost regardless of which branch fires.
            skel_neighbor_counts = Dict{Int, Int}()
            n_nodes_with_tree_conn = 0
            for sv in skel_nodes_k
                has_tree_conn = false
                for sn in Graphs.neighbors(graph_skeleton, sv)
                    nbr_nbs = skel_to_nbs[sn]
                    (nbr_nbs > 0 && nbr_nbs != k) || continue
                    skel_neighbor_counts[nbr_nbs] = get(skel_neighbor_counts, nbr_nbs, 0) + 1
                    if !has_tree_conn && assigned_nbs[nbr_nbs]
                        n_nodes_with_tree_conn += 1
                        has_tree_conn = true
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
                nbs_tree[k]     = best_tree
                assigned_nbs[k] = true
                @inbounds for i in nbs_points[k]
                    tree_id[i] = best_tree
                end
                push!(pending_assignments, (k, best_tree))
                n_assigned_this_round += 1
            else
                # Rule B: straddles existing tree NBS → merge into the closest one
                # by skeleton-edge count. Reuses the `skel_neighbor_counts` from the
                # fused loop above.
                target_nbs   = 0
                target_count = 0
                for (nbr_nbs, cnt) in skel_neighbor_counts
                    if cnt > target_count
                        target_count = cnt
                        target_nbs   = nbr_nbs
                    end
                end

                if target_nbs > 0
                    target_tid = nbs_tree[target_nbs]
                    if target_tid == Int32(-1)
                        target_tid = Int32(0)   # target was a split NBS; treat as unassigned
                    end
                    @inbounds for i in nbs_points[k]
                        tree_id[i]     = target_tid
                        tree_nbs_id[i] = Int32(target_nbs)
                    end
                    nbs_tree[k]     = target_tid
                    assigned_nbs[k] = true
                    push!(pending_assignments, (k, target_tid))
                    n_assigned_this_round += 1
                end
            end
        end

        # Defer frontier updates until iteration end so within-iteration assignments
        # don't influence later k's in the same round (matches original rebuild-fresh
        # behavior; only `assigned_nbs` and per-point `tree_id` update live).
        for (k, tid) in pending_assignments
            _update_frontier_after_assign!(frontier_info, k, tid,
                                           nbs_skel_nodes, skel_to_nbs, assigned_nbs,
                                           nbs_adj, graph_skeleton)
        end

        n_assigned_this_round == 0 && break

        @info "Assembly iteration $iteration: assigned $n_assigned_this_round NBS" total_assigned=count(assigned_nbs)
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
    _init_assembly_info(graph, nbs_id, node_id, graph_skeleton, skeleton_cloud,
                        K_nbs, max_node) -> NamedTuple

Build the index structures consumed by `assemble_segments`:

- `nbs_points::Vector{Vector{Int}}` — NBS id → point indices
- `node_to_skel::Vector{Int}` — node id → skeleton vertex (0 = not in skeleton)
- `skel_to_nbs::Vector{Int}` — skeleton vertex → NBS label
- `nbs_skel_nodes::Vector{Vector{Int}}` — NBS id → skeleton vertex set
- `nbs_adj::SparseMatrixCSC{Int,Int}` — symmetric K_nbs×K_nbs adjacency-count matrix

NBS labels and node ids are both dense within a component (1..K_nbs and 1..max_node
respectively), so each container is Vector-indexed for O(1) hash-free lookup. The
adjacency uses `SparseMatrixCSC` so missing-pair lookups return `0` without any
explicit `get(.., 0)` wrapper.
"""
function _init_assembly_info(graph::SimpleGraph{Int},
                             nbs_id::AbstractVector{<:Integer},
                             node_id::AbstractVector{<:Integer},
                             graph_skeleton::SimpleGraph{Int},
                             skeleton_cloud::PointCloud,
                             K_nbs::Int, max_node::Int)
    N      = length(nbs_id)
    n_skel = nv(graph_skeleton)

    # (a) nbs_id → point indices
    nbs_points = [Int[] for _ in 1:K_nbs]
    @inbounds for i in 1:N
        nid = Int(nbs_id[i])
        nid > 0 && push!(nbs_points[nid], i)
    end

    # (b) node_id → skeleton vertex index
    skel_node_ids = getattribute(skeleton_cloud, :node_id)
    node_to_skel  = zeros(Int, max_node)
    @inbounds for si in 1:n_skel
        nid = Int(skel_node_ids[si])
        (1 <= nid <= max_node) && (node_to_skel[nid] = si)
    end

    # (c) skeleton vertex → NBS label
    skel_to_nbs = zeros(Int, n_skel)
    @inbounds for i in 1:N
        nid = Int(node_id[i]); sid = Int(nbs_id[i])
        (nid > 0 && sid > 0) || continue
        sv = (1 <= nid <= max_node) ? node_to_skel[nid] : 0
        sv > 0 && (skel_to_nbs[sv] = sid)
    end

    # (d) NBS → skeleton vertex set
    nbs_skel_nodes = [Int[] for _ in 1:K_nbs]
    for sv in 1:n_skel
        nlab = skel_to_nbs[sv]
        nlab > 0 && push!(nbs_skel_nodes[nlab], sv)
    end

    # (e) NBS↔NBS adjacency as a symmetric SparseMatrixCSC.
    adj_rows = Int[]; adj_cols = Int[]; adj_vals = Int[]
    @inbounds for e in Graphs.edges(graph)
        a = Int(nbs_id[src(e)]); b = Int(nbs_id[dst(e)])
        (a > 0 && b > 0 && a != b) || continue
        push!(adj_rows, a); push!(adj_cols, b); push!(adj_vals, 1)
        push!(adj_rows, b); push!(adj_cols, a); push!(adj_vals, 1)
    end
    nbs_adj = sparse(adj_rows, adj_cols, adj_vals, K_nbs, K_nbs, +)

    return (nbs_points    = nbs_points,
            node_to_skel  = node_to_skel,
            skel_to_nbs   = skel_to_nbs,
            nbs_skel_nodes = nbs_skel_nodes,
            nbs_adj       = nbs_adj)
end

"""
    _seed_trees_from_nearground!(tree_id, nbs_points, agh_values, nearground_ceiling)
        -> NamedTuple

For each NBS whose minimum AGH is at or below `nearground_ceiling`, assign a fresh
tree id (1, 2, …) and mutate `tree_id` to that value for every point in the NBS.

`nbs_points::Vector{Vector{Int}}` is indexed by dense NBS id 1..K_nbs (empty entries
are allowed for ids not produced by `label_non_branching_segments`). Returns
`(nbs_tree::Vector{Int32}, assigned_nbs::BitVector, next_tree_id::Int32)`, both
indexed by NBS id (`nbs_tree[k] == 0` means unassigned).
"""
function _seed_trees_from_nearground!(tree_id::AbstractVector{Int32},
                                      nbs_points::Vector{Vector{Int}},
                                      agh_values::AbstractVector{<:Real},
                                      nearground_ceiling::Real)
    K_nbs        = length(nbs_points)
    next_tree_id = Int32(1)
    nbs_tree     = zeros(Int32, K_nbs)
    assigned_nbs = falses(K_nbs)
    ceiling_f64  = Float64(nearground_ceiling)

    @inbounds for k in 1:K_nbs
        pts = nbs_points[k]
        isempty(pts) && continue
        min_agh = Inf
        for i in pts
            v = Float64(agh_values[i])
            v < min_agh && (min_agh = v)
        end
        if min_agh <= ceiling_f64
            nbs_tree[k]     = next_tree_id
            assigned_nbs[k] = true
            for i in pts
                tree_id[i] = next_tree_id
            end
            next_tree_id += Int32(1)
        end
    end

    return (nbs_tree=nbs_tree, assigned_nbs=assigned_nbs, next_tree_id=next_tree_id)
end

"""
    _update_frontier_after_assign!(frontier_info, k, tid, nbs_skel_nodes,
                                   skel_to_nbs, assigned_nbs, nbs_adj,
                                   graph_skeleton) -> nothing

After NBS `k` transitions to assigned (tree id `tid`), remove `k` from `frontier_info`
and add its contribution to every currently-unassigned skeleton neighbor's frontier
entry. When `tid == 0` (Rule B merge into a split target with no valid tree), only
the removal happens — no tree votes are recorded.
"""
function _update_frontier_after_assign!(frontier_info::Dict{Int,Dict{Int32,Int}},
                                        k::Int, tid::Int32,
                                        nbs_skel_nodes::Vector{Vector{Int}},
                                        skel_to_nbs::Vector{Int},
                                        assigned_nbs::BitVector,
                                        nbs_adj::SparseMatrixCSC,
                                        graph_skeleton::SimpleGraph{Int})
    delete!(frontier_info, k)
    tid > 0 || return nothing

    @inbounds for sv in nbs_skel_nodes[k]
        for sn in Graphs.neighbors(graph_skeleton, sv)
            nbr_nbs = skel_to_nbs[sn]
            (nbr_nbs > 0 && !assigned_nbs[nbr_nbs]) || continue
            conn_count = nbs_adj[nbr_nbs, k]
            info = get!(frontier_info, nbr_nbs, Dict{Int32,Int}())
            info[tid] = get(info, tid, 0) + conn_count
        end
    end
    return nothing
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

    K_nbs_global = Int(maximum(nbs_id; init=0))
    K_nbs_global == 0 && return nothing

    # ── Step 5.0: identify orphan NBS in a single pass ──────────
    # An NBS is "orphan" iff at least one point belongs to it AND none of its
    # points has been assigned a tree. One O(N) sweep collects candidate points
    # per NBS and flags any NBS that has at least one already-assigned point.
    has_assigned = falses(K_nbs_global)
    cand_points  = [Int[] for _ in 1:K_nbs_global]
    @inbounds for i in 1:N
        nid = Int(nbs_id[i])
        nid > 0 || continue
        if tree_id[i] > 0
            has_assigned[nid] = true
        else
            push!(cand_points[nid], i)
        end
    end

    # Build dense orphan index 1..K_orphan, keeping only NBS that survived the
    # "any-assigned" filter and have at least one candidate point.
    orphan_nbs_ids    = Int[]
    orphan_nbs_points = Vector{Vector{Int}}()
    for nid in 1:K_nbs_global
        (has_assigned[nid] || isempty(cand_points[nid])) && continue
        push!(orphan_nbs_ids, nid)
        push!(orphan_nbs_points, cand_points[nid])
    end
    K_orphan = length(orphan_nbs_ids)
    K_orphan == 0 && return nothing

    # Reverse map: NBS id → orphan_idx (0 = not an orphan)
    orphan_idx_of_nbs = zeros(Int, K_nbs_global)
    @inbounds for (idx, nid) in enumerate(orphan_nbs_ids)
        orphan_idx_of_nbs[nid] = idx
    end

    # Pre-sized orphan point index vector
    n_orphan_pts  = sum(length, orphan_nbs_points; init=0)
    orphan_pt_idx = Vector{Int}(undef, n_orphan_pts)
    pt_pos = 0
    @inbounds for pts in orphan_nbs_points
        for i in pts
            pt_pos += 1
            orphan_pt_idx[pt_pos] = i
        end
    end

    # ── Source 1: radius graph among orphan points ───────────────
    @info "[tree_segmentation] orphan rescue: building radius graph for $n_orphan_pts orphan points (r=$occlusion_tol m)"
    orphan_coords = coords[orphan_pt_idx, :]
    orphan_graph  = build_radius_graph(orphan_coords, occlusion_tol; weights=false).graph

    # Map orphan-graph vertex → orphan_idx (dense 1..K_orphan).
    orphan_idx_of_pt = zeros(Int, n_orphan_pts)
    @inbounds for j in 1:n_orphan_pts
        nid = Int(nbs_id[orphan_pt_idx[j]])
        orphan_idx_of_pt[j] = orphan_idx_of_nbs[nid]
    end

    # coarse_o2o[idx] = Dict(neighbor_idx → connection_count). Dense outer Vector,
    # inner stays as a Dict (orphan neighborhood degrees are typically small).
    coarse_o2o = [Dict{Int,Int}() for _ in 1:K_orphan]
    @inbounds for e in Graphs.edges(orphan_graph)
        ia = orphan_idx_of_pt[src(e)]
        ib = orphan_idx_of_pt[dst(e)]
        (ia > 0 && ib > 0 && ia != ib) || continue
        coarse_o2o[ia][ib] = get(coarse_o2o[ia], ib, 0) + 1
        coarse_o2o[ib][ia] = get(coarse_o2o[ib], ia, 0) + 1
    end

    # ── Source 2: KDTree search from orphan points to assigned points ─
    # Separate orphan→tree (votes) and orphan→(tree, tnid) (apply-step tnid
    # selection). Replaces the prior triple-nested Dict and the `-tid` sentinel
    # mixed-namespace hack.
    coarse_o2t            = [Dict{Int32,Int}()              for _ in 1:K_orphan]
    orphan_to_tree_nbs_v  = [Dict{Tuple{Int32,Int32},Int}() for _ in 1:K_orphan]

    assigned_idx = Int[]
    @inbounds for i in 1:N
        tree_id[i] > 0 && push!(assigned_idx, i)
    end

    if !isempty(assigned_idx)
        assigned_3xM = Matrix{eltype(coords)}(undef, 3, length(assigned_idx))
        @inbounds for (j, i) in enumerate(assigned_idx)
            assigned_3xM[1, j] = coords[i, 1]
            assigned_3xM[2, j] = coords[i, 2]
            assigned_3xM[3, j] = coords[i, 3]
        end
        kdtree = KDTree(assigned_3xM)
        @info "[tree_segmentation] orphan rescue: KDTree query for orphan→tree connections"

        # Single pass populates both coarse_o2t and orphan_to_tree_nbs_v.
        @inbounds for orph_idx in 1:K_orphan
            orph_pts = orphan_nbs_points[orph_idx]
            o2t = coarse_o2t[orph_idx]
            o2tn = orphan_to_tree_nbs_v[orph_idx]
            for i in orph_pts
                query = SVector{3, Float64}(coords[i, 1], coords[i, 2], coords[i, 3])
                hits = inrange(kdtree, query, occlusion_tol)
                for j in hits
                    aid  = assigned_idx[j]
                    tid  = tree_id[aid]
                    tnid = tree_nbs_id[aid]
                    tid > 0 || continue
                    o2t[tid] = get(o2t, tid, 0) + 1
                    tnid > 0 || continue
                    key = (tid, tnid)
                    o2tn[key] = get(o2tn, key, 0) + 1
                end
            end
        end
        # kdtree, assigned_3xM go out of scope at end of this `if`.
    end

    # Release Step-5 intermediates before the propagation loop allocates new dicts.
    orphan_graph  = SimpleGraph{Int}(0)
    orphan_coords = zeros(eltype(coords), 0, 3)
    empty!(orphan_idx_of_pt)
    empty!(orphan_pt_idx)
    empty!(assigned_idx)
    empty!(has_assigned)
    empty!(cand_points)
    empty!(orphan_idx_of_nbs)

    # ── Iterative propagation on coarse graph ─────────────────────
    orphan_tree_id = _propagate_orphan_labels(coarse_o2o, coarse_o2t, orphan_nbs_points)

    # ── Apply orphan assignments ─────────────────────────────────
    # `next_fresh_tnbs` allocates globally-unique labels for orphans whose `best_tid`
    # has no nearby NBS (rescued purely via orphan→orphan propagation).
    next_fresh_tnbs = Int32(maximum(tree_nbs_id; init = Int32(0))) + Int32(1)

    for orph_idx in 1:K_orphan
        best_tid = orphan_tree_id[orph_idx]
        best_tid > 0 || continue

        # Best tnid within best_tid (filter the flattened (tid, tnid) → count dict)
        best_tnbs     = Int32(0)
        best_tnbs_cnt = 0
        for ((tid, tnid), cnt) in orphan_to_tree_nbs_v[orph_idx]
            tid == best_tid || continue
            if cnt > best_tnbs_cnt
                best_tnbs_cnt = cnt
                best_tnbs     = tnid
            end
        end

        if best_tnbs == 0
            best_tnbs = next_fresh_tnbs
            next_fresh_tnbs += Int32(1)
        end

        @inbounds for i in orphan_nbs_points[orph_idx]
            tree_id[i]     = best_tid
            tree_nbs_id[i] = best_tnbs
        end
    end

    return nothing
end

"""
    _propagate_orphan_labels(coarse_o2o, coarse_o2t, orphan_nbs_points) -> Vector{Int32}

Run the iterative tree-id propagation on the coarse orphan graph. At each
iteration every still-unassigned orphan votes by summing connections to
(a) directly-known trees via `coarse_o2t[idx] :: Dict(tid → count)` and
(b) already-rescued orphans via `coarse_o2o[idx] :: Dict(neighbor_idx → count)`.
The most-voted tree id wins. Iteration stops when a round rescues nothing.

Returns a `Vector{Int32}` of length `K_orphan` where entry `i` is the assigned
tree id for orphan index `i` (0 = never rescued).
"""
function _propagate_orphan_labels(coarse_o2o::Vector{Dict{Int,Int}},
                                  coarse_o2t::Vector{Dict{Int32,Int}},
                                  orphan_nbs_points::Vector{Vector{Int}})
    K_orphan = length(coarse_o2o)
    orphan_tree_id  = zeros(Int32, K_orphan)
    orphan_assigned = falses(K_orphan)
    work_buf        = Vector{Int}(undef, K_orphan)

    iteration = 0
    while true
        iteration += 1
        # Snapshot unassigned orphans into work_buf, sort by descending point count
        n = 0
        @inbounds for i in 1:K_orphan
            if !orphan_assigned[i]
                n += 1; work_buf[n] = i
            end
        end
        n == 0 && break
        unassigned = view(work_buf, 1:n)
        sort!(unassigned; by = i -> -length(orphan_nbs_points[i]))

        n_rescued = 0
        for orph_idx in unassigned
            votes = Dict{Int32, Int}()
            # Direct tree edges (always counted)
            for (tid, cnt) in coarse_o2t[orph_idx]
                votes[tid] = get(votes, tid, 0) + cnt
            end
            # Orphan-orphan edges contribute only via rescued neighbors
            for (nbr_idx, cnt) in coarse_o2o[orph_idx]
                orphan_assigned[nbr_idx] || continue
                tid = orphan_tree_id[nbr_idx]
                tid > 0 || continue
                votes[tid] = get(votes, tid, 0) + cnt
            end
            isempty(votes) && continue

            best_tid = Int32(0); best_cnt = 0
            for (tid, cnt) in votes
                if cnt > best_cnt
                    best_cnt = cnt
                    best_tid = tid
                end
            end
            orphan_tree_id[orph_idx]  = best_tid
            orphan_assigned[orph_idx] = true
            n_rescued += 1
        end
        n_rescued == 0 && break
        @info "[tree_segmentation] orphan rescue iteration $iteration: rescued $n_rescued NBS"
    end
    return orphan_tree_id
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
