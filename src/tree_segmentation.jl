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
5. Rescue orphan branches (`tree_id==0 && tree_nbs_id>0`) into neighboring grounded
   trees across occlusion gaps ([`assemble_occluded_segments`](@ref)).
6. Re-order tree IDs by descending point count (largest tree → 1), then make
   `tree_nbs_id` contiguous within each tree.
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
    threshold = cfg.tree_segmentation.nearground_agh_threshold
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

    neighbor_radius = cfg.tree_segmentation.neighbor_radius > 0 ?
                      cfg.tree_segmentation.neighbor_radius :
                      2.0 * cfg.pipeline.subsample_res
    neighbor_radius > 0 || throw(ArgumentError("tree neighbor radius must be > 0"))

    # ── Step 2: discover components via coordinate-only union-find ──
    cc_labels = connected_component_labels(coords_filtered, neighbor_radius,
                                           cfg.tree_segmentation.min_nbs_size)

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
    @info "$_LOG_PREFIX   $N filtered points → $n_components connected components (min_cc=$(cfg.tree_segmentation.min_nbs_size))"

    n_components == 0 && return empty_result

    # ── Step 3: process each component independently ──────────────
    want_skeleton = cfg.pipeline.enable_skeleton_output

    global_nbs_id      = zeros(Int32, N)
    global_node_id     = zeros(Int32, N)
    global_tree_id     = zeros(Int32, N)
    global_tree_nbs_id = zeros(Int32, N)

    # Per-component skeleton holders — only allocated/populated when skeleton
    # output is requested. The per-component skeleton is always computed inside
    # `_process_single_connected_component` (it drives assembly); these slots
    # just retain it for the cross-component merge below.
    skel_clouds = want_skeleton ? Vector{PointCloud}(undef, n_components) : PointCloud[]
    skel_graphs = want_skeleton ? Vector{SimpleGraph{Int}}(undef, n_components) : SimpleGraph{Int}[]

    # Per-CC progress reported as % of cumulative points processed. The reporter
    # is thread-safe (atomic CAS); the counter is a shared `Threads.Atomic{Int}`
    # so it remains correct under `_parallel_for`.
    progress = ProgressReporter("processing components", N)
    processed_points = Threads.Atomic{Int}(0)

    # Global id-range allocators. Each component reserves a contiguous, gap-free
    # block via `atomic_add!` (which returns the pre-increment value), then remaps
    # its local 1..local_max ids into that block. Reservation order is completion
    # order under threads — harmless because every downstream consumer groups by
    # label *value* (and `tree_id` is relabelled by point count in Step 6): the
    # point→group partition is identical regardless of which component owns which
    # block. `tree_nbs_id` shares the nbs block (same offset as `nbs_id`).
    nbs_counter  = Threads.Atomic{Int}(0)
    node_counter = Threads.Atomic{Int}(0)
    tree_counter = Threads.Atomic{Int}(0)

    # Phase 2 (parallel): run the per-component pipeline AND remap local→global ids
    # within the same task, then drop the per-point arrays immediately (no
    # `Vector{Any}` retention of all results). Writes target this component's
    # `gi ∈ cc_indices`; those index sets are disjoint across components, so the
    # shared `global_*` arrays are written race-free.
    _parallel_for(n_components, effective_nthreads(cfg)) do ci
        @inbounds begin
            cc_indices = cc_indices_by_id[ci]
            cc_coords  = coords_filtered[cc_indices, :]
            cc_agh     = agh_filtered[cc_indices]
            res = _process_single_connected_component(cc_coords, cc_agh,
                                                      neighbor_radius; cfg=cfg)

            local_nbs_max  = Int(maximum(res.nbs_id;  init=Int32(0)))
            local_node_max = Int(maximum(res.node_id; init=Int32(0)))
            local_tree_max = Int(maximum(res.tree_id; init=Int32(0)))
            my_nbs_off  = Int32(Threads.atomic_add!(nbs_counter,  local_nbs_max))
            my_node_off = Int32(Threads.atomic_add!(node_counter, local_node_max))
            my_tree_off = Int32(Threads.atomic_add!(tree_counter, local_tree_max))

            for (li, gi) in enumerate(cc_indices)
                nid  = res.nbs_id[li]
                global_nbs_id[gi]      = nid  > 0 ? nid  + my_nbs_off  : Int32(0)
                noid = res.node_id[li]
                global_node_id[gi]     = noid > 0 ? noid + my_node_off : Int32(0)
                tid  = res.tree_id[li]
                global_tree_id[gi]     = tid  > 0 ? tid  + my_tree_off : Int32(0)
                tnid = res.tree_nbs_id[li]
                global_tree_nbs_id[gi] = tnid > 0 ? tnid + my_nbs_off  : Int32(0)
            end

            if want_skeleton
                skel_clouds[ci] = res.skeleton_cloud
                skel_graphs[ci] = res.graph_skeleton
            end

            n_now = Threads.atomic_add!(processed_points, length(cc_indices)) +
                    length(cc_indices)
            report!(progress, n_now; extra="$ci/$n_components")
        end
    end

    empty!(cc_indices_by_id)   # release after per-component loop
    nbs_offset = Int32(nbs_counter[])

    # ── Step 4: merge skeletons across components (only when requested) ──
    # Kept serial: skeletons are tiny (≈ one vertex per NBS node ≪ N) so the
    # merge is cheap, and the merged buffers cannot be presized before the loop.
    if want_skeleton
        all_skel_coords    = Matrix{eltype(coords_filtered)}[]
        all_skel_attrs     = Dict{Symbol,Vector}[]
        all_skel_edges     = Tuple{Int,Int}[]
        skel_vertex_offset = 0
        for ci in 1:n_components
            skel_cloud = skel_clouds[ci]
            npoints(skel_cloud) > 0 || continue
            push!(all_skel_coords, coordinates(skel_cloud))
            push!(all_skel_attrs, _all_attributes(skel_cloud))
            for e in Graphs.edges(skel_graphs[ci])
                push!(all_skel_edges, (src(e) + skel_vertex_offset,
                                       dst(e) + skel_vertex_offset))
            end
            skel_vertex_offset += npoints(skel_cloud)
        end
        if isempty(all_skel_coords)
            merged_skel       = PointCloud(zeros(Float64, 0, 3), Dict{Symbol,Vector}())
            merged_skel_graph = SimpleGraph{Int}(0)
        else
            merged_skel       = merge_pointclouds(all_skel_coords, all_skel_attrs;
                                                  verbose=cfg.pipeline.enable_debug_info)
            merged_skel_graph = SimpleGraph{Int}(skel_vertex_offset)
            for (u, v) in all_skel_edges
                add_edge!(merged_skel_graph, u, v)
            end
        end
    else
        merged_skel       = PointCloud(zeros(Float64, 0, 3), Dict{Symbol,Vector}())
        merged_skel_graph = SimpleGraph{Int}(0)
    end

    # ── Step 5: rescue orphan branches into neighboring grounded trees across
    # occlusion gaps (orphan ⟺ tree_id==0 && tree_nbs_id>0) ──
    assemble_occluded_segments(coords_filtered, global_tree_id, global_tree_nbs_id; cfg=cfg)

    # ── Step 6: reorder tree_id by descending point count, then make tree_nbs_id
    # contiguous within each tree (globally-unique sequential blocks) ──
    global_tree_id = relabel_by_occurrence(global_tree_id; positive_only=true, T_out=Int32)
    tree_offset    = Int32(maximum(global_tree_id; init=0))
    _relabel_tree_nbs_within_trees!(global_tree_id, global_tree_nbs_id)

    # ── Step 7: attach attributes + write skeleton OBJ ────────────
    setattribute!(pc_filtered, :nbs_id,      global_nbs_id)
    setattribute!(pc_filtered, :node_id,     global_node_id)
    setattribute!(pc_filtered, :tree_id,     global_tree_id)
    setattribute!(pc_filtered, :tree_nbs_id, global_tree_nbs_id)

    if want_skeleton && !isempty(cfg.pipeline.output_dir)
        obj_path = joinpath(expanduser(cfg.pipeline.output_dir),
                            "$(cfg.pipeline.output_prefix)skeleton_graph.obj")
        _write_polyline_obj(obj_path, coordinates(merged_skel), merged_skel_graph)
        @info "$_LOG_PREFIX   wrote: $obj_path"
    end

    @info "$_LOG_PREFIX   n_components=$n_components, n_trees=$tree_offset, n_nbs=$nbs_offset"

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
A ground-disconnected component yields `tree_id == 0` everywhere (its branches keep
`tree_nbs_id`), so its points are orphans for the occlusion rescue.
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
neighborhood expansion. Connected components smaller than `cfg.tree_segmentation.min_nbs_size`
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

    min_segment_size       = cfg.tree_segmentation.min_nbs_size
    neighbor_distance      = cfg.tree_segmentation.nbs_neighbor_distance
    nearground_agh_ceiling = cfg.tree_segmentation.nearground_agh_threshold + 2.0 * cfg.pipeline.subsample_res

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
    # Debug-gated progress: only constructed (and only logged) when the flag is on.
    nbs_progress = cfg.pipeline.enable_debug_info ?
                   ProgressReporter("NBS labeling", N) : nothing

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
                linearity_angle_deg  = Float64(cfg.tree_segmentation.linearity_angle_deg),
                min_frontier_cc_size = Int(cfg.tree_segmentation.frontier_min_cc_size),
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

        nbs_progress !== nothing && report!(nbs_progress, n_labeled_total; extra="nbs=$(next_id-1)")
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
    nearground_ceiling = cfg.tree_segmentation.nearground_agh_threshold + 2.0 * cfg.pipeline.subsample_res
    merge_threshold    = cfg.tree_segmentation.assembly_merge_threshold

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

    cfg.pipeline.enable_debug_info && @info "Assembly: seeded $(next_tree_id - 1) trees from near-ground NBS"

    # A component is "grounded" iff it had at least one near-ground seed. An ungrounded
    # component is still grown (so its branches get `tree_nbs_id` labels) but its
    # `tree_id` is zeroed below, so its points become orphans for the occlusion rescue.
    had_nearground = next_tree_id > Int32(1)

    # ── Step 4.1b: fallback seed for ground-disconnected CCs ────
    # If this component has no near-ground NBS, iterative growth would leave every
    # NBS unassigned. We always seed the largest NBS as a fresh tree so step 4.2 can
    # grow through it and assign branch labels; the temporary `tree_id` is zeroed below.
    if next_tree_id == Int32(1)
        assigned_tid = _seed_largest_nbs!(tree_id, nbs_tree, assigned_nbs,
                                          nbs_points, next_tree_id; cfg=cfg)
        if assigned_tid > 0
            next_tree_id += Int32(1)
        end
    end

    # ── Step 4.2: iterative growth via skeleton graph ────────────
    iteration = _iterative_tree_growth!(tree_id, tree_nbs_id, nbs_tree, assigned_nbs,
                                        K_nbs, nbs_points, nbs_skel_nodes, skel_to_nbs,
                                        nbs_adj, graph_skeleton, Float64(merge_threshold);
                                        cfg=cfg)

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

    # Ground-disconnected component: keep the branch labels (`tree_nbs_id`) but drop the
    # temporary `tree_id` so every point becomes an orphan (`tree_id==0 && tree_nbs_id>0`)
    # for `assemble_occluded_segments` to rescue.
    had_nearground || fill!(tree_id, Int32(0))

    n_trees = length(unique(tid for tid in tree_id if tid > 0))
    n_assigned_pts = count(>(Int32(0)), tree_id)
    cfg.pipeline.enable_debug_info && @info "Assembly complete" n_trees n_assigned_points=n_assigned_pts total_points=N iterations=iteration

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
    _seed_largest_nbs!(tree_id, nbs_tree, assigned_nbs, nbs_points,
                       next_tree_id; cfg=_CFG) -> Int32

Find the single largest NBS by `length(nbs_points[k])` and seed it with
`next_tree_id`. Mutates `tree_id`, `nbs_tree`, `assigned_nbs` in place.
Returns the assigned tree id, or `Int32(0)` if every NBS slot is empty.

Intended as a fallback for `assemble_segments` when a connected component
has no near-ground NBS — gives a ground-disconnected CC its own
deterministic seed so step 4.2 can grow a tree through it instead of
leaving everything for the cross-component orphan rescue to clean up.
The caller is responsible for the "no near-ground seeds exist" gate; this
helper does not consult `assigned_nbs` to skip already-seeded NBS because
in its only intended call site nothing is seeded yet.
"""
function _seed_largest_nbs!(tree_id::AbstractVector{Int32},
                            nbs_tree::Vector{Int32},
                            assigned_nbs::BitVector,
                            nbs_points::Vector{Vector{Int}},
                            next_tree_id::Int32;
                            cfg::FLiPConfig=_CFG)
    K_nbs  = length(nbs_points)
    best_k = 0
    best_n = 0
    @inbounds for k in 1:K_nbs
        n = length(nbs_points[k])
        if n > best_n
            best_n = n
            best_k = k
        end
    end
    best_k == 0 && return Int32(0)

    nbs_tree[best_k]     = next_tree_id
    assigned_nbs[best_k] = true
    @inbounds for i in nbs_points[best_k]
        tree_id[i] = next_tree_id
    end
    cfg.pipeline.enable_debug_info && @info "Assembly: seeded largest NBS $best_k ($best_n points) as ungrounded tree $next_tree_id (no near-ground seed in this CC)"
    return next_tree_id
end

"""
    _check_merge_and_update_nbs!(point_idxs, tree_id, tree_nbs_id,
                                 frac, merge_threshold,
                                 tid_if_branch, tnid_if_branch,
                                 tid_if_merge,  tnid_if_merge) -> Symbol

Apply the Rule A / Rule B merge decision to every index in `point_idxs`.

- **Rule B** fires when `frac > merge_threshold` AND `tnid_if_merge > 0`:
  `tree_id[i] = tid_if_merge`, `tree_nbs_id[i] = tnid_if_merge` for each point.
- **Rule A** otherwise (frac below threshold OR no valid merge target):
  `tree_id[i] = tid_if_branch`, `tree_nbs_id[i] = tnid_if_branch` — the NBS is
  preserved as its own branch.

Returns `:rule_a` or `:rule_b`. Used by `_iterative_tree_growth!` so the merge rule
has a single implementation.
"""
@inline function _check_merge_and_update_nbs!(
    point_idxs::AbstractVector{Int},
    tree_id::AbstractVector{Int32},
    tree_nbs_id::AbstractVector{Int32},
    frac::Float64,
    merge_threshold::Float64,
    tid_if_branch::Int32, tnid_if_branch::Int32,
    tid_if_merge::Int32,  tnid_if_merge::Int32,
)
    is_merge = (frac > merge_threshold) && (tnid_if_merge > 0)
    out_tid  = is_merge ? tid_if_merge  : tid_if_branch
    out_tnid = is_merge ? tnid_if_merge : tnid_if_branch
    @inbounds for i in point_idxs
        tree_id[i]     = out_tid
        tree_nbs_id[i] = out_tnid
    end
    return is_merge ? :rule_b : :rule_a
end

"""
    _iterative_tree_growth!(tree_id, tree_nbs_id, nbs_tree, assigned_nbs,
                            K_nbs, nbs_points, nbs_skel_nodes, skel_to_nbs,
                            nbs_adj, graph_skeleton, merge_threshold; cfg=_CFG) -> Int

Iteratively grow trees from the currently-seeded NBS outward through the
skeleton graph. `assigned_nbs[k] == true` marks the seed set on entry; the
function consumes it and mutates `assigned_nbs`, `nbs_tree`, per-point
`tree_id`, and per-point `tree_nbs_id` in place. Returns the number of
growth iterations performed.

`frontier_info` (built locally) maps an unassigned NBS to the set of
candidate `tree_id → connection_count` votes from its already-assigned
skeleton neighbors. Within each iteration the frontier is processed largest
NBS first; each NBS picks the highest-voted tree, then dispatches to
[`_check_merge_and_update_nbs!`](@ref) which encodes the shared Rule A /
Rule B decision:

- **Rule A** (`frac_connected ≤ merge_threshold`) — most of the NBS's
  skeleton nodes are internal/unassigned, so it gets attached as a new
  branch of the winning tree; the NBS keeps its own `tree_nbs_id`.
- **Rule B** (`frac_connected > merge_threshold`) — the NBS straddles an
  already-assigned NBS, so it merges into that target's `tree_nbs_id` and
  inherits its `tree_id`.

Frontier updates from within an iteration are deferred and applied
afterwards, so within one round the decisions all see the same snapshot.
"""
function _iterative_tree_growth!(tree_id::AbstractVector{Int32},
                                 tree_nbs_id::AbstractVector{Int32},
                                 nbs_tree::Vector{Int32},
                                 assigned_nbs::BitVector,
                                 K_nbs::Int,
                                 nbs_points::Vector{Vector{Int}},
                                 nbs_skel_nodes::Vector{Vector{Int}},
                                 skel_to_nbs::Vector{Int},
                                 nbs_adj::SparseMatrixCSC{Int,Int},
                                 graph_skeleton::SimpleGraph{Int},
                                 merge_threshold::Float64;
                                 cfg::FLiPConfig=_CFG)
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

            # Rule B target: NBS with the highest skeleton-edge count (reuses the
            # `skel_neighbor_counts` from the fused loop above). `target_tnid == 0`
            # means no candidate exists, and the helper falls back to Rule A.
            target_nbs   = 0
            target_count = 0
            for (nbr_nbs, cnt) in skel_neighbor_counts
                if cnt > target_count
                    target_count = cnt
                    target_nbs   = nbr_nbs
                end
            end
            target_tid_raw = target_nbs > 0 ? nbs_tree[target_nbs] : Int32(0)
            # Split-NBS edge case: target marked -1 → treat as unassigned tree
            target_tid     = target_tid_raw == Int32(-1) ? Int32(0) : target_tid_raw
            target_tnid    = Int32(target_nbs)

            rule = _check_merge_and_update_nbs!(
                nbs_points[k], tree_id, tree_nbs_id,
                frac_connected, merge_threshold,
                best_tree,  Int32(k),       # Rule A: new branch under best_tree, keep own NBS id
                target_tid, target_tnid,    # Rule B: merge into target NBS within its tree
            )

            if rule === :rule_a
                nbs_tree[k]     = best_tree
                assigned_nbs[k] = true
                push!(pending_assignments, (k, best_tree))
                n_assigned_this_round += 1
            elseif target_tnid > 0
                nbs_tree[k]     = target_tid
                assigned_nbs[k] = true
                push!(pending_assignments, (k, target_tid))
                n_assigned_this_round += 1
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

        cfg.pipeline.enable_debug_info && @info "Assembly iteration $iteration: assigned $n_assigned_this_round NBS" total_assigned=count(assigned_nbs)
    end

    return iteration
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

# ── Step 5: occlusion-gap rescue of orphan branches ──────────────

"""
    _occluded_round_votes(frontier, coords, tree_id, tree_nbs_id, orphan_voxel_index,
                          assigned, inv_vs, link_tol2, n_thread)
        -> Dict{Tuple{Int32,Int32}, Int}

Count, for one wavefront round, the point-level links between frontier points and
still-unassigned orphan points: `(orphan tree_nbs_id, frontier tree_id) → link count`.
Each frontier point is matched against orphan points in its own voxel ±1 (the voxel size
guarantees a `≤ √link_tol2` link spans at most one voxel) under an exact squared-distance
test. Frontier points are scanned in contiguous chunks; per-chunk dicts are summed
(order-independent), so the result is deterministic.
"""
function _occluded_round_votes(frontier::Vector{Int}, coords::AbstractMatrix{<:Real},
                               tree_id::AbstractVector{Int32}, tree_nbs_id::AbstractVector{Int32},
                               orphan_voxel_index::Dict{NTuple{3,Int}, Vector{Int}},
                               assigned::Set{Int32}, inv_vs::Float64, link_tol2::Float64,
                               n_thread::Integer)
    M = length(frontier)
    M == 0 && return Dict{Tuple{Int32,Int32}, Int}()

    function count_chunk(lo, hi)
        d = Dict{Tuple{Int32,Int32}, Int}()
        @inbounds for fi in lo:hi
            f    = frontier[fi]
            ftid = tree_id[f]
            fx = float(coords[f, 1]); fy = float(coords[f, 2]); fz = float(coords[f, 3])
            vx = floor(Int, fx * inv_vs); vy = floor(Int, fy * inv_vs); vz = floor(Int, fz * inv_vs)
            for dz in -1:1, dy in -1:1, dx in -1:1
                bucket = get(orphan_voxel_index, (vx + dx, vy + dy, vz + dz), nothing)
                bucket === nothing && continue
                for c in bucket
                    branch = tree_nbs_id[c]
                    branch in assigned && continue
                    ddx = float(coords[c, 1]) - fx
                    ddy = float(coords[c, 2]) - fy
                    ddz = float(coords[c, 3]) - fz
                    (ddx * ddx + ddy * ddy + ddz * ddz) <= link_tol2 || continue
                    key = (branch, ftid)
                    d[key] = get(d, key, 0) + 1
                end
            end
        end
        return d
    end

    nt = min(Int(n_thread), M)
    (nt <= 1 || Threads.nthreads() == 1) && return count_chunk(1, M)

    chunk   = cld(M, nt)
    nchunks = cld(M, chunk)
    parts   = Vector{Dict{Tuple{Int32,Int32}, Int}}(undef, nchunks)
    _parallel_for(nchunks, nt) do c
        lo = (c - 1) * chunk + 1
        hi = min(c * chunk, M)
        parts[c] = count_chunk(lo, hi)   # distinct slot per chunk → race-free
    end
    votes = Dict{Tuple{Int32,Int32}, Int}()
    for d in parts, (k, v) in d
        votes[k] = get(votes, k, 0) + v
    end
    return votes
end

"""
    assemble_occluded_segments(coords, tree_id, tree_nbs_id; cfg) -> nothing

Rescue *orphan branches* (points with `tree_id == 0 && tree_nbs_id > 0`) into neighboring
grounded trees across occlusion gaps, via a multi-source wavefront that grows outward from
the grounded points (`tree_id > 0`).

A compact voxel index is built over the orphan points only and reused across rounds. Each
round, every current frontier point votes for the orphan branches it links to (exact point
distance `≤ occlusion_tol + sub_res`); each still-unassigned branch is then assigned, as a
whole, to the grounded `tree_id` with the most point-level links (ties broken by smaller
id), keeping its already globally-unique `tree_nbs_id`. The newly assigned points form the
next frontier, so chains of occluded segments are bridged progressively. Branches never
reached stay orphans (`tree_id == 0`). No-op when `occlusion_tol ≤ 0`.

Mutates `tree_id` in place.
"""
function assemble_occluded_segments(
    coords::AbstractMatrix{<:Real},
    tree_id::AbstractVector{Int32},
    tree_nbs_id::AbstractVector{Int32};
    cfg::FLiPConfig = _CFG,
)
    occlusion_tol = cfg.tree_segmentation.assembly_occlusion_tolerance
    occlusion_tol > 0 || return nothing
    N         = size(coords, 1)
    sub_res   = cfg.pipeline.subsample_res
    link_tol2 = (occlusion_tol + sub_res)^2
    inv_vs    = 1.0 / (occlusion_tol + 2.0 * sub_res)   # voxel ≥ link distance ⇒ ±1 covers a link
    nt        = effective_nthreads(cfg)
    VK        = NTuple{3, Int}

    voxel_of(i) = (floor(Int, float(coords[i, 1]) * inv_vs),
                   floor(Int, float(coords[i, 2]) * inv_vs),
                   floor(Int, float(coords[i, 3]) * inv_vs))

    # ── orphan points (parallel scan) → voxel index + branch→points map ──
    orphan_idx = _parallel_findall(N, nt) do i
        @inbounds tree_id[i] == 0 && tree_nbs_id[i] > 0
    end
    isempty(orphan_idx) && return nothing

    orphan_voxel_index = Dict{VK, Vector{Int}}()
    orphan_pts         = Dict{Int32, Vector{Int}}()   # tree_nbs_id → point idxs (atomic unit)
    @inbounds for i in orphan_idx
        push!(get!(orphan_voxel_index, voxel_of(i), Int[]), i)
        push!(get!(orphan_pts, tree_nbs_id[i], Int[]), i)
    end
    @info "$_LOG_PREFIX   assemble_occluded_segments: $(length(orphan_pts)) orphan branches (occlusion_tol=$occlusion_tol)"

    # ── contact voxels: orphan voxels expanded by ±1 (where a grounded point may link) ──
    contact_voxels = Set{VK}()
    for vk in keys(orphan_voxel_index)
        for dz in -1:1, dy in -1:1, dx in -1:1
            push!(contact_voxels, (vk[1] + dx, vk[2] + dy, vk[3] + dz))
        end
    end

    # ── seed frontier: grounded points sitting in a contact voxel (parallel scan) ──
    frontier = _parallel_findall(N, nt) do i
        @inbounds tree_id[i] > 0 && (voxel_of(i) in contact_voxels)
    end

    assigned = Set{Int32}()   # orphan branch ids already rescued
    while !isempty(frontier)
        votes = _occluded_round_votes(frontier, coords, tree_id, tree_nbs_id,
                                      orphan_voxel_index, assigned, inv_vs, link_tol2, nt)
        isempty(votes) && break

        # per-branch argmax frontier tree_id (tie → smaller id)
        best  = Dict{Int32, Int32}()
        bestc = Dict{Int32, Int}()
        for ((branch, tid), c) in votes
            cur    = get(bestc, branch, 0)
            curtid = get(best,  branch, typemax(Int32))
            if c > cur || (c == cur && tid < curtid)
                best[branch]  = tid
                bestc[branch] = c
            end
        end
        isempty(best) && break

        newly = Int[]
        for branch in sort!(collect(keys(best)))
            tid = best[branch]
            pts = orphan_pts[branch]
            @inbounds for i in pts
                tree_id[i] = tid          # keep tree_nbs_id (already globally unique)
            end
            push!(assigned, branch)
            append!(newly, pts)
        end
        frontier = newly
    end
    return nothing
end

"""
    _relabel_tree_nbs_within_trees!(tree_id, tree_nbs_id) -> nothing

Relabel `tree_nbs_id` so that within each `tree_id` group the non-zero values form a
contiguous block ranked by descending occurrence, with the blocks laid out sequentially by
ascending `tree_id`. The `tree_id == 0` (orphan) group sorts **last**, so valid-tree
`tree_nbs_id` occupy the low labels and orphan labels are the largest. The result is
per-tree contiguous AND globally unique (each new `tree_nbs_id` maps to exactly one
`tree_id`), keeping QSM's group-by-`tree_nbs_id` correct. `tree_nbs_id == 0` stays 0.
"""
function _relabel_tree_nbs_within_trees!(tree_id::AbstractVector{Int32},
                                         tree_nbs_id::AbstractVector{Int32})
    N = length(tree_id)
    pair_count = Dict{Tuple{Int32, Int32}, Int}()
    @inbounds for i in 1:N
        tn = tree_nbs_id[i]
        tn > 0 || continue
        key = (tree_id[i], tn)
        pair_count[key] = get(pair_count, key, 0) + 1
    end
    isempty(pair_count) && return nothing

    pairs = collect(keys(pair_count))
    # tree_id asc but the orphan group (tree_id==0) sorts LAST, then count desc, tnbs asc.
    sort!(pairs; by = p -> (p[1] == 0 ? typemax(Int32) : p[1], -pair_count[p], p[2]))

    new_label = Dict{Tuple{Int32, Int32}, Int32}()
    sizehint!(new_label, length(pairs))
    next = Int32(1)
    for p in pairs
        new_label[p] = next
        next += Int32(1)
    end
    @inbounds for i in 1:N
        tn = tree_nbs_id[i]
        tn > 0 || continue
        tree_nbs_id[i] = new_label[(tree_id[i], tn)]
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
