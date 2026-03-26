"""
    _labels_to_sorted_clusters(subset, labels) -> Vector{Vector{Int}}

Convert per-element CC labels (from `connected_component_subset!`) into a list of
vertex-index vectors sorted by descending component size. Labels are already ranked
(1 = largest) by `connected_component_subset!`, so `clusters[1]` is the largest.
Vertices with label 0 (below `min_cc_size`) are dropped.
"""
function _labels_to_sorted_clusters(subset::AbstractVector{Int}, labels::Vector{Int})
    max_label = maximum(labels; init=0)
    max_label == 0 && return Vector{Vector{Int}}()
    clusters = [Int[] for _ in 1:max_label]
    @inbounds for (i, v) in enumerate(subset)
        lab = labels[i]
        lab > 0 && push!(clusters[lab], v)
    end
    return filter!(!isempty, clusters)
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
accumulated from prior `greedy_connected_neighborhood_search` calls. Filters out
already-labeled vertices, sorts the remainder by ascending z, and returns each as a
single-point seed cluster. When `frontier_buf` is exhausted, falls back to the next
lowest-z unlabeled point via `z_cursor` (O(1) amortised). Returns an empty vector when
no unlabeled points remain (signals termination).
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
            return _labels_to_sorted_clusters(candidate_buf, labels)
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

"""
    connected_component_labels(graph::SimpleGraph{Int}, min_cc_size::Integer=1) -> Vector{Int}

Label graph connected components by descending component size. Components smaller
than `min_cc_size` receive label 0.
"""
function connected_component_labels(graph::SimpleGraph{Int}, min_cc_size::Integer=1)
    min_cc_size >= 1 || throw(ArgumentError("min_cc_size must be >= 1"))

    n = nv(graph)
    n == 0 && return Int[]

    labels = zeros(Int, n)
    components = connected_components(graph)
    order = sortperm(length.(components); rev=true)

    next_label = 1
    for idx in order
        comp = components[idx]
        length(comp) >= min_cc_size || continue
        @inbounds for v in comp
            labels[v] = next_label
        end
        next_label += 1
    end

    return labels
end

"""
    generate_proto_nodes_from_slice_label(points, graph, slice_labels; min_cc_size=1)

Generate proto-node labels from shortest-path slice labels using connected components
from an already-built graph (no graph rebuild inside this function).
"""
function generate_proto_nodes_from_slice_label(points::AbstractMatrix{<:Real},
                                               graph::SimpleGraph{Int},
                                               slice_labels::AbstractVector{<:Integer};
                                               min_cc_size::Integer=1,
                                               cc_workspace::Union{Nothing, ConnectedComponentSubsetWorkspace}=nothing)
    size(points, 2) == 3 || throw(ArgumentError("points must be N×3 matrix"))
    n = size(points, 1)
    nv(graph) == n || throw(ArgumentError("graph vertex count must match number of points"))
    length(slice_labels) == n || throw(ArgumentError("slice_labels length must match number of points"))
    min_cc_size >= 1 || throw(ArgumentError("min_cc_size must be >= 1"))

    n == 0 && return Int[]

    temp_labels = zeros(Int, n)
    proto_nodes = zeros(Int, n)
    odd_indices = Int[]
    even_indices = Int[]

    @inbounds for i in 1:n
        label_i = Int(slice_labels[i])
        if label_i > 0
            if isodd(label_i)
                push!(odd_indices, i)
            else
                push!(even_indices, i)
            end
        end
    end

    local_cc_workspace = isnothing(cc_workspace) ? ConnectedComponentSubsetWorkspace(nv(graph)) : cc_workspace

    next_temp_label = 1
    if !isempty(odd_indices)
        odd_components = connected_component_subset!(local_cc_workspace, graph, odd_indices, min_cc_size)
        @inbounds for (local_idx, point_idx) in enumerate(odd_indices)
            temp_labels[point_idx] = odd_components[local_idx]
        end
        max_odd_label = isempty(odd_components) ? 0 : maximum(odd_components)
        next_temp_label = max_odd_label + 1
    end

    if !isempty(even_indices)
        even_components = connected_component_subset!(local_cc_workspace, graph, even_indices, min_cc_size)
        @inbounds for (local_idx, point_idx) in enumerate(even_indices)
            component_label = even_components[local_idx]
            component_label > 0 || continue
            temp_labels[point_idx] = next_temp_label + component_label - 1
        end
    end

    component_to_proto = Dict{Int, Int}()
    next_proto_label = 1
    ordered_slices = sort!(collect(unique(Int(label) for label in slice_labels if label > 0)))

    for slice_label in ordered_slices
        seen_in_slice = Set{Int}()
        @inbounds for i in 1:n
            Int(slice_labels[i]) == slice_label || continue
            temp_label = temp_labels[i]
            temp_label == 0 && continue
            temp_label in seen_in_slice && continue
            push!(seen_in_slice, temp_label)
            if !haskey(component_to_proto, temp_label)
                component_to_proto[temp_label] = next_proto_label
                next_proto_label += 1
            end
        end
    end

    @inbounds for i in 1:n
        temp_label = temp_labels[i]
        temp_label == 0 && continue
        proto_nodes[i] = component_to_proto[temp_label]
    end

    return proto_nodes
end

"""
    label_non_branching_segments(graph, points; cfg) -> NamedTuple

Segment every vertex of `graph` into non-branching segments (NBS) using
`greedy_connected_neighborhood_search`. Connected components smaller than
`cfg.tree_nbs_min_segment_size` are discarded upfront (label 0). Valid
segments are relabeled by descending size (largest segment → label 1).

Returns `(nbs_id::Vector{Int32}, node_id::Vector{Int32})`.
"""
function label_non_branching_segments(
    graph::SimpleGraph{Int},
    points::AbstractMatrix{<:Real},
    agh_values::AbstractVector{<:Real};
    cfg::FLiPConfig = _CFG,
)
    N = nv(graph)
    size(points, 1) == N || throw(ArgumentError("graph vertex count must match number of points"))
    length(agh_values) == N || throw(ArgumentError("agh_values length must match graph vertex count"))

    min_segment_size      = cfg.tree_nbs_min_segment_size
    neighbor_distance     = cfg.tree_nbs_neighbor_distance
    max_iter              = cfg.tree_nbs_max_iterations
    nearground_agh_ceiling = cfg.tree_nearground_agh_threshold + 2.0 * cfg.pipeline_subsample_res

    global_nbs_id  = zeros(Int, N)
    global_node_id = zeros(Int, N)
    gsws           = GreedySearchWorkspace(N)
    unlabeled_mask = trues(N)

    # Pre-pass: discard isolated clusters smaller than min_segment_size in O(V+E).
    for comp in connected_components(graph)
        length(comp) >= min_segment_size && continue
        @inbounds for v in comp
            global_nbs_id[v]  = -1
            unlabeled_mask[v] = false
        end
    end

    # Pre-allocated buffers for _find_seed_clusters (reused across iterations)
    seed_candidate_buf = sizehint!(Int[], 1024)
    frontier_buf       = sizehint!(Int[], 1024)  # rejected frontier CCs from greedy searches

    # Pre-sort by ascending z for fallback seed selection.
    z_sorted = sortperm(view(points, :, 3); rev=false)
    z_cursor = Ref(1)

    next_id          = 1
    next_global_node = 1
    n_labeled_total  = count(!, unlabeled_mask)  # already labeled (discarded small CCs)
    last_pct_report  = 0
    t_nbs_start      = time()

    while next_id - 1 < max_iter
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
        isempty(seed_clusters) && break  # no unlabeled points remain

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

            result = greedy_connected_neighborhood_search(
                graph, start_vertices, neighbor_distance;
                vertex_mask          = unlabeled_mask,
                workspace            = gsws,
                points               = points,
                linearity_angle_deg  = Float64(cfg.tree_linearity_angle_deg),
                min_frontier_cc_size = Int(cfg.tree_min_cc_size),
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
            next_id - 1 >= max_iter && break
        end

        pct = round(Int, 100.0 * n_labeled_total / N)
        if pct >= last_pct_report + 5
            last_pct_report = pct - (pct % 5)
            @info "NBS labeling progress" pct=last_pct_report nbs_count=next_id-1 elapsed_s=round(time()-t_nbs_start, digits=1)
        end
    end

    # Relabel by descending size: largest → 1, discarded → 0.
    seg_sizes = Dict{Int, Int}()
    for id in global_nbs_id
        id > 0 || continue
        seg_sizes[id] = get(seg_sizes, id, 0) + 1
    end
    sorted_segs = sort!(collect(keys(seg_sizes)); by = id -> -seg_sizes[id])
    old_to_new  = Dict{Int, Int}(old => new for (new, old) in enumerate(sorted_segs))
    @inbounds for i in eachindex(global_nbs_id)
        id = global_nbs_id[i]
        if id > 0
            global_nbs_id[i] = old_to_new[id]
        elseif id == -1
            global_nbs_id[i] = 0
        end
    end

    return (
        nbs_id  = Int32.(global_nbs_id),
        node_id = Int32.(global_node_id),
    )
end

"""
    _write_polyline_obj(path, coords, graph)

Write a skeleton graph as an OBJ polyline file readable by CloudCompare.
Every edge is written as a separate `l u v` statement (1-indexed vertices).
"""
function _write_polyline_obj(path::AbstractString, coords::AbstractMatrix{<:Real}, graph::SimpleGraph{Int})
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

"""
    create_skeleton_cloud(graph, coords_filtered, node_id; template_pc=nothing)
        -> NamedTuple{(:skeleton_cloud, :graph_skeleton)}

Build a skeleton point cloud and an MST-pruned skeleton graph from NBS node
assignments produced by `label_non_branching_segments`.

Each non-zero node label `n` in `node_id` is represented in the skeleton by the
centroid of all points assigned to that node. The returned `skeleton_cloud` carries
extra per-point attributes `:node_id` (original node label) and `:n_points` (number
of raw points contributing to that node centroid).

The skeleton graph has one vertex per node. An edge between nodes A and B is
inserted for every point-level edge in `graph` that crosses the A/B node boundary.
Edge weight is `1 / count` where `count` is the total number of such crossing
point-pair connections (more connections -> lower weight -> preferred in MST).
Kruskal MST is applied to remove cycles while keeping the strongest connections.
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

    # Find max node id
    max_node = 0
    @inbounds for i in 1:N
        nid = Int(node_id[i])
        nid > max_node && (max_node = nid)
    end

    if max_node == 0
        empty_pc = if isnothing(template_pc)
            PointClouds.LAS((x=Float64[], y=Float64[], z=Float64[]))
        else
            PointClouds.LAS((x=Float64[], y=Float64[], z=Float64[]);
                            coord_scale=template_pc.coord_scale, coord_offset=template_pc.coord_offset)
        end
        return (skeleton_cloud=empty_pc, graph_skeleton=SimpleGraph{Int}(0))
    end

    # Accumulate per-node coordinate sums and counts
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

    # Keep only nodes with at least one point
    valid_nodes  = [n for n in 1:max_node if node_cnt[n] > 0]
    n_nodes      = length(valid_nodes)
    node_to_skel = zeros(Int, max_node)   # original node label -> skeleton vertex index
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

    # Build skeleton PointCloud
    pts_nt   = (x=skel_coords[:, 1], y=skel_coords[:, 2], z=skel_coords[:, 3])
    skel_las = if isnothing(template_pc)
        PointClouds.LAS(pts_nt)
    else
        PointClouds.LAS(pts_nt; coord_scale=template_pc.coord_scale, coord_offset=template_pc.coord_offset)
    end
    skel_pc = setattribute!(skel_las, :node_id,  skel_nids)
    skel_pc = setattribute!(skel_pc,  :n_points, skel_npts)

    # Count point-wise cross-node edge connections
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

    # Build skeleton graph — keep all cross-node edges (no MST pruning)
    n_pairs = length(edge_counts)
    if n_pairs == 0
        return (skeleton_cloud=skel_pc, graph_skeleton=SimpleGraph{Int}(n_nodes))
    end

    graph_skeleton = SimpleGraph{Int}(n_nodes)
    for ((u, v), _) in edge_counts
        add_edge!(graph_skeleton, u, v)
    end

    return (skeleton_cloud=skel_pc, graph_skeleton=graph_skeleton)
end

"""
    assemble_segments(graph, coords, nbs_id, node_id, agh_values,
                      graph_skeleton, skeleton_cloud; cfg) -> NamedTuple

Assemble non-branching segments (NBS) into individual trees by iteratively
growing from near-ground seed NBS outward through the skeleton graph.

# Arguments
- `graph::SimpleGraph{Int}`: point-level radius graph
- `coords::AbstractMatrix{<:Real}`: N×3 point coordinates
- `nbs_id::AbstractVector{<:Integer}`: per-point NBS label (0=discarded, 1..k)
- `node_id::AbstractVector{<:Integer}`: per-point node label (globally unique)
- `agh_values::AbstractVector{<:Real}`: per-point above-ground height
- `graph_skeleton::SimpleGraph{Int}`: MST skeleton graph (one vertex per node)
- `skeleton_cloud::PointCloud`: skeleton point cloud with `:node_id` attribute
- `cfg::FLiPConfig`: configuration (uses `tree_nearground_agh_threshold`)

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
    merge_threshold = cfg.tree_assembly_merge_threshold

    # ------------------------------------------------------------------
    # Step 4.0: Initialise output arrays
    # ------------------------------------------------------------------
    tree_id     = zeros(Int32, N)
    tree_nbs_id = Int32.(copy(nbs_id))

    # ------------------------------------------------------------------
    # Precomputation A: NBS → point indices
    # ------------------------------------------------------------------
    nbs_points = Dict{Int, Vector{Int}}()
    @inbounds for i in 1:N
        nid = Int(nbs_id[i])
        nid > 0 || continue
        pts = get!(nbs_points, nid, Int[])
        push!(pts, i)
    end

    isempty(nbs_points) && return (tree_nbs_id = tree_nbs_id, tree_id = tree_id)

    # ------------------------------------------------------------------
    # Precomputation B: node_id → skeleton vertex index
    # ------------------------------------------------------------------
    skel_node_ids = getattribute(skeleton_cloud, :node_id)
    n_skel = nv(graph_skeleton)
    node_to_skel = Dict{Int, Int}()
    sizehint!(node_to_skel, n_skel)
    for si in 1:n_skel
        node_to_skel[Int(skel_node_ids[si])] = si
    end

    # ------------------------------------------------------------------
    # Precomputation C: skeleton vertex → NBS label
    # ------------------------------------------------------------------
    skel_to_nbs = zeros(Int, n_skel)
    @inbounds for i in 1:N
        nid = Int(node_id[i])
        nid > 0 || continue
        sid = Int(nbs_id[i])
        sid > 0 || continue
        sv = get(node_to_skel, nid, 0)
        sv > 0 && (skel_to_nbs[sv] = sid)
    end

    # ------------------------------------------------------------------
    # Precomputation D: NBS → skeleton vertex set
    # ------------------------------------------------------------------
    nbs_skel_nodes = Dict{Int, Vector{Int}}()
    for sv in 1:n_skel
        nlab = skel_to_nbs[sv]
        nlab > 0 || continue
        push!(get!(nbs_skel_nodes, nlab, Int[]), sv)
    end

    # ------------------------------------------------------------------
    # Precomputation E: NBS adjacency via point-level graph (connection counts)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 4.1: Seed trees from near-ground NBS
    # ------------------------------------------------------------------
    next_tree_id = Int32(1)
    nbs_tree     = Dict{Int, Int32}()   # NBS label → tree_id
    assigned_nbs = Set{Int}()

    for (k, pts) in nbs_points
        min_agh = Inf
        @inbounds for i in pts
            v = Float64(agh_values[i])
            v < min_agh && (min_agh = v)
        end
        if min_agh <= nearground_ceiling
            nbs_tree[k] = next_tree_id
            push!(assigned_nbs, k)
            @inbounds for i in pts
                tree_id[i] = next_tree_id
            end
            next_tree_id += Int32(1)
        end
    end

    @info "Assembly: seeded $(next_tree_id - 1) trees from near-ground NBS"

    # ------------------------------------------------------------------
    # Step 4.2: Iterative growth via skeleton graph
    # ------------------------------------------------------------------
    iteration = 0
    while true
        iteration += 1

        # 4.2a: Find frontier NBS via skeleton graph
        # For each assigned NBS, check skeleton neighbors for unassigned NBS
        frontier_info = Dict{Int, Dict{Int32, Int}}()  # frontier_nbs → Dict(tree_id → connection_count)
        for assigned_k in assigned_nbs
            skel_nodes_k = get(nbs_skel_nodes, assigned_k, Int[])
            for sv in skel_nodes_k
                for sn in Graphs.neighbors(graph_skeleton, sv)
                    neighbor_nbs = skel_to_nbs[sn]
                    (neighbor_nbs > 0 && !(neighbor_nbs in assigned_nbs)) || continue
                    # This is a frontier NBS — record connection to the assigned tree
                    tid = nbs_tree[assigned_k]
                    info = get!(frontier_info, neighbor_nbs, Dict{Int32, Int}())
                    # Use point-level connection count for tie-breaking
                    adj_inner = get(nbs_adj, neighbor_nbs, Dict{Int, Int}())
                    conn_count = get(adj_inner, assigned_k, 0)
                    info[tid] = get(info, tid, 0) + conn_count
                end
            end
        end

        isempty(frontier_info) && break  # Step 4.4: no frontier → terminate

        # Sort frontier NBS by number of points (large to small)
        frontier_sorted = sort!(collect(keys(frontier_info));
                                by = k -> -length(get(nbs_points, k, Int[])))

        n_assigned_this_round = 0

        for k in frontier_sorted
            k in assigned_nbs && continue  # may have been assigned earlier this round

            skel_nodes_k = get(nbs_skel_nodes, k, Int[])
            tree_connections = frontier_info[k]
            n_total_nodes = length(skel_nodes_k)

            # Count how many nodes in this NBS connect to an existing tree NBS
            n_nodes_with_tree_conn = 0
            for sv in skel_nodes_k
                for sn in Graphs.neighbors(graph_skeleton, sv)
                    nbr_nbs = skel_to_nbs[sn]
                    if nbr_nbs != k && nbr_nbs in assigned_nbs
                        n_nodes_with_tree_conn += 1
                        break  # only count each node once
                    end
                end
            end

            # Determine best tree (most total point-level connections)
            best_tree = Int32(0)
            best_count = 0
            for (tid, cnt) in tree_connections
                if cnt > best_count
                    best_count = cnt
                    best_tree = tid
                end
            end
            best_tree == 0 && continue  # shouldn't happen for valid frontier

            # Fraction of nodes connected to existing tree NBS
            frac_connected = n_total_nodes > 0 ? n_nodes_with_tree_conn / n_total_nodes : 0.0

            if frac_connected <= merge_threshold
                # Rule A: Majority of nodes are internal / not connected to tree NBS
                # → valid branch, assign entire NBS to best_tree
                nbs_tree[k] = best_tree
                push!(assigned_nbs, k)
                pts = get(nbs_points, k, Int[])
                @inbounds for i in pts
                    tree_id[i] = best_tree
                end
                n_assigned_this_round += 1
            else
                # Rule B: Over threshold fraction of nodes connect to tree NBS
                # → merge entire NBS into the closest connected NBS by skeleton
                #   edge count (most skeleton edges = closest connection)
                skel_neighbor_counts = Dict{Int, Int}()  # neighbor NBS → skeleton edge count
                for sv in skel_nodes_k
                    for sn in Graphs.neighbors(graph_skeleton, sv)
                        nbr_nbs = skel_to_nbs[sn]
                        (nbr_nbs > 0 && nbr_nbs != k) || continue
                        skel_neighbor_counts[nbr_nbs] = get(skel_neighbor_counts, nbr_nbs, 0) + 1
                    end
                end

                # Pick the connected NBS with the most skeleton edges
                target_nbs = 0
                target_count = 0
                for (nbr_nbs, cnt) in skel_neighbor_counts
                    if cnt > target_count
                        target_count = cnt
                        target_nbs = nbr_nbs
                    end
                end

                if target_nbs > 0
                    # Adopt the target NBS's tree_id and nbs_id
                    target_tid = get(nbs_tree, target_nbs, Int32(0))
                    if target_tid == Int32(-1)
                        target_tid = Int32(0)  # target was a split NBS, treat as unassigned
                    end
                    pts = get(nbs_points, k, Int[])
                    @inbounds for i in pts
                        tree_id[i] = target_tid
                        tree_nbs_id[i] = Int32(target_nbs)
                    end
                    nbs_tree[k] = target_tid
                    push!(assigned_nbs, k)
                    n_assigned_this_round += 1
                end
            end
        end

        # Step 4.4: Terminate if nothing new assigned
        n_assigned_this_round == 0 && break

        @info "Assembly iteration $iteration: assigned $n_assigned_this_round NBS" total_assigned=length(assigned_nbs)
    end

    # ------------------------------------------------------------------
    # Step 4.2b: Orphan NBS rescue via coarse NBS-level graph
    # ------------------------------------------------------------------
    # Build a coarse graph where each node is either an orphan NBS or an
    # assigned tree. Edges come from two sources:
    #   1. Radius graph among orphan points (occlusion_tolerance) → orphan↔orphan
    #   2. KDTree search from orphan points to assigned points    → orphan↔tree
    # Then iteratively propagate tree_id through the coarse graph.
    occlusion_tol = cfg.tree_assembly_occlusion_tolerance
    if occlusion_tol > 0
        # Identify orphan NBS (nbs_id > 0, all points have tree_id == 0)
        orphan_nbs_points = Dict{Int, Vector{Int}}()
        @inbounds for i in 1:N
            nid = Int(nbs_id[i])
            nid > 0 && tree_id[i] == 0 || continue
            push!(get!(orphan_nbs_points, nid, Int[]), i)
        end
        @inbounds for i in 1:N
            nid = Int(nbs_id[i])
            nid > 0 && tree_id[i] > 0 && delete!(orphan_nbs_points, nid)
        end

        if !isempty(orphan_nbs_points)
            orphan_nbs_list = collect(keys(orphan_nbs_points))
            n_orphan_nbs = length(orphan_nbs_list)

            # Collect all orphan point indices
            orphan_pt_idx = Int[]
            for pts in values(orphan_nbs_points)
                append!(orphan_pt_idx, pts)
            end
            n_orphan_pts = length(orphan_pt_idx)

            # --- Source 1: radius graph among orphan points ---
            @info "Orphan rescue: building radius graph for $n_orphan_pts orphan points (r=$occlusion_tol m)"
            orphan_coords = coords[orphan_pt_idx, :]
            orphan_graph = build_radius_graph(orphan_coords, occlusion_tol).graph

            # Build coarse NBS-level adjacency from orphan graph edges
            # Nodes: orphan NBS labels. Edges: weighted by point-level crossing count.
            orphan_nbs_of_pt = zeros(Int, n_orphan_pts)
            @inbounds for (j, i) in enumerate(orphan_pt_idx)
                orphan_nbs_of_pt[j] = Int(nbs_id[i])
            end

            coarse_adj = Dict{Int, Dict{Int, Int}}()  # nbs_a → nbs_b → count
            @inbounds for e in Graphs.edges(orphan_graph)
                a = orphan_nbs_of_pt[src(e)]
                b = orphan_nbs_of_pt[dst(e)]
                (a > 0 && b > 0 && a != b) || continue
                inner_a = get!(coarse_adj, a, Dict{Int, Int}())
                inner_a[b] = get(inner_a, b, 0) + 1
                inner_b = get!(coarse_adj, b, Dict{Int, Int}())
                inner_b[a] = get(inner_b, a, 0) + 1
            end

            # --- Source 2: KDTree search from orphan points to assigned points ---
            assigned_idx = Int[]
            @inbounds for i in 1:N
                tree_id[i] > 0 && push!(assigned_idx, i)
            end

            if !isempty(assigned_idx)
                assigned_3xM = Matrix{Float64}(undef, 3, length(assigned_idx))
                @inbounds for (j, i) in enumerate(assigned_idx)
                    assigned_3xM[1, j] = coords[i, 1]
                    assigned_3xM[2, j] = coords[i, 2]
                    assigned_3xM[3, j] = coords[i, 3]
                end
                kdtree = KDTree(assigned_3xM)

                # For each orphan NBS, find connections to assigned trees
                # Use a sentinel: tree labels are negative in the coarse graph
                # to distinguish from orphan NBS labels.
                # coarse_adj[orphan_nbs][-tree_id] = count
                @info "Orphan rescue: searching KDTree for orphan→tree connections"
                for (orph_nbs, orph_pts) in orphan_nbs_points
                    for i in orph_pts
                        query = SVector{3, Float64}(coords[i, 1], coords[i, 2], coords[i, 3])
                        hits = inrange(kdtree, query, occlusion_tol)
                        for j in hits
                            aid = assigned_idx[j]
                            tid = tree_id[aid]
                            tid > 0 || continue
                            # Use -tid as the coarse node for assigned trees
                            inner = get!(coarse_adj, orph_nbs, Dict{Int, Int}())
                            inner[-tid] = get(inner, -tid, 0) + 1
                        end
                    end
                end

                # Also record tree_nbs_id connections for later merge target
                # orphan_nbs → Dict(tree_nbs_id → count) for the best tree
                orphan_to_tree_nbs = Dict{Int, Dict{Int32, Int}}()
                for (orph_nbs, orph_pts) in orphan_nbs_points
                    tnbs_counts = Dict{Int32, Int}()
                    for i in orph_pts
                        query = SVector{3, Float64}(coords[i, 1], coords[i, 2], coords[i, 3])
                        hits = inrange(kdtree, query, occlusion_tol)
                        for j in hits
                            aid = assigned_idx[j]
                            tnid = tree_nbs_id[aid]
                            tnid > 0 || continue
                            tnbs_counts[tnid] = get(tnbs_counts, tnid, 0) + 1
                        end
                    end
                    isempty(tnbs_counts) || (orphan_to_tree_nbs[orph_nbs] = tnbs_counts)
                end
            end

            # --- Iterative propagation on coarse graph ---
            # coarse_adj keys: positive = orphan NBS, negative = assigned tree (fixed)
            orphan_tree_id = Dict{Int, Int32}()  # orphan NBS → assigned tree_id (0 = unassigned)
            orphan_assigned = Set{Int}()

            orphan_iteration = 0
            while true
                orphan_iteration += 1
                n_rescued = 0

                # Sort unassigned orphan NBS by size (large first)
                unassigned = filter(k -> !(k in orphan_assigned), orphan_nbs_list)
                sort!(unassigned; by = k -> -length(orphan_nbs_points[k]))

                for orph_nbs in unassigned
                    adj = get(coarse_adj, orph_nbs, Dict{Int, Int}())
                    isempty(adj) && continue

                    # Collect tree_id votes from coarse neighbors
                    votes = Dict{Int32, Int}()
                    for (nbr, cnt) in adj
                        if nbr < 0
                            # Direct connection to assigned tree
                            votes[Int32(-nbr)] = get(votes, Int32(-nbr), 0) + cnt
                        elseif nbr in orphan_assigned
                            # Orphan NBS already rescued — use its tree_id
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
                @info "Orphan rescue iteration $orphan_iteration: rescued $n_rescued NBS"
            end

            # Apply orphan assignments to point-level arrays
            for (orph_nbs, best_tid) in orphan_tree_id
                best_tid > 0 || continue
                # Find best tree_nbs_id target
                tnbs_counts = get(orphan_to_tree_nbs, orph_nbs, Dict{Int32, Int}())
                # Filter to only counts for the chosen tree
                best_tnbs = Int32(orph_nbs)
                if !isempty(tnbs_counts)
                    best_tnbs_cnt = 0
                    for (tnid, cnt) in tnbs_counts
                        if cnt > best_tnbs_cnt
                            best_tnbs_cnt = cnt
                            best_tnbs = tnid
                        end
                    end
                end

                for i in orphan_nbs_points[orph_nbs]
                    tree_id[i] = best_tid
                    tree_nbs_id[i] = best_tnbs
                end
            end
        end
    end

    # ------------------------------------------------------------------
    # Step 4.3: Re-order tree_nbs_id — continuous labels by descending size
    # ------------------------------------------------------------------
    # Group points by (tree_id, tree_nbs_id) — Rule B merges have already
    # updated tree_nbs_id to the target NBS label during growth
    group_points = Dict{Tuple{Int32, Int32}, Vector{Int}}()
    @inbounds for i in 1:N
        tid = tree_id[i]
        nid = tree_nbs_id[i]
        (tid > 0 && nid > 0) || continue
        key = (tid, nid)
        push!(get!(group_points, key, Int[]), i)
    end

    # Sort groups by descending size, assign continuous labels from 1
    sorted_groups = sort!(collect(group_points); by = kv -> -length(kv[2]))
    global_label = Int32(1)
    for (_, pts) in sorted_groups
        @inbounds for i in pts
            tree_nbs_id[i] = global_label
        end
        global_label += 1
    end

    # Zero out tree_nbs_id for unassigned points
    @inbounds for i in 1:N
        tree_id[i] == 0 && (tree_nbs_id[i] = Int32(0))
    end

    n_trees = length(unique(tid for tid in tree_id if tid > 0))
    n_assigned_pts = count(>(Int32(0)), tree_id)
    @info "Assembly complete" n_trees n_assigned_points=n_assigned_pts total_points=N iterations=iteration

    return (tree_nbs_id = tree_nbs_id, tree_id = tree_id)
end

"""
    tree_segmentation(pc::PointCloud; cfg::FLiPConfig=_CFG) -> NamedTuple

Run tree segmentation on points with AGH attribute.
"""
function tree_segmentation(pc::PointCloud; cfg::FLiPConfig=_CFG)
    hasattribute(pc, :AGH) || throw(ArgumentError("tree_segmentation requires AGH attribute on input point cloud"))

    coords = coordinates(pc)
    agh = getattribute(pc, :AGH)

    threshold = cfg.tree_nearground_agh_threshold
    nearground_idx = findall(i -> isfinite(float(agh[i])) && float(agh[i]) > threshold, eachindex(agh))
    isempty(nearground_idx) && return (
        filtered_cloud=pc[1:0],
        pc_output=pc[1:0],
        skeleton_cloud=pc[1:0],
        n_components=0,
        neighbor_radius=0.0,
    )

    pc_filtered     = pc[nearground_idx]
    coords_filtered = coordinates(pc_filtered)
    agh_filtered    = float.(getattribute(pc_filtered, :AGH))

    neighbor_radius = cfg.tree_neighbor_radius > 0 ? cfg.tree_neighbor_radius : 2.0 * cfg.pipeline_subsample_res
    neighbor_radius > 0 || throw(ArgumentError("tree neighbor radius must be > 0"))

    g_res = build_radius_graph(coords_filtered, neighbor_radius)
    graph = g_res.graph

    nbs_res  = label_non_branching_segments(graph, coords_filtered, agh_filtered; cfg=cfg)
    skel_res = create_skeleton_cloud(graph, coords_filtered, nbs_res.node_id; template_pc=pc_filtered)
    asm_res  = assemble_segments(
        graph, coords_filtered, nbs_res.nbs_id, nbs_res.node_id, agh_filtered,
        skel_res.graph_skeleton, skel_res.skeleton_cloud; cfg=cfg,
    )

    pc_output      = setattribute!(pc_filtered, :nbs_id, nbs_res.nbs_id)
    pc_output      = setattribute!(pc_output, :node_id, nbs_res.node_id)
    pc_output      = setattribute!(pc_output, :tree_id, asm_res.tree_id)
    pc_output      = setattribute!(pc_output, :tree_nbs_id, asm_res.tree_nbs_id)
    skeleton_cloud = skel_res.skeleton_cloud

    if !isempty(cfg.pipeline_output_dir)
        obj_path = joinpath(expanduser(cfg.pipeline_output_dir), "$(cfg.pipeline_output_prefix)skeleton_graph.obj")
        _write_polyline_obj(obj_path, coordinates(skeleton_cloud), skel_res.graph_skeleton)
        println("[tree_segmentation] wrote: $obj_path")
    end

    return (
        filtered_cloud  = pc_filtered,
        pc_output       = pc_output,
        skeleton_cloud  = skeleton_cloud,
        n_components    = 0,
        neighbor_radius = neighbor_radius,
    )
end
