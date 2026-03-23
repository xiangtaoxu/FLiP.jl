"""
    _expand_nearground_cluster(graph, seed, agh_values, agh_ceiling, mask;
                               visited=nothing) -> Vector{Int}

BFS from `seed` over vertices where `mask[v]` is true and
`float(agh_values[v]) <= agh_ceiling`. Returns all discovered vertices (including
`seed`) as the connected near-ground cluster.

Pass a pre-allocated `visited::BitVector` of length `nv(graph)` to avoid O(N)
allocation per call. The BitVector is reset to `false` for all touched vertices
before returning.
"""
function _expand_nearground_cluster(
    graph::SimpleGraph{Int},
    seed::Int,
    agh_values::AbstractVector{<:Real},
    agh_ceiling::Float64,
    mask::BitVector;
    visited::Union{Nothing, BitVector}=nothing,
)
    vis = visited !== nothing ? visited : falses(nv(graph))
    cluster = Int[]
    queue   = Int[]

    vis[seed] = true
    push!(cluster, seed)
    push!(queue, seed)

    qi = 1
    while qi <= length(queue)
        v = queue[qi]
        qi += 1
        @inbounds for nbr in Graphs.neighbors(graph, v)
            vis[nbr] && continue
            mask[nbr] || continue
            float(agh_values[nbr]) <= agh_ceiling || continue
            vis[nbr] = true
            push!(cluster, nbr)
            push!(queue, nbr)
        end
    end

    # Reset touched entries so the BitVector is clean for reuse
    if visited !== nothing
        @inbounds for v in cluster
            vis[v] = false
        end
    end

    return cluster
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

    # Reusable BitVector for _expand_nearground_cluster (avoids O(N) alloc per seed)
    expand_visited = falses(N)

    # Pre-sort by ascending z for O(1) amortised seed selection.
    z_sorted = sortperm(view(points, :, 3); rev=false)
    z_cursor         = 1
    next_id          = 1
    next_global_node = 1
    n_labeled_total  = count(!, unlabeled_mask)  # already labeled (discarded small CCs)
    last_pct_report  = 0
    t_nbs_start      = time()

    while next_id - 1 < max_iter
        while z_cursor <= N && !unlabeled_mask[z_sorted[z_cursor]]
            z_cursor += 1
        end
        z_cursor > N && break
        seed = z_sorted[z_cursor]

        # If the seed AGH is within the near-ground ceiling, expand it to the full
        # connected cluster of near-ground points before starting the greedy search.
        start_vertices = if float(agh_values[seed]) <= nearground_agh_ceiling
            _expand_nearground_cluster(graph, seed, agh_values, nearground_agh_ceiling, unlabeled_mask;
                                       visited=expand_visited)
        else
            Int[seed]
        end


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

    # Build initial skeleton graph and run Kruskal MST
    n_pairs = length(edge_counts)
    if n_pairs == 0
        return (skeleton_cloud=skel_pc, graph_skeleton=SimpleGraph{Int}(n_nodes))
    end

    row_ids    = Vector{Int}(undef, 2 * n_pairs)
    col_ids    = Vector{Int}(undef, 2 * n_pairs)
    wt_vals    = Vector{Float64}(undef, 2 * n_pairs)
    init_graph = SimpleGraph{Int}(n_nodes)
    for (k, ((u, v), cnt)) in enumerate(edge_counts)
        add_edge!(init_graph, u, v)
        w = 1.0 / Float64(cnt)
        row_ids[2k-1] = u;  col_ids[2k-1] = v;  wt_vals[2k-1] = w
        row_ids[2k]   = v;  col_ids[2k]   = u;  wt_vals[2k]   = w
    end
    wt_matrix = sparse(row_ids, col_ids, wt_vals, n_nodes, n_nodes)

    mst_edges      = kruskal_mst(init_graph, wt_matrix; minimize=true)
    graph_skeleton = SimpleGraph{Int}(n_nodes)
    for e in mst_edges
        add_edge!(graph_skeleton, src(e), dst(e))
    end

    return (skeleton_cloud=skel_pc, graph_skeleton=graph_skeleton)
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

    nbs_res = label_non_branching_segments(graph, coords_filtered, agh_filtered; cfg=cfg)
    skel_res       = create_skeleton_cloud(graph, coords_filtered, nbs_res.node_id; template_pc=pc_filtered)

    pc_output      = setattribute!(pc_filtered, :nbs_id, nbs_res.nbs_id)
    pc_output      = setattribute!(pc_output, :node_id, nbs_res.node_id)
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
