function _las_from_coords(template_pc::PointCloud, coords::AbstractMatrix{<:Real})
    size(coords, 2) == 3 || throw(ArgumentError("coords must be N×3"))
    points_nt = (
        x=Float64.(coords[:, 1]),
        y=Float64.(coords[:, 2]),
        z=Float64.(coords[:, 3]),
    )
    return PointClouds.LAS(points_nt; coord_scale=template_pc.coord_scale, coord_offset=template_pc.coord_offset)
end

@inline function _argmax_z_in_subset(points::AbstractMatrix{<:Real}, subset::AbstractVector{<:Integer})
    isempty(subset) && throw(ArgumentError("subset must be non-empty"))
    best_idx = Int(subset[1])
    best_z = float(points[best_idx, 3])
    @inbounds for i in 2:length(subset)
        v = Int(subset[i])
        vz = float(points[v, 3])
        if vz > best_z
            best_z = vz
            best_idx = v
        end
    end
    return best_idx
end

function _subset_component_labels(graph::SimpleGraph{Int}, subset::Vector{Int}, min_cc_size::Integer,
                                  cc_workspace::ConnectedComponentSubsetWorkspace)
    isempty(subset) && return Int[]
    return connected_component_subset!(cc_workspace, graph, subset, min_cc_size)
end

"""
    _expand_nearground_cluster(graph, seed, agh_values, agh_ceiling, mask) -> Vector{Int}

BFS from `seed` over vertices where `mask[v]` is true and
`float(agh_values[v]) <= agh_ceiling`. Returns all discovered vertices (including
`seed`) as the connected near-ground cluster.
"""
function _expand_nearground_cluster(
    graph::SimpleGraph{Int},
    seed::Int,
    agh_values::AbstractVector{<:Real},
    agh_ceiling::Float64,
    mask::BitVector,
)
    visited  = falses(nv(graph))
    cluster  = Int[]
    queue    = Int[]

    visited[seed] = true
    push!(cluster, seed)
    push!(queue, seed)

    qi = 1
    while qi <= length(queue)
        v = queue[qi]
        qi += 1
        @inbounds for nbr in Graphs.neighbors(graph, v)
            visited[nbr] && continue
            mask[nbr]    || continue
            float(agh_values[nbr]) <= agh_ceiling || continue
            visited[nbr] = true
            push!(cluster, nbr)
            push!(queue, nbr)
        end
    end

    return cluster
end

function _build_proto_graph(points::AbstractMatrix{<:Real},
                            graph::SimpleGraph{Int},
                            subset_vertices::AbstractVector{<:Integer},
                            subset_mask::BitVector,
                            proto_labels_global::AbstractVector{<:Integer})
    present_labels = Set{Int}()
    @inbounds for v in subset_vertices
        lbl = Int(proto_labels_global[Int(v)])
        lbl > 0 && push!(present_labels, lbl)
    end

    q_labels = sort!(collect(present_labels))
    nq = length(q_labels)
    nq == 0 && return (graph=SimpleGraph{Int}(0), points=zeros(Float64, 0, 3), labels=Int[], edge_vectors=Dict{Tuple{Int, Int}, NTuple{3, Float64}}())

    label_to_vertex = Dict{Int, Int}(lbl => i for (i, lbl) in enumerate(q_labels))
    q_points = zeros(Float64, nq, 3)
    counts = zeros(Int, nq)

    @inbounds for v_any in subset_vertices
        v = Int(v_any)
        lbl = Int(proto_labels_global[v])
        lbl > 0 || continue
        qv = label_to_vertex[lbl]
        q_points[qv, 1] += float(points[v, 1])
        q_points[qv, 2] += float(points[v, 2])
        q_points[qv, 3] += float(points[v, 3])
        counts[qv] += 1
    end

    @inbounds for qv in 1:nq
        q_points[qv, 1] /= counts[qv]
        q_points[qv, 2] /= counts[qv]
        q_points[qv, 3] /= counts[qv]
    end

    q_graph = SimpleGraph(nq)
    pair_vectors = Dict{Tuple{Int, Int}, Vector{NTuple{3, Float64}}}()

    @inbounds for u_any in subset_vertices
        u = Int(u_any)
        subset_mask[u] || continue
        lu = Int(proto_labels_global[u])
        lu > 0 || continue

        qu = label_to_vertex[lu]
        for v in Graphs.neighbors(graph, u)
            u < v || continue
            subset_mask[v] || continue

            lv = Int(proto_labels_global[v])
            lv > 0 || continue
            lu == lv && continue

            qv = label_to_vertex[lv]
            add_edge!(q_graph, qu, qv)

            if qu < qv
                key = (qu, qv)
                vec = (
                    float(points[v, 1]) - float(points[u, 1]),
                    float(points[v, 2]) - float(points[u, 2]),
                    float(points[v, 3]) - float(points[u, 3]),
                )
                push!(get!(pair_vectors, key, NTuple{3, Float64}[]), vec)
            else
                key = (qv, qu)
                vec = (
                    float(points[u, 1]) - float(points[v, 1]),
                    float(points[u, 2]) - float(points[v, 2]),
                    float(points[u, 3]) - float(points[v, 3]),
                )
                push!(get!(pair_vectors, key, NTuple{3, Float64}[]), vec)
            end
        end
    end

    edge_vectors = Dict{Tuple{Int, Int}, NTuple{3, Float64}}()
    for (key, vecs) in pair_vectors
        ux = median(first.(vecs))
        uy = median((v -> v[2]).(vecs))
        uz = median(last.(vecs))
        med = (Float64(ux), Float64(uy), Float64(uz))
        edge_vectors[key] = med
        edge_vectors[(key[2], key[1])] = (-med[1], -med[2], -med[3])
    end

    return (graph=q_graph, points=q_points, labels=q_labels, edge_vectors=edge_vectors)
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
        odd_components = _subset_component_labels(graph, odd_indices, min_cc_size, local_cc_workspace)
        @inbounds for (local_idx, point_idx) in enumerate(odd_indices)
            temp_labels[point_idx] = odd_components[local_idx]
        end
        max_odd_label = isempty(odd_components) ? 0 : maximum(odd_components)
        next_temp_label = max_odd_label + 1
    end

    if !isempty(even_indices)
        even_components = _subset_component_labels(graph, even_indices, min_cc_size, local_cc_workspace)
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

function refine_linear_connected_segment(args...; kwargs...)
    return nothing
end

function assemble_linear_connected_segment(args...; kwargs...)
    return nothing
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

    # Pre-sort by ascending z for O(1) amortised seed selection.
    z_sorted = sortperm(view(points, :, 3); rev=false)
    z_cursor         = 1
    next_id          = 1
    next_global_node = 1

    while next_id - 1 < max_iter
        while z_cursor <= N && !unlabeled_mask[z_sorted[z_cursor]]
            z_cursor += 1
        end
        z_cursor > N && break
        seed = z_sorted[z_cursor]

        # If the seed AGH is within the near-ground ceiling, expand it to the full
        # connected cluster of near-ground points before starting the greedy search.
        start_vertices = if float(agh_values[seed]) <= nearground_agh_ceiling
            _expand_nearground_cluster(graph, seed, agh_values, nearground_agh_ceiling, unlabeled_mask)
        else
            Int[seed]
        end


        result = greedy_connected_neighborhood_search(
            graph, start_vertices, neighbor_distance;
            vertex_mask = unlabeled_mask,
            workspace   = gsws,
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

    pc_output      = setattribute!(pc_filtered, :nbs_id, nbs_res.nbs_id)
    pc_output      = setattribute!(pc_output, :node_id, nbs_res.node_id)
    skeleton_cloud = pc_filtered[1:0]  # placeholder; skeleton extraction not yet implemented

    return (
        filtered_cloud  = pc_filtered,
        pc_output       = pc_output,
        skeleton_cloud  = skeleton_cloud,
        n_components    = 0,
        neighbor_radius = neighbor_radius,
    )
end
