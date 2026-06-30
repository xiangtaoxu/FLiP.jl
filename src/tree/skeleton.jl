"""
Skeleton construction: one vertex per NBS node, edges from the point radius graph.
"""

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

