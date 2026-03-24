"""
    connected_component_labels(points::AbstractMatrix{<:Real}, max_distance::Real,
                               min_cc_size::Integer=1) -> Vector{Int}

Label exact Euclidean connected components in a point cloud.

Two points are connected when their Euclidean distance is less than or equal to
`max_distance`. Connectivity is computed exactly using KDTree radius queries and
union-find over points, without materializing a full graph.

# Arguments
- `points`: N×3 matrix of XYZ coordinates
- `max_distance`: Maximum Euclidean distance for connectivity (must be > 0)
- `min_cc_size`: Minimum component size to keep. Components with fewer points are
    assigned label `0` (must be >= 1)

# Returns
- `Vector{Int}`: Contiguous component labels in `1:k` for retained components,
    where label `1` is the largest retained component; labels increase with
    decreasing retained component size. Components below `min_cc_size` receive `0`.
"""
function connected_component_labels(points::AbstractMatrix{<:Real}, max_distance::Real,
                                    min_cc_size::Integer)
    _validate_graph_points(points)
    max_distance > 0 || throw(ArgumentError("max_distance must be > 0"))
    min_cc_size >= 1 || throw(ArgumentError("min_cc_size must be >= 1"))

    n = size(points, 1)
    n == 0 && return Int[]
    n == 1 && return (min_cc_size == 1 ? [1] : [0])

    tree = _graph_kdtree(points)
    radius = float(max_distance)
    parent = collect(1:n)
    rnk = zeros(Int, n)

    @inbounds for i in 1:n
        neighbors_i = inrange(tree, vec(@view points[i, :]), radius)
        for j in neighbors_i
            j > i || continue
            _uf_union!(parent, rnk, i, j)
        end
    end

    return _compact_component_labels_by_size!(parent, Int(min_cc_size))
end

function connected_component_labels(points::AbstractMatrix{<:Real}, max_distance::Real)
    return connected_component_labels(points, max_distance, 1)
end

"""
    build_radius_graph(points::AbstractMatrix{<:Real}, radius::Real)
        -> NamedTuple{(:graph, :weights), Tuple{SimpleGraph{Int}, SparseMatrixCSC{Float64,Int}}}

Build an undirected radius-neighbor graph for a point cloud.

An edge `(i, j)` is added when the Euclidean distance between points `i` and `j`
is less than or equal to `radius`. The returned sparse weight matrix stores the
Euclidean edge lengths symmetrically.

# Arguments
- `points`: N×3 matrix of XYZ coordinates
- `radius`: Neighborhood radius for edge creation (must be > 0)

# Returns
- `NamedTuple`: `graph` is a `SimpleGraph{Int}` and `weights` is a symmetric sparse
  matrix of Euclidean edge lengths
"""
function build_radius_graph(points::AbstractMatrix{<:Real}, radius::Real)
    _validate_graph_points(points)
    radius > 0 || throw(ArgumentError("radius must be > 0"))

    n = size(points, 1)
    graph = SimpleGraph(n)
    n == 0 && return (graph=graph, weights=spzeros(Float64, 0, 0))

    tree = _graph_kdtree(points)
    search_radius = float(radius)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    @inbounds for i in 1:n
        neighbors_i = inrange(tree, vec(@view points[i, :]), search_radius)
        for j in neighbors_i
            j > i || continue
            add_edge!(graph, i, j)
            dij = _edge_distance(points, i, j)
            push!(rows, i)
            push!(cols, j)
            push!(vals, dij)
            push!(rows, j)
            push!(cols, i)
            push!(vals, dij)
        end
    end

    return (graph=graph, weights=sparse(rows, cols, vals, n, n))
end

"""
    build_radius_graph(points::AbstractMatrix{<:Real}, radius::Real, method::Symbol)
        -> NamedTuple{(:graph, :weights), Tuple{SimpleGraph{Int}, SparseMatrixCSC{Float64,Int}}}

Build an undirected radius-neighbor graph using a selectable construction method.

Supported methods:
- `:iterative`: incremental `add_edge!` construction (same as 2-argument method)
- `:edge_iterator`: collect all edges first and construct graph with
  `SimpleGraphFromIterator`
"""
function build_radius_graph(points::AbstractMatrix{<:Real}, radius::Real, method::Symbol)
    method === :iterative && return build_radius_graph(points, radius)
    method === :edge_iterator && return _build_radius_graph_from_edgelist(points, radius)
    throw(ArgumentError("Unsupported build_radius_graph method: $(method). Use :iterative or :edge_iterator"))
end

function _collect_radius_edges_and_weights(points::AbstractMatrix{<:Real}, radius::Real)
    _validate_graph_points(points)
    radius > 0 || throw(ArgumentError("radius must be > 0"))

    n = size(points, 1)
    n == 0 && return (n=0, edge_tuples=Tuple{Int, Int}[], rows=Int[], cols=Int[], vals=Float64[])

    tree = _graph_kdtree(points)
    search_radius = float(radius)
    edge_tuples = Tuple{Int, Int}[]
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    @inbounds for i in 1:n
        neighbors_i = inrange(tree, vec(@view points[i, :]), search_radius)
        for j in neighbors_i
            j > i || continue
            push!(edge_tuples, (i, j))
            dij = _edge_distance(points, i, j)
            push!(rows, i)
            push!(cols, j)
            push!(vals, dij)
            push!(rows, j)
            push!(cols, i)
            push!(vals, dij)
        end
    end

    return (n=n, edge_tuples=edge_tuples, rows=rows, cols=cols, vals=vals)
end

"""
    _build_radius_graph_from_edgelist(points::AbstractMatrix{<:Real}, radius::Real)
        -> NamedTuple{(:graph, :weights), Tuple{SimpleGraph{Int}, SparseMatrixCSC{Float64,Int}}}

Build an undirected radius-neighbor graph by first collecting all edges and then
constructing a `SimpleGraph` in one pass using `SimpleGraphFromIterator`.

The returned sparse weight matrix stores symmetric Euclidean edge lengths.
"""
function _build_radius_graph_from_edgelist(points::AbstractMatrix{<:Real}, radius::Real)
    data = _collect_radius_edges_and_weights(points, radius)
    n = data.n
    graph = SimpleGraph(n)
    if n > 0
        edge_iter = (Edge{Int}(u, v) for (u, v) in data.edge_tuples)
        graph = SimpleGraphFromIterator(edge_iter)
        nv(graph) < n && add_vertices!(graph, n - nv(graph))
    end

    weights = sparse(data.rows, data.cols, data.vals, n, n)
    return (graph=graph, weights=weights)
end

"""
    build_graph(graph::SimpleGraph{Int}, points::AbstractMatrix{<:Real},
                comp_idx::Vector{Int}, vertex_local_index::Vector{Int})
        -> NamedTuple{(:graph, :weights), Tuple{SimpleGraph{Int}, SparseMatrixCSC{Float64,Int}}}

Build a component-induced graph and symmetric weight matrix for `comp_idx` using
an already-built full graph.

# Arguments
- `graph`: Full graph whose vertices index `points`
- `points`: N×3 matrix of XYZ coordinates associated with `graph`
- `comp_idx`: Global vertex indices for the component to induce
- `vertex_local_index`: Reusable global-to-local index map (length `nv(graph)`)

# Returns
- `NamedTuple`: `graph` is the induced `SimpleGraph{Int}` on the component, and
  `weights` is a symmetric sparse matrix of Euclidean edge lengths
"""
function build_graph(graph::SimpleGraph{Int}, points::AbstractMatrix{<:Real},
                     comp_idx::Vector{Int}, vertex_local_index::Vector{Int})
    _validate_graph_points(points)
    nv(graph) == size(points, 1) || throw(ArgumentError("graph vertex count must match number of points"))
    length(vertex_local_index) == nv(graph) || throw(ArgumentError("vertex_local_index length must equal nv(graph)"))

    n_local = length(comp_idx)
    comp_graph = SimpleGraph(n_local)
    n_local == 0 && return (graph=comp_graph, weights=spzeros(Float64, 0, 0))

    @inbounds for (i, v) in enumerate(comp_idx)
        vertex_local_index[v] = i
    end

    rows = Int[]
    cols = Int[]
    vals = Float64[]
    sizehint!(rows, n_local * 20)
    sizehint!(cols, n_local * 20)
    sizehint!(vals, n_local * 20)

    @inbounds for (new_v, old_v) in enumerate(comp_idx)
        for nbr in Graphs.neighbors(graph, old_v)
            new_nbr = vertex_local_index[nbr]
            if new_v < new_nbr
                add_edge!(comp_graph, new_v, new_nbr)
                dij = _edge_distance(points, old_v, nbr)
                push!(rows, new_v)
                push!(cols, new_nbr)
                push!(vals, dij)
                push!(rows, new_nbr)
                push!(cols, new_v)
                push!(vals, dij)
            end
        end
    end

    @inbounds for v in comp_idx
        vertex_local_index[v] = 0
    end

    return (graph=comp_graph, weights=sparse(rows, cols, vals, n_local, n_local))
end

mutable struct ConnectedComponentSubsetWorkspace
    allowed::BitVector
    visited::BitVector
    labels_global::Vector{Int}
    queue::Vector{Int}
end

function ConnectedComponentSubsetWorkspace(n_vertices::Integer)
    n_vertices >= 0 || throw(ArgumentError("n_vertices must be >= 0"))
    n = Int(n_vertices)
    return ConnectedComponentSubsetWorkspace(falses(n), falses(n), zeros(Int, n), Int[])
end

mutable struct ShortestPathSubsetWorkspace
    allowed::BitVector
    visited::BitVector
    local_index::Vector{Int}
    distances::Vector{Float64}
    parents::Vector{Int}
    heap_vertices::Vector{Int}
    heap_dists::Vector{Float64}
end

function ShortestPathSubsetWorkspace(n_vertices::Integer)
    n_vertices >= 0 || throw(ArgumentError("n_vertices must be >= 0"))
    n = Int(n_vertices)
    return ShortestPathSubsetWorkspace(
        falses(n),
        falses(n),
        zeros(Int, n),
        fill(Inf, n),
        zeros(Int, n),
        Int[],
        Float64[],
    )
end

"""
    GreedySearchWorkspace

Reusable scratch space for `greedy_connected_neighborhood_search`.

Pre-allocate once with `GreedySearchWorkspace(nv(graph))` and pass via the
`workspace` keyword to avoid per-call allocation of O(N) arrays.
"""
mutable struct GreedySearchWorkspace
    mask_allowed::BitVector      # eligible vertices for this call
    included::BitVector          # vertices accepted into the grown region
    in_queue::BitVector          # BFS dedup flag (reset via touched_queue)
    node_id_map::Vector{Int}     # per-vertex frontier wave ID
    touched_queue::Vector{Int}   # indices set in in_queue this BFS round
    layer_a::Vector{Int}         # BFS double-buffer A
    layer_b::Vector{Int}         # BFS double-buffer B
    new_pts::Vector{Int}         # neighborhood candidates collected each round
    new_frontier::Vector{Int}    # accepted next frontier after CC check
    vertices_buf::Vector{Int}    # accumulates all accepted vertices
    cc_ws::ConnectedComponentSubsetWorkspace
    # Incremental per-node centroid accumulators (indexed by node_id, push!-grown)
    node_sum_x::Vector{Float64}
    node_sum_y::Vector{Float64}
    node_sum_z::Vector{Float64}
    node_count::Vector{Int}
    # Per-CC scratch buffers for _find_best_frontier (reused each iteration)
    frontier_cc_cx::Vector{Float64}
    frontier_cc_cy::Vector{Float64}
    frontier_cc_cz::Vector{Float64}
    frontier_cc_count::Vector{Int}
end

function GreedySearchWorkspace(n_vertices::Integer)
    n_vertices >= 0 || throw(ArgumentError("n_vertices must be >= 0"))
    n = Int(n_vertices)
    return GreedySearchWorkspace(
        falses(n), falses(n), falses(n),
        zeros(Int, n),
        sizehint!(Int[], 512),
        sizehint!(Int[], 64), sizehint!(Int[], 64),
        sizehint!(Int[], 256), sizehint!(Int[], 64),
        sizehint!(Int[], 256),
        ConnectedComponentSubsetWorkspace(n),
        sizehint!(Float64[], 64),
        sizehint!(Float64[], 64),
        sizehint!(Float64[], 64),
        sizehint!(Int[], 64),
        sizehint!(Float64[], 8),
        sizehint!(Float64[], 8),
        sizehint!(Float64[], 8),
        sizehint!(Int[], 8),
    )
end

"""
    connected_component_subset!(workspace, graph, subset, min_cc_size=1) -> Vector{Int}

Compute connected-component labels for vertices restricted to `subset` without
constructing an induced subgraph.

Labels are returned in subset order and ranked by decreasing component size.
Components smaller than `min_cc_size` receive label `0`.
"""
function connected_component_subset!(workspace::ConnectedComponentSubsetWorkspace,
                                     graph::SimpleGraph{Int},
                                     subset::AbstractVector{<:Integer},
                                     min_cc_size::Integer=1)
    min_cc_size >= 1 || throw(ArgumentError("min_cc_size must be >= 1"))
    _validate_subset_workspace(workspace, graph)

    # Deduplicate using workspace.allowed as an O(1) BitVector lookup — avoids
    # allocating a Set{Int} over the input. workspace.allowed/visited/labels_global
    # are guaranteed clean (false/zero) at entry by the end-of-call cleanup below
    # (or by the constructor on first use).
    subset_vertices = Int[]
    sizehint!(subset_vertices, length(subset))
    @inbounds for v_raw in subset
        v = Int(v_raw)
        _validate_vertex_index(graph, v, "subset vertex")
        workspace.allowed[v] && continue   # duplicate — skip
        workspace.allowed[v] = true
        push!(subset_vertices, v)
    end
    isempty(subset_vertices) && return Int[]

    components = Vector{Vector{Int}}()
    for seed in subset_vertices
        workspace.visited[seed] && continue

        component = Int[]
        empty!(workspace.queue)
        push!(workspace.queue, seed)
        workspace.visited[seed] = true

        head = 1
        while head <= length(workspace.queue)
            u = workspace.queue[head]
            head += 1
            push!(component, u)

            for nbr in Graphs.neighbors(graph, u)
                workspace.allowed[nbr] || continue
                workspace.visited[nbr] && continue
                workspace.visited[nbr] = true
                push!(workspace.queue, nbr)
            end
        end

        push!(components, component)
    end

    order = sortperm(length.(components); rev=true)
    next_label = 1
    for idx in order
        component = components[idx]
        length(component) >= min_cc_size || continue
        @inbounds for v in component
            workspace.labels_global[v] = next_label
        end
        next_label += 1
    end

    labels_subset = Vector{Int}(undef, length(subset_vertices))
    @inbounds for (i, v) in enumerate(subset_vertices)
        labels_subset[i]           = workspace.labels_global[v]
        # Touched-bit cleanup: reset only visited entries, not all N elements.
        workspace.allowed[v]       = false
        workspace.visited[v]       = false
        workspace.labels_global[v] = 0
    end

    return labels_subset
end

function connected_component_subset!(graph::SimpleGraph{Int},
                                     subset::AbstractVector{<:Integer},
                                     min_cc_size::Integer=1)
    workspace = ConnectedComponentSubsetWorkspace(nv(graph))
    return connected_component_subset!(workspace, graph, subset, min_cc_size)
end

"""
    shortest_path_subset!(workspace, graph, weights, subset, target_idx) -> NamedTuple

Compute Dijkstra shortest-path distances from `target_idx` restricted to vertices
in `subset`, without constructing an induced subgraph.

Returns distances and parents in subset order (`subset` field stores that order).
Parent indices are subset-local (0 for no parent).
"""
function shortest_path_subset!(workspace::ShortestPathSubsetWorkspace,
                               graph::SimpleGraph{Int},
                               weights::SparseMatrixCSC{<:Real,<:Integer},
                               subset::AbstractVector{<:Integer},
                               target_idx::Integer)
    n = nv(graph)
    size(weights) == (n, n) || throw(ArgumentError("weights must be an N×N sparse matrix"))
    _validate_subset_workspace(workspace, graph)

    subset_vertices = _validated_subset_vertices(graph, subset)
    isempty(subset_vertices) && return (distances=Float64[], parents=Int[], target_idx=0, subset=Int[])

    target = Int(target_idx)
    _validate_vertex_index(graph, target, "target_idx")

    fill!(workspace.allowed, false)
    fill!(workspace.visited, false)
    fill!(workspace.local_index, 0)
    fill!(workspace.distances, Inf)
    fill!(workspace.parents, 0)
    empty!(workspace.heap_vertices)
    empty!(workspace.heap_dists)

    @inbounds for (i, v) in enumerate(subset_vertices)
        workspace.allowed[v] = true
        workspace.local_index[v] = i
    end

    workspace.allowed[target] || throw(ArgumentError("target_idx must belong to subset"))

    workspace.distances[target] = 0.0
    _heap_push!(workspace.heap_vertices, workspace.heap_dists, target, 0.0)

    while !isempty(workspace.heap_vertices)
        u, du = _heap_pop_min!(workspace.heap_vertices, workspace.heap_dists)
        workspace.visited[u] && continue
        workspace.visited[u] = true
        du > workspace.distances[u] + 1e-12 && continue

        for nbr in Graphs.neighbors(graph, u)
            workspace.allowed[nbr] || continue
            workspace.visited[nbr] && continue

            w = float(weights[u, nbr])
            w > 0 || continue
            nd = workspace.distances[u] + w
            if nd + 1e-12 < workspace.distances[nbr]
                workspace.distances[nbr] = nd
                workspace.parents[nbr] = u
                _heap_push!(workspace.heap_vertices, workspace.heap_dists, nbr, nd)
            end
        end
    end

    target_local = workspace.local_index[target]
    subset_dists = Vector{Float64}(undef, length(subset_vertices))
    subset_parents = Vector{Int}(undef, length(subset_vertices))

    @inbounds for (i, v) in enumerate(subset_vertices)
        subset_dists[i] = workspace.distances[v]
        parent_v = workspace.parents[v]
        subset_parents[i] = parent_v == 0 ? 0 : workspace.local_index[parent_v]
        workspace.allowed[v] = false
        workspace.local_index[v] = 0
    end

    return (distances=subset_dists, parents=subset_parents, target_idx=target_local, subset=subset_vertices)
end

function shortest_path_subset!(graph::SimpleGraph{Int},
                               weights::SparseMatrixCSC{<:Real,<:Integer},
                               subset::AbstractVector{<:Integer},
                               target_idx::Integer)
    workspace = ShortestPathSubsetWorkspace(nv(graph))
    return shortest_path_subset!(workspace, graph, weights, subset, target_idx)
end

"""
    quotient_graph(points::AbstractMatrix{<:Real}, graph::SimpleGraph{Int},
                   labels::AbstractVector{<:Integer}) -> NamedTuple

Build a quotient graph by collapsing vertices that share the same label.

Quotient vertices are ordered by ascending unique label value. The point associated
with each quotient vertex is the centroid of all input points carrying that label.
Intra-label edges are removed; inter-label edges are collapsed into a single
undirected edge between quotient vertices.

# Arguments
- `points`: N×3 matrix of XYZ coordinates associated with the input graph vertices
- `graph`: Input graph with `N` vertices
- `labels`: Length-`N` integer labels defining the quotient partition

# Returns
- `NamedTuple` with fields:
  - `graph`: Quotient `SimpleGraph{Int}`
  - `points`: Quotient point cloud as a K×3 centroid matrix
  - `labels`: Sorted unique labels corresponding to quotient vertices
    - `edge_vectors`: Directed edge vectors between quotient vertices. For each
        quotient edge `(u, v)`, the vector is the component-wise median of all
        full-cloud edge vectors crossing the associated label pair, with both
        directions populated (`(u, v)` and `(v, u) = -(u, v)`).
"""
function quotient_graph(points::AbstractMatrix{<:Real}, graph::SimpleGraph{Int},
                        labels::AbstractVector{<:Integer})
    _validate_quotient_graph_inputs(points, graph, labels)

    n = size(points, 1)
        n == 0 && return (graph=SimpleGraph{Int}(0), points=zeros(Float64, 0, 3), labels=Int[], edge_vectors=Dict{Tuple{Int, Int}, NTuple{3, Float64}}())

    quotient_labels = sort!(collect(unique(Int.(labels))))
    nq = length(quotient_labels)
    label_to_vertex = Dict{Int, Int}(label => vertex for (vertex, label) in enumerate(quotient_labels))

    quotient_points = zeros(Float64, nq, 3)
    counts = zeros(Int, nq)

    @inbounds for i in 1:n
        qv = label_to_vertex[Int(labels[i])]
        quotient_points[qv, 1] += float(points[i, 1])
        quotient_points[qv, 2] += float(points[i, 2])
        quotient_points[qv, 3] += float(points[i, 3])
        counts[qv] += 1
    end

    @inbounds for qv in 1:nq
        quotient_points[qv, 1] /= counts[qv]
        quotient_points[qv, 2] /= counts[qv]
        quotient_points[qv, 3] /= counts[qv]
    end

    quotient = SimpleGraph(nq)
    pair_vectors = Dict{Tuple{Int, Int}, Vector{NTuple{3, Float64}}}()
    for edge in Graphs.edges(graph)
        src_idx = Graphs.src(edge)
        dst_idx = Graphs.dst(edge)
        src_vertex = label_to_vertex[Int(labels[src_idx])]
        dst_vertex = label_to_vertex[Int(labels[dst_idx])]
        src_vertex == dst_vertex && continue
        add_edge!(quotient, src_vertex, dst_vertex)

        if src_vertex < dst_vertex
            key = (src_vertex, dst_vertex)
            vec = (
                float(points[dst_idx, 1]) - float(points[src_idx, 1]),
                float(points[dst_idx, 2]) - float(points[src_idx, 2]),
                float(points[dst_idx, 3]) - float(points[src_idx, 3]),
            )
            push!(get!(pair_vectors, key, NTuple{3, Float64}[]), vec)
        else
            key = (dst_vertex, src_vertex)
            vec = (
                float(points[src_idx, 1]) - float(points[dst_idx, 1]),
                float(points[src_idx, 2]) - float(points[dst_idx, 2]),
                float(points[src_idx, 3]) - float(points[dst_idx, 3]),
            )
            push!(get!(pair_vectors, key, NTuple{3, Float64}[]), vec)
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

    return (graph=quotient, points=quotient_points, labels=quotient_labels, edge_vectors=edge_vectors)
end

"""
    shortest_path_distances(points, graph, weights, target_idx::Integer)
        -> NamedTuple

Compute Euclidean shortest-path distances from all graph vertices to `target_idx`.

# Returns
- `NamedTuple` with fields `distances`, `parents`, `state`, and `target_idx`
"""
function shortest_path_distances(points::AbstractMatrix{<:Real}, graph::SimpleGraph{Int},
                                 weights::SparseMatrixCSC{<:Real,<:Integer}, target_idx::Integer)
    _validate_shortest_path_inputs(points, graph, weights)
    _validate_vertex_index(graph, target_idx, "target_idx")

    state = dijkstra_shortest_paths(graph, Int(target_idx), weights)
    return (distances=state.dists, parents=state.parents, state=state, target_idx=Int(target_idx))
end

"""
    shortest_path_distances(points, graph, weights, target_point::AbstractVector{<:Real})
        -> NamedTuple

Compute Euclidean shortest-path distances from all graph vertices to the graph vertex
nearest to `target_point`.
"""
function shortest_path_distances(points::AbstractMatrix{<:Real}, graph::SimpleGraph{Int},
                                 weights::SparseMatrixCSC{<:Real,<:Integer},
                                 target_point::AbstractVector{<:Real})
    _validate_graph_points(points)
    target_idx = _nearest_vertex_index(points, target_point)
    return shortest_path_distances(points, graph, weights, target_idx)
end

"""
    slice_by_shortest_path(points, graph, weights, target, slice_length::Real) -> NamedTuple

Assign slice labels based on cumulative shortest-path distance to a target.

Finite shortest-path distances are converted to 1-based slice labels using
`floor(distance / slice_length) + 1`. Unreachable vertices receive label `0`.

# Returns
- `NamedTuple` with fields `slice_labels`, `distances`, `parents`, and `target_idx`
"""
function slice_by_shortest_path(points::AbstractMatrix{<:Real}, graph::SimpleGraph{Int},
                                weights::SparseMatrixCSC{<:Real,<:Integer},
                                target::Union{Integer, AbstractVector{<:Real}}, slice_length::Real)
    slice_length > 0 || throw(ArgumentError("slice_length must be > 0"))

    sp = shortest_path_distances(points, graph, weights, target)
    labels = zeros(Int, length(sp.distances))

    @inbounds for i in eachindex(sp.distances)
        di = sp.distances[i]
        if isfinite(di)
            labels[i] = floor(Int, di / float(slice_length)) + 1
        end
    end

    return (slice_labels=labels, distances=sp.distances, parents=sp.parents, target_idx=sp.target_idx)
end

"""
    generate_proto_nodes_from_slice_label(points::AbstractMatrix{<:Real},
                                          slice_labels::AbstractVector{<:Integer},
                                          max_distance::Real,
                                          min_cc_size::Integer=1) -> Vector{Int}

Generate proto-node labels from shortest-path slice labels.

Points with positive odd `slice_label` values are grouped together and labeled by
Euclidean connected components using `connected_component_labels`. The same process
is repeated for positive even `slice_label` values. Points with `slice_label == 0`
keep `proto_node == 0`.

Final proto-node labels are globally unique and ordered by ascending `slice_label`.
Within a slice, component ordering follows first occurrence in input point order.
Components that span multiple slices retain the same proto-node label across those
slices.

# Arguments
- `points`: N×3 matrix of XYZ coordinates
- `slice_labels`: Length-`N` vector of slice labels, typically from `slice_by_shortest_path`
- `max_distance`: Maximum Euclidean distance for connected-component labeling
- `min_cc_size`: Minimum component size to keep when computing odd/even connected
    components. Components with fewer points are assigned `0`.

# Returns
- `Vector{Int}`: Proto-node labels for each point. Zero-labeled slices remain `0`.
"""
function generate_proto_nodes_from_slice_label(points::AbstractMatrix{<:Real},
                                               slice_labels::AbstractVector{<:Integer},
                                               max_distance::Real,
                                               min_cc_size::Integer=1)
    _validate_graph_points(points)
    max_distance > 0 || throw(ArgumentError("max_distance must be > 0"))
    min_cc_size >= 1 || throw(ArgumentError("min_cc_size must be >= 1"))

    n = size(points, 1)
    length(slice_labels) == n || throw(ArgumentError("slice_labels length must match number of points"))
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

    next_temp_label = 1
    if !isempty(odd_indices)
        odd_components = connected_component_labels(points[odd_indices, :], max_distance, min_cc_size)
        @inbounds for (local_idx, point_idx) in enumerate(odd_indices)
            temp_labels[point_idx] = odd_components[local_idx]
        end
        max_odd_label = isempty(odd_components) ? 0 : maximum(odd_components)
        next_temp_label = max_odd_label + 1
    end

    if !isempty(even_indices)
        even_components = connected_component_labels(points[even_indices, :], max_distance, min_cc_size)
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
    longest_linear_path(graph::SimpleGraph{Int}, points::AbstractMatrix{<:Real}, root::Integer;
                        edge_vectors=nothing)
        -> NamedTuple{(:vertices, :length), Tuple{Vector{Int}, Float64}}

Extract a root-origin path from an undirected graph using angle-first neighbor selection.

Starting at `root`, the first step selects the unvisited neighbor with the smallest Z
value (tie-broken by shortest edge distance). Subsequent steps select the most linear
unvisited neighbor whose edge is within 60° of the incoming direction; if no neighbor
qualifies, the path terminates. If `edge_vectors[(u, v)]` is provided, these vectors
are used for linearity filtering; otherwise point-coordinate edge vectors are used.
Already-visited vertices are never revisited.
"""
function longest_linear_path(graph::SimpleGraph{Int}, points::AbstractMatrix{<:Real}, root::Integer;
                             edge_vectors::Union{Nothing, AbstractDict{Tuple{Int, Int}, <:NTuple{3, <:Real}}}=nothing)
    _validate_graph_points(points)
    nv(graph) == size(points, 1) || throw(ArgumentError("graph vertex count must match number of points"))
    _validate_vertex_index(graph, root, "root")

    path = Int[Int(root)]
    total_length = 0.0
    previous = 0
    current = Int(root)
    visited = falses(nv(graph))
    visited[current] = true

    while true
        unvisited = [n for n in Graphs.neighbors(graph, current) if !visited[n]]
        isempty(unvisited) && break

        next_vertex = if previous == 0
            _select_min_z_neighbor(current, unvisited, points)
        else
            _select_linear_neighbor(previous, current, unvisited, points, edge_vectors)
        end
        isnothing(next_vertex) && break

        visited[next_vertex] = true
        total_length += _edge_distance(points, current, next_vertex)
        push!(path, next_vertex)
        previous, current = current, next_vertex
    end

    return (vertices=path, length=total_length)
end

"""
    greedy_connected_neighborhood_search(graph, start_vertices, neighbor_distance;
                                         vertex_mask=nothing, workspace=nothing,
                                         points=nothing, linearity_angle_deg=80.0,
                                         min_frontier_cc_size=1)
        -> NamedTuple{(:vertices, :node_ids, :max_node_id)}

Greedy connectivity expansion starting from one or more seed vertices `start_vertices`.

All vertices in `start_vertices` are treated as node_id 1 (the initial frontier).
Each iteration performs a BFS of up to `neighbor_distance` hops from the current
frontier (graph edge distance only — no spatial coordinates required), restricted
to vertices in `vertex_mask`. The collected neighborhood vertices are checked for
connectivity, and only the largest connected component is accepted as the new
frontier. Expansion continues until no new vertices can be reached.

Each accepted frontier wave is assigned a monotonically increasing `node_id`
(starting from 1 for all seed vertices in `start_vertices`). This records the
sequential structure of the expanded path.

# Arguments
- `graph`: The graph to search
- `start_vertices`: Seed vertices for the expansion (`AbstractVector{<:Integer}`).
  All must be valid vertex indices within `vertex_mask`. Duplicates are silently ignored.
- `neighbor_distance`: Maximum number of hops (graph edges) used to expand the frontier
  each iteration

# Keyword Arguments
- `vertex_mask`: Eligible vertices. Accepts a `Vector{<:Integer}` of global vertex IDs,
  a `BitVector` of length `nv(graph)` (fastest — avoids per-call `falses(n)` rebuild),
  or `nothing` (all vertices eligible).
- `workspace`: Optional pre-allocated `GreedySearchWorkspace(nv(graph))` for zero
  per-call allocation. Strongly recommended when calling repeatedly.
- `points`: Optional N×3 coordinate matrix. When provided, enables angular
  filtering of frontier connected components against the NBS growth direction.
  Default: `nothing` (no angular filtering).
- `linearity_angle_deg`: Maximum angular deviation (degrees) between a CC's
  growth direction and the NBS principal direction. CCs exceeding this threshold
  are excluded from the frontier. Only active when `points` is provided.
  Default: `80.0`.
- `min_frontier_cc_size`: Minimum number of points a frontier connected component
  must contain to be considered. Smaller CCs are discarded. Default: `1`.

# Returns
- `NamedTuple` with fields:
  - `vertices`: `Vector{Int}` of global vertex indices in the grown region
  - `node_ids`: `Vector{Int}` of per-vertex frontier IDs (same order as `vertices`);
    `node_id == 1` for all seed vertices, incrementing by 1 for each accepted frontier wave
  - `max_node_id`: highest frontier ID assigned (equals number of frontier waves)
"""
function greedy_connected_neighborhood_search(
    graph::SimpleGraph{Int},
    start_vertices::AbstractVector{<:Integer},
    neighbor_distance::Integer;
    vertex_mask::Union{Nothing, AbstractVector{<:Integer}, BitVector}=nothing,
    workspace::Union{Nothing, GreedySearchWorkspace}=nothing,
    points::Union{Nothing, AbstractMatrix{<:Real}}=nothing,
    linearity_angle_deg::Float64=80.0,
    min_frontier_cc_size::Int=1,
)
    neighbor_distance >= 1 || throw(ArgumentError("neighbor_distance must be >= 1"))
    isempty(start_vertices) && throw(ArgumentError("start_vertices must not be empty"))

    n = nv(graph)
    if points !== nothing
        size(points, 1) == n || throw(ArgumentError("points row count must match nv(graph)"))
    end

    ws = workspace !== nothing ? workspace : GreedySearchWorkspace(n)
    length(ws.mask_allowed) == n || throw(ArgumentError("workspace size must match nv(graph)"))

    # Build mask_allowed — always fully specified each call, so no inter-call cleanup needed.
    if vertex_mask === nothing
        fill!(ws.mask_allowed, true)
    elseif vertex_mask isa BitVector
        length(vertex_mask) == n || throw(ArgumentError("vertex_mask length must equal nv(graph)"))
        ws.mask_allowed .= vertex_mask     # fast bit-copy O(N/64)
    else
        fill!(ws.mask_allowed, false)
        @inbounds for v in vertex_mask
            v_i = Int(v)
            1 <= v_i <= n || throw(ArgumentError("vertex_mask contains out-of-range vertex $v_i"))
            ws.mask_allowed[v_i] = true
        end
    end

    # Initialise per-call state — clean only the seed entries; everything else is
    # cleaned at the end of the previous call (or freshly zero-constructed).
    empty!(ws.vertices_buf)
    empty!(ws.node_sum_x)
    empty!(ws.node_sum_y)
    empty!(ws.node_sum_z)
    empty!(ws.node_count)
    frontier = ws.new_frontier
    empty!(frontier)
    next_node_id = 1

    @inbounds for sv_raw in start_vertices
        sv = Int(sv_raw)
        1 <= sv <= n || throw(ArgumentError("start_vertices contains out-of-range vertex $sv"))
        ws.mask_allowed[sv] || throw(ArgumentError("start_vertex $sv must be within vertex_mask"))
        ws.included[sv] && continue   # skip duplicates silently
        ws.included[sv]    = true
        ws.node_id_map[sv] = 1
        push!(ws.vertices_buf, sv)
        push!(frontier, sv)
    end

    # Accumulate centroids for start vertices (node_id = 1)
    if points !== nothing
        push!(ws.node_sum_x, 0.0)
        push!(ws.node_sum_y, 0.0)
        push!(ws.node_sum_z, 0.0)
        push!(ws.node_count, 0)
        @inbounds for v in ws.vertices_buf
            ws.node_sum_x[1] += Float64(points[v, 1])
            ws.node_sum_y[1] += Float64(points[v, 2])
            ws.node_sum_z[1] += Float64(points[v, 3])
            ws.node_count[1] += 1
        end
    end

    while true
        # --- Load frontier into layer_a, prepare double-buffer layer_b ---
        layer     = ws.layer_a
        alt_layer = ws.layer_b
        empty!(layer)
        @inbounds for v in frontier
            push!(layer, v)
        end

        empty!(ws.new_pts)
        empty!(ws.touched_queue)

        # Seed in_queue: O(|frontier|), not O(N)
        @inbounds for v in layer
            ws.in_queue[v] = true
            push!(ws.touched_queue, v)
        end

        # Layered BFS up to neighbor_distance hops
        for _hop in 1:neighbor_distance
            empty!(alt_layer)
            @inbounds for v in layer
                for nbr in Graphs.neighbors(graph, v)
                    ws.in_queue[nbr] && continue
                    ws.in_queue[nbr] = true
                    push!(ws.touched_queue, nbr)
                    ws.mask_allowed[nbr] || continue
                    ws.included[nbr]     && continue
                    push!(alt_layer, nbr)
                    push!(ws.new_pts, nbr)
                end
            end
            isempty(alt_layer) && break
            layer, alt_layer = alt_layer, layer   # swap; no allocation
        end

        # Reset only the touched in_queue bits (O(|touched|) << O(N))
        @inbounds for v in ws.touched_queue
            ws.in_queue[v] = false
        end

        isempty(ws.new_pts) && break

        # --- Connectivity check: keep only the largest connected component ---
        cc = connected_component_subset!(ws.cc_ws, graph, ws.new_pts)
        next_node_id += 1
        empty!(frontier)

        # Extend centroid cache for the new node_id
        if points !== nothing
            push!(ws.node_sum_x, 0.0)
            push!(ws.node_sum_y, 0.0)
            push!(ws.node_sum_z, 0.0)
            push!(ws.node_count, 0)
        end

        max_cc_original = maximum(cc)

        # Find best frontier CC: size filter + linearity rejection + quality metric
        best_cc = _find_best_frontier(cc, ws, points, next_node_id,
                                       min_frontier_cc_size, linearity_angle_deg)

        # Retrospective branch refinement: triggered by pre-filter CC count,
        # uses post-filter best_cc as cc_chosen.
        if max_cc_original > 1 && best_cc > 0
            cc_chosen_verts = Int[ws.new_pts[i] for i in eachindex(ws.new_pts) if cc[i] == best_cc]
            cc_other_verts  = Int[ws.new_pts[i] for i in eachindex(ws.new_pts) if cc[i] != best_cc]
            _refine_branching(graph, ws, cc_chosen_verts, cc_other_verts,
                              next_node_id, neighbor_distance; points=points)
        end

        @inbounds for (i, v) in enumerate(ws.new_pts)
            ws.included[v] = true        # always mark — prevents rediscovery in future iterations
            push!(ws.vertices_buf, v)    # always track — needed for cleanup
            if cc[i] == best_cc && best_cc > 0
                ws.node_id_map[v] = next_node_id
                push!(frontier, v)
                # Accumulate centroid for the new node
                if points !== nothing
                    ws.node_sum_x[next_node_id] += Float64(points[v, 1])
                    ws.node_sum_y[next_node_id] += Float64(points[v, 2])
                    ws.node_sum_z[next_node_id] += Float64(points[v, 3])
                    ws.node_count[next_node_id] += 1
                end
            end
        end

        isempty(frontier) && break
    end

    # Extract results — filter out discarded vertices (node_id_map == 0)
    vertices = sizehint!(Int[], length(ws.vertices_buf))
    node_ids = sizehint!(Int[], length(ws.vertices_buf))
    rejected = sizehint!(Int[], 64)
    @inbounds for v in ws.vertices_buf
        nid = ws.node_id_map[v]
        if nid == 0
            push!(rejected, v)   # unchosen frontier CC vertex
        else
            push!(vertices, v)
            push!(node_ids, nid)
        end
    end

    # Cleanup only touched entries — O(segment size), not O(N)
    @inbounds for v in ws.vertices_buf
        ws.included[v]    = false
        ws.node_id_map[v] = 0
    end

    return (vertices=vertices, node_ids=node_ids, max_node_id=next_node_id,
            rejected_frontier=rejected)
end

"""
    _compute_nbs_direction(ws, next_node_id; min_nodes=3)
        -> Union{Nothing, NTuple{3,Float64}}

Compute the principal growth direction of the NBS by PCA on cached node
centroids. Reads per-node coordinate sums from `ws.node_sum_x/y/z` and
`ws.node_count` (populated incrementally in the main loop). Builds a 3×3
covariance matrix and uses `eigen(Symmetric(...))` instead of full SVD.

Returns the first principal component as a 3-tuple oriented from earliest
to latest node, or `nothing` if fewer than `min_nodes` valid centroids exist.
"""
function _compute_nbs_direction(
    ws::GreedySearchWorkspace,
    next_node_id::Int;
    min_nodes::Int=3,
)
    n_nodes = next_node_id - 1
    n_nodes < min_nodes && return nothing

    # Compute mean centroid from cached sums
    valid_count = 0
    mean_x = 0.0; mean_y = 0.0; mean_z = 0.0
    @inbounds for nid in 1:n_nodes
        ws.node_count[nid] > 0 || continue
        valid_count += 1
        inv_c = 1.0 / ws.node_count[nid]
        mean_x += ws.node_sum_x[nid] * inv_c
        mean_y += ws.node_sum_y[nid] * inv_c
        mean_z += ws.node_sum_z[nid] * inv_c
    end
    valid_count < min_nodes && return nothing
    inv_n = 1.0 / valid_count
    mean_x *= inv_n; mean_y *= inv_n; mean_z *= inv_n

    # Build 3×3 covariance matrix directly (O(n_nodes), not O(|vertices_buf|))
    c11 = 0.0; c12 = 0.0; c13 = 0.0
    c22 = 0.0; c23 = 0.0; c33 = 0.0
    @inbounds for nid in 1:n_nodes
        ws.node_count[nid] > 0 || continue
        inv_c = 1.0 / ws.node_count[nid]
        dx = ws.node_sum_x[nid] * inv_c - mean_x
        dy = ws.node_sum_y[nid] * inv_c - mean_y
        dz = ws.node_sum_z[nid] * inv_c - mean_z
        c11 += dx * dx; c12 += dx * dy; c13 += dx * dz
        c22 += dy * dy; c23 += dy * dz
        c33 += dz * dz
    end

    # eigen on Symmetric 3×3 — eigenvalues in ascending order
    C = Symmetric([c11 c12 c13; c12 c22 c23; c13 c23 c33])
    F = eigen(C)
    # Largest eigenvalue is last; its eigenvector is PC1
    dvec = (F.vectors[1, 3], F.vectors[2, 3], F.vectors[3, 3])

    # Orient: project first and last valid centroids onto PC1;
    # flip if last centroid's projection is smaller (direction should point
    # from earliest to latest node).
    first_nid = 0
    last_nid  = 0
    @inbounds for nid in 1:n_nodes
        ws.node_count[nid] > 0 || continue
        if first_nid == 0; first_nid = nid; end
        last_nid = nid
    end
    inv_f = 1.0 / ws.node_count[first_nid]
    inv_l = 1.0 / ws.node_count[last_nid]
    proj_first = (ws.node_sum_x[first_nid] * inv_f - mean_x) * dvec[1] +
                 (ws.node_sum_y[first_nid] * inv_f - mean_y) * dvec[2] +
                 (ws.node_sum_z[first_nid] * inv_f - mean_z) * dvec[3]
    proj_last  = (ws.node_sum_x[last_nid] * inv_l - mean_x) * dvec[1] +
                 (ws.node_sum_y[last_nid] * inv_l - mean_y) * dvec[2] +
                 (ws.node_sum_z[last_nid] * inv_l - mean_z) * dvec[3]
    if proj_last < proj_first
        dvec = (-dvec[1], -dvec[2], -dvec[3])
    end

    return dvec
end

"""
    _find_best_frontier(cc, ws, points, next_node_id, min_cc_size,
                        linearity_angle_deg) -> Int

Select the best frontier connected component from `cc`, mutating `cc` in-place
to zero out all non-chosen labels. Internally computes the NBS growth direction
via `_compute_nbs_direction` when enough nodes are available.

1. **Size filter**: CCs with fewer than `min_cc_size` points are discarded.
2. **Quality selection** (when direction is available, i.e. ≥ 3 prior nodes):
   reject CCs whose growth direction deviates more than `linearity_angle_deg`
   from the NBS principal direction, then pick the CC with the highest quality
   metric `cc_count × cos_angle`. Returns `0` if all CCs are rejected.
3. **Early-stage fallback** (when direction is unavailable or `points` is
   `nothing`): pick the largest surviving CC (smallest nonzero label).

Returns the chosen CC label (`0` if no CCs survive filtering).
"""
function _find_best_frontier(
    cc::Vector{Int},
    ws::GreedySearchWorkspace,
    points::Union{Nothing, AbstractMatrix{<:Real}},
    next_node_id::Int,
    min_cc_size::Int,
    linearity_angle_deg::Float64,
)::Int
    n_cc = maximum(cc; init=0)
    n_cc < 1 && return 0

    # --- Per-CC point count and centroid (reuse workspace buffers) ---
    resize!(ws.frontier_cc_cx, n_cc)
    resize!(ws.frontier_cc_cy, n_cc)
    resize!(ws.frontier_cc_cz, n_cc)
    resize!(ws.frontier_cc_count, n_cc)
    fill!(ws.frontier_cc_cx, 0.0)
    fill!(ws.frontier_cc_cy, 0.0)
    fill!(ws.frontier_cc_cz, 0.0)
    fill!(ws.frontier_cc_count, 0)
    cc_cx = ws.frontier_cc_cx
    cc_cy = ws.frontier_cc_cy
    cc_cz = ws.frontier_cc_cz
    cc_count = ws.frontier_cc_count
    if points !== nothing
        @inbounds for (i, v) in enumerate(ws.new_pts)
            label = cc[i]
            label >= 1 || continue
            cc_count[label] += 1
            cc_cx[label] += Float64(points[v, 1])
            cc_cy[label] += Float64(points[v, 2])
            cc_cz[label] += Float64(points[v, 3])
        end
    else
        @inbounds for (i, _) in enumerate(ws.new_pts)
            label = cc[i]
            label >= 1 || continue
            cc_count[label] += 1
        end
    end

    # --- Size filter: zero out small CCs ---
    size_reject = UInt64(0)
    @inbounds for c in 1:min(n_cc, 64)
        if cc_count[c] < min_cc_size
            size_reject |= UInt64(1) << (c - 1)
        end
    end
    if size_reject != UInt64(0)
        @inbounds for i in eachindex(cc)
            c = cc[i]
            if c > 0 && c <= 64 && (size_reject >> (c - 1)) & UInt64(1) == UInt64(1)
                cc[i] = 0
            end
        end
    end

    # --- Largest remaining CC (smallest nonzero label = largest by convention) ---
    largest_cc = 0
    @inbounds for c in 1:n_cc
        if cc_count[c] >= min_cc_size
            largest_cc = c
            break
        end
    end
    largest_cc == 0 && return 0

    # --- Compute NBS direction (needs ≥ 3 prior nodes and points) ---
    nbs_dvec = nothing
    if points !== nothing && next_node_id > 3
        nbs_dvec = _compute_nbs_direction(ws, next_node_id)
    end

    # No direction available (early iterations or no points): return largest CC
    if nbs_dvec === nothing
        return largest_cc
    end

    # --- Previous node centroid from cache (O(1)) ---
    prev_nid = next_node_id - 1
    if prev_nid < 1 || prev_nid > length(ws.node_count) || ws.node_count[prev_nid] <= 0
        return largest_cc
    end
    inv_pc = 1.0 / ws.node_count[prev_nid]
    prev_cx = ws.node_sum_x[prev_nid] * inv_pc
    prev_cy = ws.node_sum_y[prev_nid] * inv_pc
    prev_cz = ws.node_sum_z[prev_nid] * inv_pc

    # --- Find best CC by quality metric: cc_count × cos_angle ---
    cos_threshold = cosd(linearity_angle_deg)
    best_cc = 0
    best_quality = -Inf
    @inbounds for c in 1:min(n_cc, 64)
        cc_count[c] >= min_cc_size || continue
        inv_c = 1.0 / cc_count[c]
        dx = cc_cx[c] * inv_c - prev_cx
        dy = cc_cy[c] * inv_c - prev_cy
        dz = cc_cz[c] * inv_c - prev_cz
        norm_d = sqrt(dx * dx + dy * dy + dz * dz)
        norm_d < 1e-12 && continue
        inv_norm = 1.0 / norm_d
        cos_angle = (dx * nbs_dvec[1] + dy * nbs_dvec[2] + dz * nbs_dvec[3]) * inv_norm
        cos_angle < cos_threshold && continue   # linearity angle rejection
        quality = cc_count[c] * cos_angle
        if quality > best_quality
            best_quality = quality
            best_cc = c
        end
    end

    # All CCs rejected by linearity filter → return 0 (terminates NBS growth)
    best_cc == 0 && return 0

    # Zero out non-best CC labels
    @inbounds for i in eachindex(cc)
        c = cc[i]
        if c != best_cc
            cc[i] = 0
        end
    end

    return best_cc
end

"""
Lightweight BFS workspace with touched-vertex cleanup for subset-restricted
multi-source BFS. Vectors are sized to `nv(graph)` but only touched entries
are reset between calls — O(|touched|) cleanup, not O(N).
"""
mutable struct _BFSWorkspace
    allowed::BitVector       # subset membership
    visited::BitVector       # BFS visited flag
    distances::Vector{Int}   # hop distances; -1 = unreached
    queue::Vector{Int}       # BFS queue (reused across calls)
    touched::Vector{Int}     # vertices touched this call (for cleanup)
end

function _BFSWorkspace(n::Int)
    ws = _BFSWorkspace(falses(n), falses(n), fill(-1, n),
                       sizehint!(Int[], 256), sizehint!(Int[], 256))
    return ws
end


"""
    _cc_diameter_hops(graph, cc_vertices, bfs_ws) -> Int

Approximate the diameter of `cc_vertices` in hop count using double-BFS:
BFS from an arbitrary vertex to find the farthest, then BFS from the farthest.
Uses `_bfs_run_and_read!` / `_bfs_cleanup!` to read distances before resetting.
"""
function _cc_diameter_hops(graph::SimpleGraph{Int},
                                cc_vertices::AbstractVector{<:Integer},
                                bfs_ws::_BFSWorkspace)
    # --- First pass: BFS from arbitrary vertex ---
    empty!(bfs_ws.queue)
    empty!(bfs_ws.touched)
    @inbounds for v in cc_vertices
        bfs_ws.allowed[v] = true
    end
    start = Int(cc_vertices[1])
    bfs_ws.visited[start] = true
    bfs_ws.distances[start] = 0
    push!(bfs_ws.queue, start)
    push!(bfs_ws.touched, start)

    head = 1
    @inbounds while head <= length(bfs_ws.queue)
        u = bfs_ws.queue[head]
        head += 1
        du = bfs_ws.distances[u]
        for nbr in Graphs.neighbors(graph, u)
            bfs_ws.allowed[nbr] || continue
            bfs_ws.visited[nbr] && continue
            bfs_ws.visited[nbr] = true
            bfs_ws.distances[nbr] = du + 1
            push!(bfs_ws.queue, nbr)
            push!(bfs_ws.touched, nbr)
        end
    end

    # Find farthest vertex
    farthest = start
    max_d = 0
    @inbounds for v in bfs_ws.touched
        d = bfs_ws.distances[v]
        if d > max_d
            max_d = d
            farthest = v
        end
    end

    # Cleanup first pass
    @inbounds for v in bfs_ws.touched
        bfs_ws.visited[v] = false
        bfs_ws.distances[v] = -1
    end
    @inbounds for v in cc_vertices
        bfs_ws.allowed[v] = false
    end

    # --- Second pass: BFS from farthest ---
    empty!(bfs_ws.queue)
    empty!(bfs_ws.touched)
    @inbounds for v in cc_vertices
        bfs_ws.allowed[v] = true
    end
    bfs_ws.visited[farthest] = true
    bfs_ws.distances[farthest] = 0
    push!(bfs_ws.queue, farthest)
    push!(bfs_ws.touched, farthest)

    head = 1
    @inbounds while head <= length(bfs_ws.queue)
        u = bfs_ws.queue[head]
        head += 1
        du = bfs_ws.distances[u]
        for nbr in Graphs.neighbors(graph, u)
            bfs_ws.allowed[nbr] || continue
            bfs_ws.visited[nbr] && continue
            bfs_ws.visited[nbr] = true
            bfs_ws.distances[nbr] = du + 1
            push!(bfs_ws.queue, nbr)
            push!(bfs_ws.touched, nbr)
        end
    end

    # Read diameter
    diam = 0
    @inbounds for v in bfs_ws.touched
        d = bfs_ws.distances[v]
        d > diam && (diam = d)
    end

    # Cleanup second pass
    @inbounds for v in bfs_ws.touched
        bfs_ws.visited[v] = false
        bfs_ws.distances[v] = -1
    end
    @inbounds for v in cc_vertices
        bfs_ws.allowed[v] = false
    end

    return diam
end

"""
    _refine_branching(graph, ws, cc_chosen, cc_other, next_node_id, neighbor_distance; points=nothing) -> Int

Retrospectively reassign ambiguous vertices near a branch point using geodesic
distance tiebreaking with pure upstream anchors.

When a branching is detected (multiple CCs in frontier), vertices from recent
waves may belong to the discarded branch. This function identifies the ambiguous
zone (adaptive, based on CC diameter), computes hop distances to both CCs and to
pure upstream anchors via multi-source BFS, then zeroes `node_id_map` for vertices
that are closer to the discarded CC.

Returns the number of reassigned vertices.

Modifies `ws.node_id_map` in-place (sets reassigned vertices to 0).
Does NOT modify `ws.included` — reassigned vertices remain marked to prevent
re-exploration.
"""
function _refine_branching(graph::SimpleGraph{Int},
                           ws::GreedySearchWorkspace,
                           cc_chosen::Vector{Int},
                           cc_other::Vector{Int},
                           next_node_id::Int,
                           neighbor_distance::Int;
                           points::Union{Nothing, AbstractMatrix{<:Real}}=nothing)
    # Early exit: need at least 4 waves (2 pure + 1 ambiguous + current frontier)
    next_node_id < 4 && return 0

    # --- Determine ambiguous zone size from CC diameter ---
    bfs_ws = _BFSWorkspace(nv(graph))
    diameter = _cc_diameter_hops(graph, cc_chosen, bfs_ws)
    n_ambiguous_waves = max(1, diameter ÷ neighbor_distance)
    n_pure_waves = 2

    # Wave IDs: seed=1, first expansion=2, ..., last accepted=next_node_id-1
    # (next_node_id is the current frontier wave being assigned to cc_chosen)
    last_accepted = next_node_id - 1
    first_ambiguous_node = last_accepted - n_ambiguous_waves + 1

    if first_ambiguous_node < 2
        first_pure_node = 1
        first_ambiguous_node = min(3, last_accepted)
    else
        first_pure_node = first_ambiguous_node - n_pure_waves
        if first_pure_node < 1
            first_pure_node = 1
        end
    end

    # --- Collect vertex sets from workspace ---
    pure_upstream = Int[]
    ambiguous     = Int[]
    @inbounds for v in ws.vertices_buf
        nid = ws.node_id_map[v]
        nid == 0 && continue
        if nid >= first_pure_node && nid < first_ambiguous_node
            push!(pure_upstream, v)
        elseif nid >= first_ambiguous_node && nid <= last_accepted
            push!(ambiguous, v)
        end
    end

    isempty(pure_upstream) && return 0
    isempty(ambiguous)     && return 0

    # --- Multi-source BFS from each group ---
    # We need to read distances before cleanup, so we inline the BFS calls
    # and copy distances for ambiguous vertices between passes.
    all_subset = vcat(pure_upstream, ambiguous, cc_chosen, cc_other)
    N_amb = length(ambiguous)

    # BFS from cc_chosen → read distances for ambiguous vertices
    _bfs_run_and_read!(bfs_ws, graph, all_subset, cc_chosen)
    dc = Vector{Int}(undef, N_amb)
    @inbounds for (i, v) in enumerate(ambiguous)
        dc[i] = bfs_ws.distances[v]
    end
    _bfs_cleanup!(bfs_ws, all_subset)

    # BFS from cc_other → read distances for ambiguous vertices
    _bfs_run_and_read!(bfs_ws, graph, all_subset, cc_other)
    du = Vector{Int}(undef, N_amb)
    @inbounds for (i, v) in enumerate(ambiguous)
        du[i] = bfs_ws.distances[v]
    end
    _bfs_cleanup!(bfs_ws, all_subset)

    # BFS from pure_upstream → read distances for ambiguous vertices
    _bfs_run_and_read!(bfs_ws, graph, all_subset, pure_upstream)
    d_up = Vector{Int}(undef, N_amb)
    @inbounds for (i, v) in enumerate(ambiguous)
        d_up[i] = bfs_ws.distances[v]
    end
    _bfs_cleanup!(bfs_ws, all_subset)

    # --- Adaptive threshold: median upstream distance for ambiguous vertices ---
    n_finite = 0
    sum_up = 0
    finite_dists = Int[]
    @inbounds for i in 1:N_amb
        d_up[i] >= 0 || continue
        push!(finite_dists, d_up[i])
    end
    upstream_threshold = isempty(finite_dists) ? typemax(Int) : _median_int(finite_dists)

    # --- Reassign ambiguous vertices ---
    n_reassigned = 0
    @inbounds for i in 1:N_amb
        dci = dc[i]
        dui = du[i]
        reassign = false
        # Unreachable from chosen but reachable from other → reassign
        if dci < 0 && dui >= 0
            reassign = true
        elseif dci < 0 || dui < 0
            # Unreachable from other or both unreachable → keep
        elseif dci <= dui
            # Closer to chosen → keep
        else
            # Closer to other branch — check if in ambiguous ratio zone
            reassign = true
            if dui > 0
                ratio = dci / dui
                if ratio > 0.8 && ratio < 1.2
                    # Tiebreaker: upstream connectivity
                    if d_up[i] >= 0 && d_up[i] <= upstream_threshold
                        reassign = false   # well-connected to upstream → keep
                    end
                end
            end
        end

        if reassign
            v = ambiguous[i]
            old_nid = ws.node_id_map[v]
            ws.node_id_map[v] = 0
            # Update centroid cache: subtract this vertex's contribution
            if points !== nothing && old_nid >= 1 && old_nid <= length(ws.node_count)
                ws.node_sum_x[old_nid] -= Float64(points[v, 1])
                ws.node_sum_y[old_nid] -= Float64(points[v, 2])
                ws.node_sum_z[old_nid] -= Float64(points[v, 3])
                ws.node_count[old_nid] -= 1
            end
            n_reassigned += 1
        end
    end

    return n_reassigned
end

# Run BFS without cleanup (caller reads distances then calls _bfs_cleanup!)
function _bfs_run_and_read!(bfs_ws::_BFSWorkspace,
                            graph::SimpleGraph{Int},
                            subset::AbstractVector{<:Integer},
                            sources::AbstractVector{<:Integer})
    empty!(bfs_ws.queue)
    empty!(bfs_ws.touched)
    @inbounds for v in subset
        bfs_ws.allowed[v] = true
    end
    @inbounds for s in sources
        sv = Int(s)
        if !bfs_ws.visited[sv]
            bfs_ws.visited[sv] = true
            bfs_ws.distances[sv] = 0
            push!(bfs_ws.queue, sv)
            push!(bfs_ws.touched, sv)
        end
    end
    head = 1
    @inbounds while head <= length(bfs_ws.queue)
        u = bfs_ws.queue[head]
        head += 1
        du = bfs_ws.distances[u]
        for nbr in Graphs.neighbors(graph, u)
            bfs_ws.allowed[nbr] || continue
            bfs_ws.visited[nbr] && continue
            bfs_ws.visited[nbr] = true
            bfs_ws.distances[nbr] = du + 1
            push!(bfs_ws.queue, nbr)
            push!(bfs_ws.touched, nbr)
        end
    end
    return nothing
end

# Reset workspace after reading distances
function _bfs_cleanup!(bfs_ws::_BFSWorkspace, subset::AbstractVector{<:Integer})
    @inbounds for v in bfs_ws.touched
        bfs_ws.visited[v] = false
        bfs_ws.distances[v] = -1
        bfs_ws.allowed[v] = false
    end
    @inbounds for v in subset
        bfs_ws.allowed[v] = false
    end
    return nothing
end

# Integer median without Float64 conversion
function _median_int(v::Vector{Int})
    sort!(v)
    n = length(v)
    if isodd(n)
        return v[(n + 1) ÷ 2]
    else
        return (v[n ÷ 2] + v[n ÷ 2 + 1]) ÷ 2
    end
end

function _validate_graph_points(points::AbstractMatrix{<:Real})
    size(points, 2) == 3 || throw(ArgumentError("points must be N×3 matrix"))
    return nothing
end

function _graph_kdtree(points::AbstractMatrix{<:Real})
    return KDTree(Matrix{Float64}(transpose(points)))
end

@inline function _edge_distance(points::AbstractMatrix{<:Real}, i::Int, j::Int)
    dx = float(points[i, 1]) - float(points[j, 1])
    dy = float(points[i, 2]) - float(points[j, 2])
    dz = float(points[i, 3]) - float(points[j, 3])
    return sqrt(dx * dx + dy * dy + dz * dz)
end

function _compact_component_labels_by_size!(parent::Vector{Int}, min_cc_size::Int=1)
    min_cc_size >= 1 || throw(ArgumentError("min_cc_size must be >= 1"))

    labels = zeros(Int, length(parent))
    root_sizes = Dict{Int, Int}()

    @inbounds for i in eachindex(parent)
        root = _uf_find!(parent, i)
        root_sizes[root] = get(root_sizes, root, 0) + 1
    end

    roots = collect(keys(root_sizes))
    sort!(roots; by=r -> (-root_sizes[r], r))

    root_to_label = Dict{Int, Int}()
    next_label = 1
    for root in roots
        if root_sizes[root] >= min_cc_size
            root_to_label[root] = next_label
            next_label += 1
        end
    end

    @inbounds for i in eachindex(parent)
        root = _uf_find!(parent, i)
        labels[i] = get(root_to_label, root, 0)
    end

    return labels
end

function _validate_shortest_path_inputs(points::AbstractMatrix{<:Real}, graph::SimpleGraph{Int},
                                        weights::SparseMatrixCSC{<:Real,<:Integer})
    _validate_graph_points(points)
    n = size(points, 1)
    nv(graph) == n || throw(ArgumentError("graph vertex count must match number of points"))
    size(weights) == (n, n) || throw(ArgumentError("weights must be an N×N sparse matrix"))
    return nothing
end

function _validate_quotient_graph_inputs(points::AbstractMatrix{<:Real}, graph::SimpleGraph{Int},
                                         labels::AbstractVector{<:Integer})
    _validate_graph_points(points)
    n = size(points, 1)
    nv(graph) == n || throw(ArgumentError("graph vertex count must match number of points"))
    length(labels) == n || throw(ArgumentError("labels length must match number of points"))
    return nothing
end

function _validate_vertex_index(graph::SimpleGraph{Int}, idx::Integer, name::AbstractString)
    1 <= idx <= nv(graph) || throw(ArgumentError("$name must be between 1 and $(nv(graph))"))
    return nothing
end

function _validate_subset_workspace(workspace::ConnectedComponentSubsetWorkspace, graph::SimpleGraph{Int})
    n = nv(graph)
    length(workspace.allowed) == n || throw(ArgumentError("workspace size must match nv(graph)"))
    length(workspace.visited) == n || throw(ArgumentError("workspace size must match nv(graph)"))
    length(workspace.labels_global) == n || throw(ArgumentError("workspace size must match nv(graph)"))
    return nothing
end

function _validate_subset_workspace(workspace::ShortestPathSubsetWorkspace, graph::SimpleGraph{Int})
    n = nv(graph)
    length(workspace.allowed) == n || throw(ArgumentError("workspace size must match nv(graph)"))
    length(workspace.visited) == n || throw(ArgumentError("workspace size must match nv(graph)"))
    length(workspace.local_index) == n || throw(ArgumentError("workspace size must match nv(graph)"))
    length(workspace.distances) == n || throw(ArgumentError("workspace size must match nv(graph)"))
    length(workspace.parents) == n || throw(ArgumentError("workspace size must match nv(graph)"))
    return nothing
end

function _validated_subset_vertices(graph::SimpleGraph{Int}, subset::AbstractVector{<:Integer})
    seen = Set{Int}()
    vertices = Int[]
    sizehint!(vertices, length(subset))

    @inbounds for v_raw in subset
        v = Int(v_raw)
        _validate_vertex_index(graph, v, "subset vertex")
        v in seen && continue
        push!(seen, v)
        push!(vertices, v)
    end

    return vertices
end

function _heap_push!(heap_vertices::Vector{Int}, heap_dists::Vector{Float64}, vertex::Int, dist::Float64)
    push!(heap_vertices, vertex)
    push!(heap_dists, dist)

    i = length(heap_vertices)
    while i > 1
        parent = i >>> 1
        heap_dists[parent] <= heap_dists[i] && break
        heap_vertices[parent], heap_vertices[i] = heap_vertices[i], heap_vertices[parent]
        heap_dists[parent], heap_dists[i] = heap_dists[i], heap_dists[parent]
        i = parent
    end

    return nothing
end

function _heap_pop_min!(heap_vertices::Vector{Int}, heap_dists::Vector{Float64})
    n = length(heap_vertices)
    n > 0 || throw(ArgumentError("heap is empty"))

    min_vertex = heap_vertices[1]
    min_dist = heap_dists[1]

    if n == 1
        pop!(heap_vertices)
        pop!(heap_dists)
        return min_vertex, min_dist
    end

    heap_vertices[1] = pop!(heap_vertices)
    heap_dists[1] = pop!(heap_dists)

    i = 1
    while true
        left = i << 1
        right = left + 1
        smallest = i

        if left <= length(heap_vertices) && heap_dists[left] < heap_dists[smallest]
            smallest = left
        end
        if right <= length(heap_vertices) && heap_dists[right] < heap_dists[smallest]
            smallest = right
        end
        smallest == i && break

        heap_vertices[i], heap_vertices[smallest] = heap_vertices[smallest], heap_vertices[i]
        heap_dists[i], heap_dists[smallest] = heap_dists[smallest], heap_dists[i]
        i = smallest
    end

    return min_vertex, min_dist
end

function _nearest_vertex_index(points::AbstractMatrix{<:Real}, target_point::AbstractVector{<:Real})
    size(points, 1) > 0 || throw(ArgumentError("points must contain at least one vertex"))
    length(target_point) == 3 || throw(ArgumentError("target_point must have length 3"))

    tree = _graph_kdtree(points)
    idxs, _ = knn(tree, collect(Float64, target_point), 1, true)
    return idxs[1]
end

function _validate_tree_graph(tree::SimpleGraph{Int})
    n = nv(tree)
    n > 0 || throw(ArgumentError("tree must contain at least one vertex"))
    ne(tree) == n - 1 || throw(ArgumentError("graph must be a tree with nv - 1 edges"))
    return nothing
end

function _root_tree(tree::SimpleGraph{Int}, root::Int)
    n = nv(tree)
    parent = zeros(Int, n)
    parent[root] = root
    children = [Int[] for _ in 1:n]
    order = Int[]
    queue = Int[root]
    head = 1

    while head <= length(queue)
        vertex = queue[head]
        head += 1
        push!(order, vertex)

        for neighbor in Graphs.neighbors(tree, vertex)
            if parent[neighbor] == 0
                parent[neighbor] = vertex
                push!(children[vertex], neighbor)
                push!(queue, neighbor)
            elseif parent[vertex] != neighbor
                throw(ArgumentError("graph must be acyclic"))
            end
        end
    end

    length(order) == n || throw(ArgumentError("graph must be connected"))
    return parent, children, order
end

function _select_initial_child(current::Int, children::Vector{Int}, downstream::Vector{Float64},
                               points::AbstractMatrix{<:Real})
    best_child = children[1]
    best_length = _edge_distance(points, current, best_child) + downstream[best_child]

    for child in @view children[2:end]
        branch_length = _edge_distance(points, current, child) + downstream[child]
        if branch_length > best_length || (branch_length == best_length && child < best_child)
            best_child = child
            best_length = branch_length
        end
    end

    return best_child
end

function _select_linear_child(previous::Int, current::Int, children::Vector{Int},
                              downstream::Vector{Float64}, points::AbstractMatrix{<:Real})
    best_child = children[1]
    best_cosine = _direction_cosine(points, previous, current, best_child)
    best_length = _edge_distance(points, current, best_child) + downstream[best_child]

    for child in @view children[2:end]
        child_cosine = _direction_cosine(points, previous, current, child)
        branch_length = _edge_distance(points, current, child) + downstream[child]

        if child_cosine > best_cosine + 1e-12 ||
           (abs(child_cosine - best_cosine) <= 1e-12 && branch_length > best_length + 1e-12) ||
           (abs(child_cosine - best_cosine) <= 1e-12 && abs(branch_length - best_length) <= 1e-12 && child < best_child)
            best_child = child
            best_cosine = child_cosine
            best_length = branch_length
        end
    end

    return best_child
end

@inline function _edge_direction(points::AbstractMatrix{<:Real}, u::Int, v::Int,
                                 edge_vectors::Union{Nothing, AbstractDict{Tuple{Int, Int}, <:NTuple{3, <:Real}}})
    if !isnothing(edge_vectors)
        vec = get(edge_vectors, (u, v), nothing)
        if !isnothing(vec)
            return float(vec[1]), float(vec[2]), float(vec[3])
        end
    end

    return (
        float(points[v, 1]) - float(points[u, 1]),
        float(points[v, 2]) - float(points[u, 2]),
        float(points[v, 3]) - float(points[u, 3]),
    )
end

@inline function _direction_cosine(points::AbstractMatrix{<:Real}, previous::Int, current::Int, child::Int,
                                   edge_vectors::Union{Nothing, AbstractDict{Tuple{Int, Int}, <:NTuple{3, <:Real}}}=nothing)
    ux, uy, uz = _edge_direction(points, previous, current, edge_vectors)
    vx, vy, vz = _edge_direction(points, current, child, edge_vectors)

    nu = sqrt(ux * ux + uy * uy + uz * uz)
    nv = sqrt(vx * vx + vy * vy + vz * vz)
    (nu > 0 && nv > 0) || return -Inf
    return clamp((ux * vx + uy * vy + uz * vz) / (nu * nv), -1.0, 1.0)
end

# Minimum Z (with edge-distance tie-break) neighbor selection for the first step of a path.
function _select_min_z_neighbor(current::Int, candidates::Vector{Int},
                                points::AbstractMatrix{<:Real})
    best = candidates[1]
    best_z = float(points[best, 3])
    best_dist = _edge_distance(points, current, best)

    for v in @view candidates[2:end]
        vz = float(points[v, 3])
        vd = _edge_distance(points, current, v)
        if vz < best_z - 1e-12 || (abs(vz - best_z) <= 1e-12 && vd < best_dist)
            best = v
            best_z = vz
            best_dist = vd
        end
    end

    return best
end

# Most-linear neighbor selection for subsequent steps, with a 60° cutoff.
# Returns `nothing` if no candidate is within 60° of the incoming direction.
function _select_linear_neighbor(previous::Int, current::Int, candidates::Vector{Int},
                                 points::AbstractMatrix{<:Real},
                                 edge_vectors::Union{Nothing, AbstractDict{Tuple{Int, Int}, <:NTuple{3, <:Real}}}=nothing)
    const_cos60 = 0.5               # cos(60°) = 0.5

    best = nothing
    best_cosine = -Inf

    for v in candidates
        c = _direction_cosine(points, previous, current, v, edge_vectors)
        c < const_cos60 && continue   # reject turns steeper than 60°
        if c > best_cosine + 1e-12 || (abs(c - best_cosine) <= 1e-12 && (isnothing(best) || v < best))
            best = v
            best_cosine = c
        end
    end

    return best
end