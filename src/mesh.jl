"""
Mesh operations built from point clouds.
"""

"""
    XYTriMesh

Container for an XY-plane triangulated surface with source vertex z values.
"""
struct XYTriMesh{T}
    triangulation::T
    points_xy::Vector{NTuple{2, Float64}}
    z::Vector{Float64}
    triangles::Vector{NTuple{3, Int}}
    triangle_graph::SimpleGraph{Int}
    triangle_key_to_index::Dict{NTuple{3, Int}, Int}
end

@inline function _sort2(a::Int, b::Int)
    a <= b ? (a, b) : (b, a)
end

@inline function _sort3(a::Int, b::Int, c::Int)
    if a > b
        a, b = b, a
    end
    if b > c
        b, c = c, b
    end
    if a > b
        a, b = b, a
    end
    return (a, b, c)
end

@inline _canonical_triangle(t::NTuple{3, Int}) = _sort3(t[1], t[2], t[3])

"""
    delaunay_triangulation_xy(points::AbstractMatrix{<:Real}) -> XYTriMesh

Build an XY-plane Delaunay triangulation from an input XYZ cloud.
The triangulation is constructed using only XY coordinates, while z values are
retained at vertices for later interpolation and z-distance computation.

# Arguments
- `points`: N×3 matrix of XYZ coordinates

# Returns
- `XYTriMesh`: triangulation, XY vertices, z values, triangle list, adjacency graph,
    and triangle index map
"""
function delaunay_triangulation_xy(points::AbstractMatrix{<:Real})
    size(points, 2) == 3 || throw(ArgumentError("points must be N×3 matrix"))

    n = size(points, 1)
    n >= 3 || throw(ArgumentError("need at least 3 points for triangulation"))

    points_xy = Vector{NTuple{2, Float64}}(undef, n)
    z = Vector{Float64}(undef, n)

    @inbounds for i in 1:n
        xi = float(points[i, 1])
        yi = float(points[i, 2])
        zi = float(points[i, 3])

        (isfinite(xi) && isfinite(yi) && isfinite(zi)) ||
            throw(ArgumentError("points contain non-finite values"))

        points_xy[i] = (xi, yi)
        z[i] = zi
    end

    tri = triangulate(points_xy)

    triangles = NTuple{3, Int}[]
    for t in get_triangles(tri)
        all(v -> v > 0, t) || continue
        push!(triangles, (Int(t[1]), Int(t[2]), Int(t[3])))
    end

    isempty(triangles) && throw(ArgumentError("triangulation contains no solid triangles"))

    triangle_key_to_index = Dict{NTuple{3, Int}, Int}()
    sizehint!(triangle_key_to_index, length(triangles))
    for (i, t) in enumerate(triangles)
        triangle_key_to_index[_canonical_triangle(t)] = i
    end

    triangle_graph = SimpleGraph(length(triangles))
    edge_owner = Dict{NTuple{2, Int}, Int}()

    for (i, t) in enumerate(triangles)
        e1 = _sort2(t[1], t[2])
        e2 = _sort2(t[2], t[3])
        e3 = _sort2(t[3], t[1])

        for e in (e1, e2, e3)
            j = get(edge_owner, e, 0)
            if j == 0
                edge_owner[e] = i
            else
                add_edge!(triangle_graph, i, j)
            end
        end
    end

    return XYTriMesh(tri, points_xy, z, triangles, triangle_graph, triangle_key_to_index)
end

@inline function _point_in_triangle_xy(px::Float64, py::Float64,
                                       a::NTuple{2, Float64},
                                       b::NTuple{2, Float64},
                                       c::NTuple{2, Float64};
                                       tol::Float64=1e-12)
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c

    den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    abs(den) > eps(Float64) || return false

    w1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / den
    w2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / den
    w3 = 1.0 - w1 - w2

    return (w1 >= -tol) && (w2 >= -tol) && (w3 >= -tol)
end

@inline function _contains_point(tri_idx::Int,
                                 px::Float64,
                                 py::Float64,
                                 triangles::Vector{NTuple{3, Int}},
                                 points_xy::Vector{NTuple{2, Float64}})
    t = triangles[tri_idx]
    return _point_in_triangle_xy(px, py, points_xy[t[1]], points_xy[t[2]], points_xy[t[3]])
end

@inline function _interpolate_z_in_triangle(px::Float64, py::Float64,
                                            a::NTuple{2, Float64},
                                            b::NTuple{2, Float64},
                                            c::NTuple{2, Float64},
                                            za::Float64, zb::Float64, zc::Float64)
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c

    den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    abs(den) > eps(Float64) || return NaN

    w1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / den
    w2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / den
    w3 = 1.0 - w1 - w2

    return w1 * za + w2 * zb + w3 * zc
end

"""
    cloud_to_mesh_distance_z(points::AbstractMatrix{<:Real}, mesh::XYTriMesh) -> Vector{Float64}

Compute signed z-direction distance from each cloud point to a reference XY-triangulated mesh.
For each query point `(x, y, zq)`, locate its containing triangle in the mesh XY projection,
interpolate mesh z at `(x, y)`, and return:

- `dz = zq - z_mesh(x, y)`

Points outside the mesh XY convex hull return `NaN`.

# Arguments
- `points`: N×3 matrix of query XYZ coordinates
- `mesh`: Reference mesh created by `delaunay_triangulation_xy`

# Returns
- `Vector{Float64}`: Signed z residuals (`NaN` outside hull)
"""
function cloud_to_mesh_distance_z(points::AbstractMatrix{<:Real}, mesh::XYTriMesh)
    size(points, 2) == 3 || throw(ArgumentError("points must be N×3 matrix"))
    length(mesh.points_xy) == length(mesh.z) || throw(ArgumentError("mesh points_xy and z must have the same length"))
    nv(mesh.triangle_graph) == length(mesh.triangles) || throw(ArgumentError("triangle_graph size must match number of triangles"))

    n = size(points, 1)
    out = Vector{Float64}(undef, n)
    last_tri_idx = 0

    @inbounds for i in 1:n
        xq = float(points[i, 1])
        yq = float(points[i, 2])
        zq = float(points[i, 3])

        if !(isfinite(xq) && isfinite(yq) && isfinite(zq))
            out[i] = NaN
            continue
        end

        tri_idx = 0
        if 1 <= last_tri_idx <= length(mesh.triangles)
            if _contains_point(last_tri_idx, xq, yq, mesh.triangles, mesh.points_xy)
                tri_idx = last_tri_idx
            else
                for nb in Graphs.neighbors(mesh.triangle_graph, last_tri_idx)
                    if _contains_point(nb, xq, yq, mesh.triangles, mesh.points_xy)
                        tri_idx = nb
                        break
                    end
                end
            end
        end

        if tri_idx == 0
            tri = find_triangle(mesh.triangulation, (xq, yq))
            if any(v -> v <= 0, tri)
                out[i] = NaN
                continue
            end
            tri_key = _canonical_triangle((Int(tri[1]), Int(tri[2]), Int(tri[3])))
            tri_idx = get(mesh.triangle_key_to_index, tri_key, 0)
            tri_idx == 0 && (out[i] = NaN; continue)
        end

        t = mesh.triangles[tri_idx]
        v1, v2, v3 = t

        zmesh = _interpolate_z_in_triangle(
            xq,
            yq,
            mesh.points_xy[v1],
            mesh.points_xy[v2],
            mesh.points_xy[v3],
            float(mesh.z[v1]),
            float(mesh.z[v2]),
            float(mesh.z[v3]),
        )

        out[i] = isfinite(zmesh) ? (zq - zmesh) : NaN
        last_tri_idx = tri_idx
    end

    return out
end

"""
    sample_mesh_xy(mesh::XYTriMesh, xy_resolution::Real) -> Matrix{Float64}

Sample an XY-triangulated mesh on a regular XY lattice and return sampled ground
points as an N×3 matrix `(x, y, z_mesh)`.

The lattice spans the mesh XY bounding box with spacing `xy_resolution`.
Candidate samples outside the mesh hull are discarded.

# Arguments
- `mesh`: Reference mesh created by `delaunay_triangulation_xy`
- `xy_resolution`: XY sample spacing (must be > 0)

# Returns
- `Matrix{Float64}`: N×3 sampled ground points `(x, y, z_mesh)`
"""
function sample_mesh_xy(mesh::XYTriMesh, xy_resolution::Real)
    xy_resolution > 0 || throw(ArgumentError("xy_resolution must be > 0"))

    n_vertices = length(mesh.points_xy)
    n_vertices > 0 || throw(ArgumentError("mesh has no vertices"))

    xmin = Inf
    xmax = -Inf
    ymin = Inf
    ymax = -Inf

    @inbounds for (x, y) in mesh.points_xy
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
    end

    step = float(xy_resolution)
    nx = max(1, floor(Int, (xmax - xmin) / step) + 1)
    ny = max(1, floor(Int, (ymax - ymin) / step) + 1)
    n_candidates = nx * ny

    query = Matrix{Float64}(undef, n_candidates, 3)
    k = 1
    @inbounds for iy in 0:(ny - 1)
        y = ymin + iy * step
        for ix in 0:(nx - 1)
            x = xmin + ix * step
            query[k, 1] = x
            query[k, 2] = y
            query[k, 3] = 0.0
            k += 1
        end
    end

    dz = cloud_to_mesh_distance_z(query, mesh)
    n_keep = count(isfinite, dz)
    n_keep > 0 || throw(ArgumentError("no valid mesh samples at the requested xy_resolution"))

    sampled = Matrix{Float64}(undef, n_keep, 3)
    out_idx = 1
    @inbounds for i in 1:n_candidates
        dzi = dz[i]
        if isfinite(dzi)
            sampled[out_idx, 1] = query[i, 1]
            sampled[out_idx, 2] = query[i, 2]
            sampled[out_idx, 3] = -dzi
            out_idx += 1
        end
    end

    return sampled
end
