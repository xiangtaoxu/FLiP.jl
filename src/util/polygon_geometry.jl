"""
2-D polygon geometry for ground segmentation: convex hull, outward buffering, area.
"""

"""
    convex_hull_2d(points::AbstractMatrix{<:Real}) -> Matrix{Float64}

Compute the 2D convex hull of points projected onto the XY plane.

# Arguments
- `points`: N×2 or N×3 matrix of coordinates (only X and Y are used)

# Returns
- `Matrix{Float64}`: M×2 matrix of hull vertices in counter-clockwise order
  (not closed — last vertex does NOT repeat the first)
"""
function convex_hull_2d(points::AbstractMatrix{<:Real})
    n = size(points, 1)
    n >= 3 || throw(ArgumentError("need at least 3 points for convex hull"))

    pts = [(Float64(points[i, 1]), Float64(points[i, 2])) for i in 1:n]
    ch = DelaunayTriangulation.convex_hull(pts)
    vidx = DelaunayTriangulation.get_vertices(ch)
    p = DelaunayTriangulation.get_points(ch)

    # vidx is closed (first == last), drop the repeated vertex
    m = length(vidx) - 1
    hull = Matrix{Float64}(undef, m, 2)
    @inbounds for i in 1:m
        pt = p[vidx[i]]
        hull[i, 1] = pt[1]
        hull[i, 2] = pt[2]
    end
    return hull
end

# ── Polygon buffer ────────────────────────────────────────────────

# Floor on cos(half-angle) at near-degenerate vertices: prevents the bisector
# offset (buffer / cos_half) from exploding into a long spike. 0.1 corresponds
# to an interior angle ≲ 12°, which convex hulls of ground returns essentially
# never produce; tuned empirically.
const _BUFFER_COS_HALF_FLOOR = 0.1

"""
    buffer_polygon(polygon::AbstractMatrix{<:Real}, buffer::Real) -> Matrix{Float64}

Expand a convex polygon outward by `buffer` meters.

Each vertex is offset along the outward bisector of its two adjacent edges,
scaled so that every edge is displaced by exactly `buffer`.

# Arguments
- `polygon`: M×2 matrix of vertices (ordered, not closed)
- `buffer`: offset distance in meters (must be ≥ 0; 0 = no expansion)

# Returns
- `Matrix{Float64}`: M×2 matrix of buffered polygon vertices
"""
function buffer_polygon(polygon::AbstractMatrix{<:Real}, buffer::Real)
    buffer >= 0 || throw(ArgumentError("buffer must be ≥ 0"))
    m = size(polygon, 1)
    m >= 3 || throw(ArgumentError("polygon must have at least 3 vertices"))

    result = Matrix{Float64}(undef, m, 2)

    @inbounds for i in 1:m
        prev = mod1(i - 1, m)
        next = mod1(i + 1, m)

        # Edge vectors
        e1x = polygon[i, 1] - polygon[prev, 1]
        e1y = polygon[i, 2] - polygon[prev, 2]
        e2x = polygon[next, 1] - polygon[i, 1]
        e2y = polygon[next, 2] - polygon[i, 2]

        # Outward normals (rotate edge 90° clockwise for CCW polygon → outward)
        n1x, n1y =  e1y, -e1x
        n2x, n2y =  e2y, -e2x

        # Normalize
        len1 = sqrt(n1x^2 + n1y^2)
        len2 = sqrt(n2x^2 + n2y^2)
        n1x /= len1; n1y /= len1
        n2x /= len2; n2y /= len2

        # Bisector
        bx = n1x + n2x
        by = n1y + n2y
        blen = sqrt(bx^2 + by^2)
        bx /= blen; by /= blen

        # Scale factor: buffer / cos(half_angle) where cos(half_angle) = dot(n1, bisector)
        cos_half = n1x * bx + n1y * by
        cos_half = max(cos_half, _BUFFER_COS_HALF_FLOOR)
        offset = buffer / cos_half

        result[i, 1] = polygon[i, 1] + bx * offset
        result[i, 2] = polygon[i, 2] + by * offset
    end

    return result
end

# ── Polygon area ──────────────────────────────────────────────────

"""
    polygon_area(polygon::AbstractMatrix{<:Real}) -> Float64

Compute the area of a polygon using the shoelace formula.

# Arguments
- `polygon`: M×2 matrix of vertices (ordered, not closed)

# Returns
- `Float64`: Absolute area of the polygon
"""
function polygon_area(polygon::AbstractMatrix{<:Real})
    m = size(polygon, 1)
    m >= 3 || throw(ArgumentError("polygon must have at least 3 vertices"))
    area = 0.0
    @inbounds for i in 1:m
        j = mod1(i + 1, m)
        area += polygon[i, 1] * polygon[j, 2]
        area -= polygon[j, 1] * polygon[i, 2]
    end
    return abs(area) / 2.0
end

# ── 3D PCA ────────────────────────────────────────────────────────

