"""
Geometry helpers.

Functions:
- `convex_hull_2d(points)`        — convex hull of points projected onto XY
- `buffer_polygon(poly, d)`       — expand a convex polygon outward by `d`
- `polygon_area(poly)`            — shoelace-formula polygon area
- `pca_linearity(coords, idx, t)` — 3D PCA + linearity check, returns PC1 direction
"""

using LinearAlgebra: Symmetric, eigen

# ── 2D convex hull ────────────────────────────────────────────────

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

"""
    pca_linearity(coords, indices, linearity_threshold) -> NamedTuple or nothing

Compute 3D PCA on the points `coords[indices, :]` (an N×3 matrix). Returns a
`NamedTuple{(:center, :eigenvalues, :direction, :linearity)}` where:

- `center::NTuple{3,Float64}` — centroid (mean of x, y, z over `indices`)
- `eigenvalues::NTuple{3,Float64}` — ascending order (λ₁ ≤ λ₂ ≤ λ₃)
- `direction::NTuple{3,Float64}` — unit PC1 vector (eigenvector of λ₃)
- `linearity::Float64` — (λ₃ − λ₂) / λ₃, in [0, 1]

Returns `nothing` if `length(indices) < 3`, if λ₃ ≤ 0 (degenerate), or if
`linearity < linearity_threshold`. The direction is unoriented — callers that
care about a specific orientation (e.g. low-z to high-z for tree stems) must
apply their own convention.
"""
function pca_linearity(coords::AbstractMatrix{<:Real},
                       indices::AbstractVector{<:Integer},
                       linearity_threshold::Real)
    n = length(indices)
    n < 3 && return nothing

    # Centroid
    mx = 0.0; my = 0.0; mz = 0.0
    @inbounds for i in indices
        mx += coords[i, 1]; my += coords[i, 2]; mz += coords[i, 3]
    end
    inv_n = 1.0 / n
    mx *= inv_n; my *= inv_n; mz *= inv_n

    # 3×3 covariance
    c11 = 0.0; c12 = 0.0; c13 = 0.0
    c22 = 0.0; c23 = 0.0; c33 = 0.0
    @inbounds for i in indices
        dx = coords[i, 1] - mx
        dy = coords[i, 2] - my
        dz = coords[i, 3] - mz
        c11 += dx * dx; c12 += dx * dy; c13 += dx * dz
        c22 += dy * dy; c23 += dy * dz; c33 += dz * dz
    end

    C = Symmetric([c11 c12 c13; c12 c22 c23; c13 c23 c33])
    F = eigen(C)
    # eigenvalues ascending: F.values[1] ≤ [2] ≤ [3]
    λ3 = F.values[3]
    λ2 = F.values[2]
    λ3 > 0 || return nothing

    linearity = (λ3 - λ2) / λ3
    linearity < linearity_threshold && return nothing

    dvec = (F.vectors[1, 3], F.vectors[2, 3], F.vectors[3, 3])
    return (center=(mx, my, mz),
            eigenvalues=(F.values[1], F.values[2], F.values[3]),
            direction=dvec,
            linearity=linearity)
end

# ── Perpendicular basis ─────────────────────────────────────────────────────

"""
    _build_perpendicular_basis(d) -> (e1, e2)

Build an orthonormal basis `(e1, e2)` perpendicular to a unit direction `d`.
Chooses the coordinate axis least aligned with `d` as a reference, then forms
`e1 = normalize(d × ref)` and `e2 = d × e1`. Both returns are
`NTuple{3,Float64}`. Numerically stable for any non-degenerate `d`.
"""
function _build_perpendicular_basis(d::NTuple{3,Float64})
    dx, dy, dz = d
    # Choose axis least aligned with d
    ax = abs(dx); ay = abs(dy); az = abs(dz)
    if ax <= ay && ax <= az
        ref = (1.0, 0.0, 0.0)
    elseif ay <= az
        ref = (0.0, 1.0, 0.0)
    else
        ref = (0.0, 0.0, 1.0)
    end
    # e1 = normalize(d × ref)
    e1x = dy * ref[3] - dz * ref[2]
    e1y = dz * ref[1] - dx * ref[3]
    e1z = dx * ref[2] - dy * ref[1]
    e1_norm = sqrt(e1x^2 + e1y^2 + e1z^2)
    e1 = (e1x / e1_norm, e1y / e1_norm, e1z / e1_norm)
    # e2 = d × e1
    e2x = dy * e1[3] - dz * e1[2]
    e2y = dz * e1[1] - dx * e1[3]
    e2z = dx * e1[2] - dy * e1[1]
    return (e1, (e2x, e2y, e2z))
end
