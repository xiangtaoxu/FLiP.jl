"""
PCA helpers: principal-axis linearity (NBS labeling + node modeling) and a perpendicular
orthonormal basis from a unit direction.
"""

"""
    _pca3_principal(c11, c12, c13, c22, c23, c33)
        -> (eigenvalues::NTuple{3,Float64}, pc1::NTuple{3,Float64})

Eigen-decompose the symmetric 3×3 covariance whose upper triangle is
`[c11 c12 c13; · c22 c23; · · c33]`. Returns eigenvalues ascending
(λ₁ ≤ λ₂ ≤ λ₃) and the unit PC1 (eigenvector of the largest eigenvalue λ₃).
Shared by every 3×3 PCA in the package so the eigen convention lives in one place.
"""
@inline function _pca3_principal(c11::Float64, c12::Float64, c13::Float64,
                                 c22::Float64, c23::Float64, c33::Float64)
    C = Symmetric([c11 c12 c13; c12 c22 c23; c13 c23 c33])
    F = eigen(C)
    # eigenvalues ascending: F.values[1] ≤ [2] ≤ [3]; PC1 = eigenvector of λ₃ (last column)
    return ((F.values[1], F.values[2], F.values[3]),
            (F.vectors[1, 3], F.vectors[2, 3], F.vectors[3, 3]))
end

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

    eigvals, dvec = _pca3_principal(c11, c12, c13, c22, c23, c33)
    λ3 = eigvals[3]
    λ2 = eigvals[2]
    λ3 > 0 || return nothing

    linearity = (λ3 - λ2) / λ3
    linearity < linearity_threshold && return nothing

    return (center=(mx, my, mz),
            eigenvalues=eigvals,
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

# ── Finite-cylinder primitives (used by post-QSM NBS merging) ────────────────
#
# A finite cylinder is given by its midpoint `center`, unit `axis`, `radius`, and
# `half_height` (= height / 2). These primitives are deterministic (no RNG): the
# volume estimator samples a FIXED global voxel lattice so self-volumes and pair
# intersections are computed on the same cells, which keeps overlap ratios ≤ 1
# and fully reproducible — matching FLiP's no-random-seed guarantee.

