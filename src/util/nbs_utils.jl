"""
Pure NBS-modeling kernels (array-in / array-out, unit-testable): Taubin circle fit, point
unrolling to (rho,phi), per-slice rho-outlier filtering, the 2D periodic rho-surface build /
gap-fill / smoothing, and frustum geometry.
"""

# Generalized-eigenvalue positivity tolerance in taubin_circle_fit
const TAUBIN_EIG_TOL = 1e-12

"""
    taubin_circle_fit(u, v) -> (cx, cy, r)

Fit a circle to 2D points using the Taubin algebraic method (1991).
Robust to partial arcs. Returns center (cx, cy) and radius r.
Falls back to centroid + mean distance if SVD fails.
"""
function taubin_circle_fit(u::AbstractVector{<:Real}, v::AbstractVector{<:Real})
    n = length(u)
    @assert n == length(v)

    um = mean(u); vm = mean(v)
    uc = u .- um; vc = v .- vm

    # Build constraint matrix for Taubin method
    # Minimize algebraic distance subject to gradient-weighted normalization
    # Z = [u² + v², u, v, 1]  →  ZᵀZ eigenproblem with constraint matrix M
    z1 = uc .^ 2 .+ vc .^ 2
    Z = hcat(z1, uc, vc, ones(n))
    M = Z' * Z ./ n

    # Constraint matrix N (Taubin normalization)
    mean_z1 = mean(z1)
    N = zeros(4, 4)
    N[1, 1] = 8.0 * mean_z1
    N[1, 2] = N[2, 1] = 4.0 * mean(uc)   # should be ~0 since centered
    N[1, 3] = N[3, 1] = 4.0 * mean(vc)
    N[2, 2] = 1.0
    N[3, 3] = 1.0

    # Solve generalized eigenvalue problem M*a = η*N*a
    # Find smallest positive generalized eigenvalue
    try
        F = eigen(Symmetric(M), Symmetric(N))
        # Find smallest positive eigenvalue
        best_idx = 0
        best_val = Inf
        for j in 1:4
            λ = F.values[j]
            if λ > TAUBIN_EIG_TOL && λ < best_val
                best_val = λ
                best_idx = j
            end
        end
        if best_idx > 0
            a = F.vectors[:, best_idx]
            cx_c = -a[2] / (2.0 * a[1])
            cy_c = -a[3] / (2.0 * a[1])
            r = sqrt(cx_c^2 + cy_c^2 + (a[2]^2 + a[3]^2 - 4.0 * a[1] * a[4]) / (4.0 * a[1]^2))
            return (um + cx_c, vm + cy_c, abs(r))
        end
    catch
        # Fall through to simple method
    end

    # Fallback: simple algebraic fit (Kasa)
    A = hcat(2.0 .* uc, 2.0 .* vc, ones(n))
    b = uc .^ 2 .+ vc .^ 2
    try
        x = A \ b
        cx_c = x[1]; cy_c = x[2]
        r = sqrt(x[3] + cx_c^2 + cy_c^2)
        return (um + cx_c, vm + cy_c, abs(r))
    catch
        # Ultimate fallback: centroid + mean distance
        dists = sqrt.(uc .^ 2 .+ vc .^ 2)
        return (um, vm, mean(dists))
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# 2b. Unroll points & per-slice rho statistics
# ───────────────────────────────────────────────────────────────────────────────

"""
    _unroll_points(coords, indices, centers, point_slice_ids, e1, e2)
        -> (rho, phi)

Convert points to cylindrical coordinates relative to the smoothed centerline.
"""
function _unroll_points(coords::AbstractMatrix{<:Real}, indices::Vector{Int},
                        centers::Matrix{Float64}, point_slice_ids::Vector{Int},
                        e1::NTuple{3,Float64}, e2::NTuple{3,Float64})
    n = length(indices)
    rho = Vector{Float64}(undef, n)
    phi = Vector{Float64}(undef, n)

    @inbounds for j in 1:n
        i = indices[j]
        s = point_slice_ids[j]
        dx = coords[i, 1] - centers[s, 1]
        dy = coords[i, 2] - centers[s, 2]
        dz = coords[i, 3] - centers[s, 3]
        u_val = dx * e1[1] + dy * e1[2] + dz * e1[3]
        v_val = dx * e2[1] + dy * e2[2] + dz * e2[3]
        rho[j] = sqrt(u_val^2 + v_val^2)
        phi[j] = atan(v_val, u_val)
    end

    return (rho, phi)
end

"""
    _filter_rho_outliers(rho, phi, pt_slice_ids, slice_point_indices, indices,
                          n_slices, rho_percentile)
        -> (rho, phi, pt_slice_ids, slice_point_indices, indices)

For each slice, drop points whose `rho` exceeds the per-slice
`quantile(rho_slice, rho_percentile)`. The kept points are returned as a
compacted survivor set so all downstream consumers (`_method_spline_2d`,
`_generate_surface_points`, slice→node mapping) operate on contiguous arrays
without sentinel handling.

When `rho_percentile >= 1.0` returns the inputs unchanged (no-op fast path).
"""
function _filter_rho_outliers(rho::Vector{Float64}, phi::Vector{Float64},
                               pt_slice_ids::Vector{Int},
                               slice_point_indices::Vector{Vector{Int}},
                               indices::Vector{Int},
                               n_slices::Int,
                               rho_percentile::Float64)
    rho_percentile >= 1.0 && return (rho, phi, pt_slice_ids, slice_point_indices, indices)

    # Per-slice rho cutoffs (Inf for slices with no points → keeps no-op for them)
    thresh = fill(Inf, n_slices)
    @inbounds for s in 1:n_slices
        ids = slice_point_indices[s]
        isempty(ids) && continue
        thresh[s] = quantile(@view(rho[ids]), rho_percentile)
    end

    n = length(rho)
    keep_mask = falses(n)
    @inbounds for j in 1:n
        keep_mask[j] = rho[j] <= thresh[pt_slice_ids[j]]
    end

    n_kept = count(keep_mask)
    rho_kept           = Vector{Float64}(undef, n_kept)
    phi_kept           = Vector{Float64}(undef, n_kept)
    pt_slice_ids_kept  = Vector{Int}(undef, n_kept)
    indices_kept       = Vector{Int}(undef, n_kept)
    old_to_new         = zeros(Int, n)
    pos = 0
    @inbounds for j in 1:n
        if keep_mask[j]
            pos += 1
            rho_kept[pos]          = rho[j]
            phi_kept[pos]          = phi[j]
            pt_slice_ids_kept[pos] = pt_slice_ids[j]
            indices_kept[pos]      = indices[j]
            old_to_new[j] = pos
        end
    end

    slice_point_indices_kept = [Int[] for _ in 1:n_slices]
    @inbounds for s in 1:n_slices
        for j in slice_point_indices[s]
            nj = old_to_new[j]
            nj > 0 && push!(slice_point_indices_kept[s], nj)
        end
    end

    return (rho_kept, phi_kept, pt_slice_ids_kept, slice_point_indices_kept, indices_kept)
end

# ───────────────────────────────────────────────────────────────────────────────
# 2c. 2D periodic surface smoothing & integration
# ───────────────────────────────────────────────────────────────────────────────

"""
    _method_spline_2d(rho, phi, pt_slice_ids, n_slices, cfg)
        -> Vector{NamedTuple{(:cross_area, :circumference, :completeness)}}

Compute cross-sectional area and circumference for all slices of an NBS
using 2D surface smoothing (periodic in phi, non-periodic in z).
Returns a vector indexed by slice (1:n_slices); slices with no data
have zeros. The angular bin count adapts to NBS size via the median of `rho`.
"""
function _method_spline_2d(rho::Vector{Float64}, phi::Vector{Float64},
                           pt_slice_ids::Vector{Int}, n_slices::Int,
                           cfg::FLiPConfig)
    qsm = cfg.tree.model                                          # local alias
    T = NamedTuple{(:cross_area, :circumference, :completeness), Tuple{Float64, Float64, Float64}}
    results = Vector{T}(undef, n_slices)
    fill!(results, (cross_area=0.0, circumference=0.0, completeness=0.0))

    rho_median_global = isempty(rho) ? 0.01 : median(rho)
    surface_res = qsm.surface_res_scalar * cfg.pipeline.subsample_res
    phi_bin_num = clamp(ceil(Int, 2π * rho_median_global / surface_res),
                        qsm.phi_bin_min, qsm.phi_bin_max)
    dphi = 2π / phi_bin_num

    # Build 2D surface (rho already pre-filtered upstream via qsm.rho_percentile)
    surface = _build_rho_surface(rho, phi, pt_slice_ids, n_slices, phi_bin_num)

    # Compute completeness per slice before gap-filling
    completeness = Vector{Float64}(undef, n_slices)
    @inbounds for s in 1:n_slices
        n_filled = count(b -> isfinite(surface[b, s]), 1:phi_bin_num)
        completeness[s] = n_filled / phi_bin_num
    end

    # Fill gaps and smooth
    _fill_gaps_2d!(surface)
    _smooth_surface_2d!(surface, 0.5, qsm.spl_z_smoothing)

    # Extract per-slice metrics
    @inbounds for s in 1:n_slices
        completeness[s] <= 0 && continue

        # Central differences for derivative (periodic)
        circ = 0.0
        area = 0.0
        for b in 1:phi_bin_num
            r = surface[b, s]
            isfinite(r) || continue
            bm = mod1(b - 1, phi_bin_num)
            bp = mod1(b + 1, phi_bin_num)
            dr = (surface[bp, s] - surface[bm, s]) / (2.0 * dphi)
            circ += sqrt(r^2 + dr^2) * dphi
            area += 0.5 * r^2 * dphi
        end
        results[s] = (cross_area=area, circumference=circ, completeness=completeness[s])
    end

    return (results, surface, phi_bin_num)
end

"""
    _build_rho_surface(rho, phi, pt_slice_ids, n_slices, phi_bin_num)
        -> Matrix{Float64}  # (phi_bin_num, n_slices)

Bin all NBS points into a 2D grid of rho values; each cell is the arithmetic
mean of all rho values falling into it, empty cells are NaN. Rho-outlier
filtering happens upstream in `_filter_rho_outliers` (driven by
`cfg.tree.model.rho_percentile`), so this routine has no percentile knob of its own.
"""
function _build_rho_surface(rho::Vector{Float64}, phi::Vector{Float64},
                            pt_slice_ids::Vector{Int}, n_slices::Int,
                            phi_bin_num::Int)
    dphi = 2π / phi_bin_num
    bin_sum = zeros(phi_bin_num, n_slices)
    bin_count = zeros(Int, phi_bin_num, n_slices)

    @inbounds for j in eachindex(rho)
        b = clamp(floor(Int, (phi[j] + π) / dphi) + 1, 1, phi_bin_num)
        s = pt_slice_ids[j]
        bin_sum[b, s] += rho[j]
        bin_count[b, s] += 1
    end

    surface = fill(NaN, phi_bin_num, n_slices)
    @inbounds for s in 1:n_slices, b in 1:phi_bin_num
        if bin_count[b, s] > 0
            surface[b, s] = bin_sum[b, s] / bin_count[b, s]
        end
    end
    return surface
end

"""
    _fill_gaps_2d!(surface)

Fill NaN cells in the 2D surface via nearest-neighbor interpolation.
Phi direction (axis 1) wraps periodically; z direction (axis 2) clamps.
"""
function _fill_gaps_2d!(surface::Matrix{Float64})
    nphi, nz = size(surface)
    max_search = max(nphi ÷ 2, nz)

    @inbounds for s in 1:nz, b in 1:nphi
        isnan(surface[b, s]) || continue

        # Search 4 cardinal directions for nearest non-NaN neighbor
        wsum = 0.0
        rsum = 0.0

        # phi+ direction (periodic)
        for offset in 1:max_search
            bp = mod1(b + offset, nphi)
            if isfinite(surface[bp, s])
                w = 1.0 / offset
                wsum += w
                rsum += w * surface[bp, s]
                break
            end
        end
        # phi- direction (periodic)
        for offset in 1:max_search
            bm = mod1(b - offset, nphi)
            if isfinite(surface[bm, s])
                w = 1.0 / offset
                wsum += w
                rsum += w * surface[bm, s]
                break
            end
        end
        # z+ direction (clamped)
        for offset in 1:(nz - s)
            if isfinite(surface[b, s + offset])
                w = 1.0 / offset
                wsum += w
                rsum += w * surface[b, s + offset]
                break
            end
        end
        # z- direction (clamped)
        for offset in 1:(s - 1)
            if isfinite(surface[b, s - offset])
                w = 1.0 / offset
                wsum += w
                rsum += w * surface[b, s - offset]
                break
            end
        end

        if wsum > 0
            surface[b, s] = rsum / wsum
        end
    end
end

"""
    _smooth_surface_2d!(surface, s_phi, s_z, n_passes=1)

Separable 3-point stencil smoothing of the rho surface.
Periodic in phi (axis 1), Neumann boundary in z (axis 2).
"""
function _smooth_surface_2d!(surface::Matrix{Float64},
                              s_phi::Float64, s_z::Float64,
                              n_passes::Int=1)
    nphi, nz = size(surface)
    buf = similar(surface)

    for _ in 1:n_passes
        # Phi pass (periodic)
        w_center_phi = 1.0 - s_phi
        w_side_phi = s_phi / 2.0
        @inbounds for s in 1:nz, b in 1:nphi
            bm = mod1(b - 1, nphi)
            bp = mod1(b + 1, nphi)
            buf[b, s] = w_center_phi * surface[b, s] +
                         w_side_phi * (surface[bm, s] + surface[bp, s])
        end

        # Z pass (Neumann boundary)
        if nz == 1
            copyto!(surface, buf)
        else
            w_center_z = 1.0 - s_z
            w_side_z = s_z / 2.0
            @inbounds for s in 1:nz, b in 1:nphi
                if s == 1
                    surface[b, s] = w_center_z * buf[b, s] + s_z * buf[b, 2]
                elseif s == nz
                    surface[b, s] = w_center_z * buf[b, s] + s_z * buf[b, nz - 1]
                else
                    surface[b, s] = w_center_z * buf[b, s] +
                                     w_side_z * (buf[b, s - 1] + buf[b, s + 1])
                end
            end
        end
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# 2d. Surface point cloud generation
# ───────────────────────────────────────────────────────────────────────────────

"""Compute frustum volume and surface area between consecutive nodes."""
function _frustum_metrics(r1::Float64, r2::Float64, h::Float64)
    vol = (π / 3.0) * h * (r1^2 + r2^2 + r1 * r2)
    sa = π * (r1 + r2) * sqrt(h^2 + (r1 - r2)^2)
    return (vol, sa)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Output aggregation & I/O
# ═══════════════════════════════════════════════════════════════════════════════

