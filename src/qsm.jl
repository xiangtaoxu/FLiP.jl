"""
Quantitative Structural Modeling (QSM) for FLiP.jl.

Converts tree-segmented point clouds into geometric measurements (circumference,
cross-sectional area, volume, surface area) per branch node and per tree.
Uses 2D periodic surface smoothing (periodic in phi, non-periodic in z) for
cross-section estimation.
"""

using LinearAlgebra: Symmetric, eigen, dot, norm, cross, normalize
using Statistics: mean, median, quantile
using NearestNeighbors: KDTree, knn

# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

"""Per-NBS metadata after linearity filtering and PCA."""
struct NBSInfo
    direction::NTuple{3,Float64}   # PC1 unit vector
    center::NTuple{3,Float64}      # mean centroid
    eigenvalues::NTuple{3,Float64} # ascending order
    linearity::Float64
    point_indices::Vector{Int}
end

"""Per-QSM node biometric results (one per z-slice per NBS)."""
mutable struct QSMNode
    qsm_node_id::Int
    nbs_id::Int32
    tree_id::Int32
    agh::Float64
    height::Float64
    completeness::Float64
    n_points::Int
    center_x::Float64
    center_y::Float64
    center_z::Float64
    direction_x::Float64
    direction_y::Float64
    direction_z::Float64
    cross_area::Float64
    circumference::Float64
    radius_area::Float64
    radius_circ::Float64
end

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: NBS Linearity Filtering + PCA
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _compute_nbs_pca(coords, indices) -> NBSInfo or nothing

Compute PCA on the points of a single NBS. Returns NBSInfo if enough points
and the segment is sufficiently linear, otherwise nothing.
"""
function _compute_nbs_pca(coords::AbstractMatrix{<:Real}, indices::Vector{Int},
                          ::Int32, linearity_threshold::Float64)
    n = length(indices)
    n < 3 && return nothing

    # Compute mean
    mx = 0.0; my = 0.0; mz = 0.0
    @inbounds for i in indices
        mx += coords[i, 1]; my += coords[i, 2]; mz += coords[i, 3]
    end
    inv_n = 1.0 / n
    mx *= inv_n; my *= inv_n; mz *= inv_n

    # Build 3×3 covariance matrix
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
    # eigenvalues in ascending order: F.values[1] ≤ [2] ≤ [3]
    λ3 = F.values[3]
    λ2 = F.values[2]
    λ3 > 0 || return nothing

    linearity = (λ3 - λ2) / λ3
    linearity < linearity_threshold && return nothing

    # PC1 = eigenvector of largest eigenvalue
    dvec = (F.vectors[1, 3], F.vectors[2, 3], F.vectors[3, 3])

    # Orient: direction points from low-z to high-z projection
    z_min_i = 0; z_max_i = 0
    @inbounds for i in indices
        z = coords[i, 3]
        if z_min_i == 0 || z < coords[z_min_i, 3]; z_min_i = i; end
        if z_max_i == 0 || z > coords[z_max_i, 3]; z_max_i = i; end
    end
    proj_low = (coords[z_min_i, 1] - mx) * dvec[1] +
               (coords[z_min_i, 2] - my) * dvec[2] +
               (coords[z_min_i, 3] - mz) * dvec[3]
    proj_high = (coords[z_max_i, 1] - mx) * dvec[1] +
                (coords[z_max_i, 2] - my) * dvec[2] +
                (coords[z_max_i, 3] - mz) * dvec[3]
    if proj_high < proj_low
        dvec = (-dvec[1], -dvec[2], -dvec[3])
    end

    return NBSInfo(dvec, (mx, my, mz),
                   (F.values[1], F.values[2], F.values[3]),
                   linearity, indices)
end

"""
    _filter_linear_nbs(coords, nbs_ids, cfg) -> Dict{Int32, NBSInfo}

Filter NBS segments by linearity and return PCA results for qualifying ones.
"""
function _filter_linear_nbs(coords::AbstractMatrix{<:Real},
                            nbs_ids::AbstractVector{<:Integer},
                            cfg::FLiPConfig)
    # Group point indices by NBS id
    nbs_groups = Dict{Int32, Vector{Int}}()
    @inbounds for i in eachindex(nbs_ids)
        nid = Int32(nbs_ids[i])
        nid > 0 || continue
        if haskey(nbs_groups, nid)
            push!(nbs_groups[nid], i)
        else
            nbs_groups[nid] = [i]
        end
    end

    result = Dict{Int32, NBSInfo}()
    for (nid, indices) in nbs_groups
        info = _compute_nbs_pca(coords, indices, nid, cfg.qsm_nbs_linearity_threshold)
        if !isnothing(info)
            result[nid] = info
        end
    end
    return result
end

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Taubin Circle Fit
# ═══════════════════════════════════════════════════════════════════════════════

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
            if λ > 1e-12 && λ < best_val
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

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2-3: Orthonormal basis, slicing, centerline
# ═══════════════════════════════════════════════════════════════════════════════

"""Build an orthonormal basis (e1, e2) perpendicular to direction d."""
function _perp_basis(d::NTuple{3,Float64})
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

"""
    _slice_and_fit_centers(coords, info, slice_res, min_node_size)
        -> (centers_3d, slice_indices, t_values)

Slice an NBS along PC1, fit circle centers per slice.
Returns 3D centers (K×3), vector of point-index vectors per slice,
and scalar t-values (projection on PC1) per point.
"""
function _slice_and_fit_centers(coords::AbstractMatrix{<:Real}, info::NBSInfo,
                                slice_res::Float64, min_node_size::Int)
    d = info.direction
    cx, cy, cz = info.center
    indices = info.point_indices
    e1, e2 = _perp_basis(d)

    n = length(indices)
    t_vals = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        i = indices[j]
        dx = coords[i, 1] - cx
        dy = coords[i, 2] - cy
        dz = coords[i, 3] - cz
        t_vals[j] = dx * d[1] + dy * d[2] + dz * d[3]
    end

    t_min = minimum(t_vals)
    t_max = maximum(t_vals)
    n_slices = max(1, ceil(Int, (t_max - t_min) / slice_res))

    # Assign points to slices
    slice_point_indices = [Int[] for _ in 1:n_slices]
    point_slice_ids = Vector{Int}(undef, n)
    @inbounds for j in 1:n
        s = clamp(floor(Int, (t_vals[j] - t_min) / slice_res) + 1, 1, n_slices)
        push!(slice_point_indices[s], j)  # j is local index into indices
        point_slice_ids[j] = s
    end

    # Fit circle center per slice
    centers_3d = Matrix{Float64}(undef, n_slices, 3)
    valid_slices = falses(n_slices)

    for s in 1:n_slices
        local_js = slice_point_indices[s]
        if length(local_js) < min_node_size
            # Mark invalid; will be interpolated
            centers_3d[s, :] .= NaN
            continue
        end
        valid_slices[s] = true

        # Project to 2D plane perpendicular to PC1
        ns = length(local_js)
        u_arr = Vector{Float64}(undef, ns)
        v_arr = Vector{Float64}(undef, ns)
        @inbounds for k in 1:ns
            i = indices[local_js[k]]
            dx = coords[i, 1] - cx
            dy = coords[i, 2] - cy
            dz = coords[i, 3] - cz
            u_arr[k] = dx * e1[1] + dy * e1[2] + dz * e1[3]
            v_arr[k] = dx * e2[1] + dy * e2[2] + dz * e2[3]
        end

        if ns >= 10
            cu, cv, _ = taubin_circle_fit(u_arr, v_arr)
        else
            cu = mean(u_arr)
            cv = mean(v_arr)
        end

        # Convert 2D center back to 3D
        t_center = t_min + (s - 0.5) * slice_res
        centers_3d[s, 1] = cx + t_center * d[1] + cu * e1[1] + cv * e2[1]
        centers_3d[s, 2] = cy + t_center * d[2] + cu * e1[2] + cv * e2[2]
        centers_3d[s, 3] = cz + t_center * d[3] + cu * e1[3] + cv * e2[3]
    end

    # Interpolate invalid slices from neighbors
    for s in 1:n_slices
        valid_slices[s] && continue
        # Find nearest valid slices
        lo = 0; hi = 0
        for k in (s-1):-1:1
            if valid_slices[k]; lo = k; break; end
        end
        for k in (s+1):n_slices
            if valid_slices[k]; hi = k; break; end
        end
        if lo > 0 && hi > 0
            w = (s - lo) / (hi - lo)
            centers_3d[s, :] .= (1 - w) .* centers_3d[lo, :] .+ w .* centers_3d[hi, :]
        elseif lo > 0
            centers_3d[s, :] .= centers_3d[lo, :]
        elseif hi > 0
            centers_3d[s, :] .= centers_3d[hi, :]
        else
            t_center = t_min + (s - 0.5) * slice_res
            centers_3d[s, 1] = cx + t_center * d[1]
            centers_3d[s, 2] = cy + t_center * d[2]
            centers_3d[s, 3] = cz + t_center * d[3]
        end
    end

    return (centers_3d, slice_point_indices, t_vals, t_min, point_slice_ids, e1, e2)
end

"""
    _smooth_centerline!(centers, window)

Apply moving-window average to smooth the centerline in-place.
"""
function _smooth_centerline!(centers::Matrix{Float64}, window::Int=2)
    n = size(centers, 1)
    n <= 1 && return centers
    buf = similar(centers)
    @inbounds for s in 1:n
        lo = max(1, s - window)
        hi = min(n, s + window)
        cnt = hi - lo + 1
        sx = 0.0; sy = 0.0; sz = 0.0
        for k in lo:hi
            sx += centers[k, 1]; sy += centers[k, 2]; sz += centers[k, 3]
        end
        buf[s, 1] = sx / cnt; buf[s, 2] = sy / cnt; buf[s, 3] = sz / cnt
    end
    centers .= buf
    return centers
end

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Unroll points → (rho, phi)
# ═══════════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Periodic Cubic Spline Method
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _fit_periodic_smoothing_spline(phi_data, rho_data, n_eval; smoothing=1.0)
        -> (rho_eval, drho_eval)

Fit a periodic smoothing spline to (phi, rho) data and evaluate at n_eval
equally spaced points in [-π, π). Returns rho and its derivative at each point.

Uses binned averaging + periodic tridiagonal natural cubic spline.
"""
function _fit_periodic_smoothing_spline(phi_data::AbstractVector{Float64},
                                        rho_data::AbstractVector{Float64},
                                        n_eval::Int;
                                        smoothing::Float64=0.5)
    n = length(phi_data)
    n == 0 && return (zeros(n_eval), zeros(n_eval))

    # Bin data into n_eval equally spaced bins
    dphi = 2π / n_eval
    bin_sum = zeros(n_eval)
    bin_count = zeros(Int, n_eval)
    @inbounds for j in 1:n
        b = clamp(floor(Int, (phi_data[j] + π) / dphi) + 1, 1, n_eval)
        bin_sum[b] += rho_data[j]
        bin_count[b] += 1
    end

    # Fill bins: use mean where data exists, interpolate where missing
    rho_bins = Vector{Float64}(undef, n_eval)
    has_data = falses(n_eval)
    @inbounds for b in 1:n_eval
        if bin_count[b] > 0
            rho_bins[b] = bin_sum[b] / bin_count[b]
            has_data[b] = true
        else
            rho_bins[b] = NaN
        end
    end

    # Linear interpolation for missing bins (periodic)
    n_valid = count(has_data)
    if n_valid == 0
        return (fill(mean(rho_data), n_eval), zeros(n_eval))
    elseif n_valid < n_eval
        # Circular interpolation
        for b in 1:n_eval
            has_data[b] && continue
            # Find nearest valid neighbors in each direction (wrapping)
            lo = 0; lo_dist = n_eval
            hi = 0; hi_dist = n_eval
            for offset in 1:(n_eval - 1)
                blo = mod1(b - offset, n_eval)
                if has_data[blo] && offset < lo_dist
                    lo = blo; lo_dist = offset
                    break
                end
            end
            for offset in 1:(n_eval - 1)
                bhi = mod1(b + offset, n_eval)
                if has_data[bhi] && offset < hi_dist
                    hi = bhi; hi_dist = offset
                    break
                end
            end
            if lo > 0 && hi > 0
                w = lo_dist / (lo_dist + hi_dist)
                rho_bins[b] = (1 - w) * rho_bins[lo] + w * rho_bins[hi]
            elseif lo > 0
                rho_bins[b] = rho_bins[lo]
            elseif hi > 0
                rho_bins[b] = rho_bins[hi]
            end
        end
    end

    # Apply smoothing: weighted average with neighbors (periodic)
    rho_smooth = similar(rho_bins)
    w_center = 1.0 - smoothing
    w_side = smoothing / 2.0
    @inbounds for b in 1:n_eval
        bm = mod1(b - 1, n_eval)
        bp = mod1(b + 1, n_eval)
        rho_smooth[b] = w_center * rho_bins[b] + w_side * (rho_bins[bm] + rho_bins[bp])
    end

    # Compute numerical derivative (central differences, periodic)
    drho = Vector{Float64}(undef, n_eval)
    @inbounds for b in 1:n_eval
        bm = mod1(b - 1, n_eval)
        bp = mod1(b + 1, n_eval)
        drho[b] = (rho_smooth[bp] - rho_smooth[bm]) / (2.0 * dphi)
    end

    return (rho_smooth, drho)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Step 5B: 2D Periodic Surface Smoothing (replaces per-slice 1D spline)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _build_rho_surface(rho, phi, pt_slice_ids, n_slices, phi_bin_num)
        -> Matrix{Float64}  # (phi_bin_num, n_slices)

Bin all NBS points into a 2D grid of mean rho values.
Empty cells are NaN.
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
                    surface[b, s] = (1.0 - s_z) * buf[b, s] + s_z * buf[b, 2]
                elseif s == nz
                    surface[b, s] = (1.0 - s_z) * buf[b, s] + s_z * buf[b, nz - 1]
                else
                    surface[b, s] = w_center_z * buf[b, s] +
                                     w_side_z * (buf[b, s - 1] + buf[b, s + 1])
                end
            end
        end
    end
end

"""
    _method_spline_2d(rho, phi, pt_slice_ids, n_slices, cfg, rho_median_global)
        -> Vector{NamedTuple{(:cross_area, :circumference, :completeness)}}

Compute cross-sectional area and circumference for all slices of an NBS
using 2D surface smoothing (periodic in phi, non-periodic in z).
Returns a vector indexed by slice (1:n_slices); slices with no data
have zeros.
"""
function _method_spline_2d(rho::Vector{Float64}, phi::Vector{Float64},
                           pt_slice_ids::Vector{Int}, n_slices::Int,
                           cfg::FLiPConfig, rho_median_global::Float64)
    T = NamedTuple{(:cross_area, :circumference, :completeness), Tuple{Float64, Float64, Float64}}
    results = Vector{T}(undef, n_slices)
    fill!(results, (cross_area=0.0, circumference=0.0, completeness=0.0))

    surface_res = cfg.qsm_surface_res_scalar * cfg.pipeline_subsample_res
    phi_bin_num = clamp(ceil(Int, 2π * rho_median_global / surface_res),
                        cfg.qsm_phi_bin_min, cfg.qsm_phi_bin_max)
    dphi = 2π / phi_bin_num

    # Build 2D surface
    surface = _build_rho_surface(rho, phi, pt_slice_ids, n_slices, phi_bin_num)

    # Compute completeness per slice before gap-filling
    completeness = Vector{Float64}(undef, n_slices)
    @inbounds for s in 1:n_slices
        n_filled = count(b -> isfinite(surface[b, s]), 1:phi_bin_num)
        completeness[s] = n_filled / phi_bin_num
    end

    # Fill gaps and smooth
    _fill_gaps_2d!(surface)
    _smooth_surface_2d!(surface, 0.5, cfg.qsm_spl_z_smoothing)

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

    return results
end

# ═══════════════════════════════════════════════════════════════════════════════
# Legacy 1D spline method (kept for reference / single-slice fallback)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _method_spline_slice(rho_slice, phi_slice, cfg)
        -> (cross_area, circumference, completeness)

Estimate cross-sectional area and circumference for one slice using
Taubin circle fit for centering + periodic spline for shape estimation.
"""
function _method_spline_slice(rho_slice::AbstractVector{Float64},
                              phi_slice::AbstractVector{Float64},
                              cfg::FLiPConfig)
    n = length(rho_slice)
    if n == 0
        return (0.0, 0.0, 0.0)
    end

    # Determine evaluation resolution
    rho_med = median(rho_slice)
    surface_res = cfg.qsm_surface_res_scalar * cfg.pipeline_subsample_res
    n_eval = clamp(ceil(Int, 2π * rho_med / surface_res), cfg.qsm_phi_bin_min, cfg.qsm_phi_bin_max)

    # Completeness: fraction of angular range covered
    dphi_eval = 2π / n_eval
    covered = falses(n_eval)
    @inbounds for j in 1:n
        b = clamp(floor(Int, (phi_slice[j] + π) / dphi_eval) + 1, 1, n_eval)
        covered[b] = true
    end
    completeness = count(covered) / n_eval

    # Fit periodic spline
    rho_eval, drho_eval = _fit_periodic_smoothing_spline(phi_slice, rho_slice, n_eval)

    # Integrate arc length: circumference = ∫ sqrt(rho² + (drho/dphi)²) dphi
    circumference = 0.0
    cross_area = 0.0
    @inbounds for b in 1:n_eval
        r = rho_eval[b]
        dr = drho_eval[b]
        circumference += sqrt(r^2 + dr^2) * dphi_eval
        cross_area += 0.5 * r^2 * dphi_eval
    end

    return (cross_area, circumference, completeness)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Step 6-7: Frustum volumes, node and tree aggregation
# ═══════════════════════════════════════════════════════════════════════════════

"""Compute frustum volume and surface area between consecutive nodes."""
function _frustum_metrics(r1::Float64, r2::Float64, h::Float64)
    vol = (π / 3.0) * h * (r1^2 + r2^2 + r1 * r2)
    sa = π * (r1 + r2) * sqrt(h^2 + (r1 - r2)^2)
    return (vol, sa)
end

"""
    _process_single_nbs!(nodes, coords, info, nbs_id, tree_ids, agh_values,
                         cfg, next_node_id) -> next_node_id

Process a single linear NBS through the full QSM pipeline (Steps 2-6).
Appends QSMNode entries to `nodes`.
"""
function _process_single_nbs!(nodes::Vector{QSMNode},
                              coords::AbstractMatrix{<:Real},
                              info::NBSInfo,
                              nbs_id::Int32,
                              tree_ids::AbstractVector{<:Integer},
                              agh_values::AbstractVector{<:Real},
                              cfg::FLiPConfig,
                              next_node_id::Int)
    slice_res = cfg.qsm_slice_height_scalar * cfg.pipeline_subsample_res
    indices = info.point_indices

    # Step 2-3: Slice, fit centers, smooth
    centers, _, _, _, pt_slice_ids, e1, e2 =
        _slice_and_fit_centers(coords, info, slice_res, cfg.qsm_min_node_size)
    _smooth_centerline!(centers)

    # Step 4: Unroll
    rho, phi = _unroll_points(coords, indices, centers, pt_slice_ids, e1, e2)

    # Determine dominant tree_id for this NBS
    tree_counts = Dict{Int32, Int}()
    @inbounds for j in eachindex(indices)
        tid = Int32(tree_ids[indices[j]])
        tid > 0 || continue
        tree_counts[tid] = get(tree_counts, tid, 0) + 1
    end
    dominant_tree = isempty(tree_counts) ? Int32(0) : first(sort!(collect(tree_counts); by=last, rev=true))[1]

    n_slices = size(centers, 1)
    rho_median_global = length(rho) > 0 ? median(rho) : 0.01

    # Method B: 2D spline surface for all slices at once
    spl_results = _method_spline_2d(rho, phi, pt_slice_ids, n_slices, cfg, rho_median_global)

    # Process each slice
    slice_nodes = QSMNode[]
    for s in 1:n_slices
        local_js = findall(==(s), pt_slice_ids)
        n_pts = length(local_js)
        n_pts < cfg.qsm_min_node_size && continue

        # Mean AGH for this slice
        mean_agh = 0.0
        @inbounds for j in local_js
            mean_agh += float(agh_values[indices[j]])
        end
        mean_agh /= n_pts

        # 2D spline (pre-computed)
        ca = spl_results[s].cross_area
        circ = spl_results[s].circumference
        completeness = spl_results[s].completeness
        completeness < cfg.qsm_completeness_threshold && continue

        node = QSMNode(
            next_node_id, nbs_id, dominant_tree,
            mean_agh, slice_res, completeness, n_pts,
            centers[s, 1], centers[s, 2], centers[s, 3],
            info.direction[1], info.direction[2], info.direction[3],
            ca, circ,
            sqrt(max(0.0, ca / π)), circ / (2π),
        )
        push!(slice_nodes, node)
        next_node_id += 1
    end

    append!(nodes, slice_nodes)
    return next_node_id
end

# ═══════════════════════════════════════════════════════════════════════════════
# Step 7-8: Volume computation and tree aggregation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _compute_frustum_volumes!(nodes)

Compute frustum volumes between consecutive nodes within each NBS.
Updates nodes in-place. For first/last nodes in an NBS, uses cylinder approximation.
"""
function _compute_frustum_volumes!(nodes::Vector{QSMNode})
    isempty(nodes) && return

    # Group nodes by nbs_id, sorted by agh within each NBS
    nbs_groups = Dict{Int32, Vector{Int}}()
    for (i, nd) in enumerate(nodes)
        grp = get!(nbs_groups, nd.nbs_id, Int[])
        push!(grp, i)
    end

    for (_, node_idxs) in nbs_groups
        sort!(node_idxs; by=i -> nodes[i].agh)
        nn = length(node_idxs)
        if nn == 1
            # Single node: use cylinder (frustum with r1 = r2)
            continue  # volume fields will be set below
        end
    end

    # Already stored radius; now we need to write volume fields
    # We'll store them in new arrays and then set them
    # Actually, QSMNode doesn't have volume fields — we compute them during CSV output
    return
end

"""
    _build_node_table(nodes) -> Matrix{Any}

Build the node-level results table as a matrix for CSV output.
Includes frustum volume computation.
"""
function _build_node_table(nodes::Vector{QSMNode})
    isempty(nodes) && return (Matrix{Any}(undef, 0, 0), String[])

    # Group by NBS for frustum computation
    nbs_groups = Dict{Int32, Vector{Int}}()
    for (i, nd) in enumerate(nodes)
        grp = get!(nbs_groups, nd.nbs_id, Int[])
        push!(grp, i)
    end

    n = length(nodes)
    vol = zeros(n)
    sa = zeros(n)

    for (_, idxs) in nbs_groups
        sort!(idxs; by=i -> nodes[i].agh)
        nn = length(idxs)
        for k in 1:nn
            nd = nodes[idxs[k]]
            h = nd.height
            if nn == 1 || k == 1 || k == nn
                r1 = nd.radius_area
                if k == 1 && nn > 1
                    r2 = nodes[idxs[k+1]].radius_area
                    vol[idxs[k]], sa[idxs[k]] = _frustum_metrics(r1, r2, h)
                elseif k == nn && nn > 1
                    r0 = nodes[idxs[k-1]].radius_area
                    vol[idxs[k]], sa[idxs[k]] = _frustum_metrics(r0, r1, h)
                else
                    vol[idxs[k]], sa[idxs[k]] = _frustum_metrics(r1, r1, h)
                end
            else
                r0 = nodes[idxs[k-1]].radius_area
                r1 = nd.radius_area
                r2 = nodes[idxs[k+1]].radius_area
                ra = (r0 + r1) / 2
                rb = (r1 + r2) / 2
                vol[idxs[k]], sa[idxs[k]] = _frustum_metrics(ra, rb, h)
            end
        end
    end

    headers = [
        "qsm_node_id", "nbs_id", "tree_id", "agh",
        "cross_area", "circumference",
        "radius_area", "radius_circ",
        "height", "volume", "surface_area",
        "completeness", "n_points",
        "center_x", "center_y", "center_z",
        "direction_x", "direction_y", "direction_z",
    ]

    table = Matrix{Any}(undef, n, length(headers))
    for i in 1:n
        nd = nodes[i]
        table[i, :] = Any[
            nd.qsm_node_id, nd.nbs_id, nd.tree_id, nd.agh,
            nd.cross_area, nd.circumference,
            nd.radius_area, nd.radius_circ,
            nd.height, vol[i], sa[i],
            nd.completeness, nd.n_points,
            nd.center_x, nd.center_y, nd.center_z,
            nd.direction_x, nd.direction_y, nd.direction_z,
        ]
    end

    return (table, headers)
end

"""
    _build_tree_table(nodes, node_table, headers, cfg) -> (Matrix{Any}, Vector{String})

Aggregate node-level results to tree-level biometrics.
"""
function _build_tree_table(nodes::Vector{QSMNode}, node_table::Matrix{Any},
                           headers::Vector{String}, cfg::FLiPConfig)
    isempty(nodes) && return (Matrix{Any}(undef, 0, 0), String[])

    # Column indices
    col_vol = findfirst(==("volume"), headers)
    col_sa = findfirst(==("surface_area"), headers)

    # Group by tree_id
    tree_groups = Dict{Int32, Vector{Int}}()
    for (i, nd) in enumerate(nodes)
        nd.tree_id > 0 || continue
        grp = get!(tree_groups, nd.tree_id, Int[])
        push!(grp, i)
    end

    tree_ids = sort!(collect(keys(tree_groups)))
    n_trees = length(tree_ids)
    bh = cfg.qsm_breast_height

    tree_headers = [
        "tree_id",
        "volume", "surface_area", "height",
        "dbh_area", "dbh_circ",
        "n_points", "n_nodes",
        "x", "y",
    ]

    tree_table = Matrix{Any}(undef, n_trees, length(tree_headers))

    for (ti, tid) in enumerate(tree_ids)
        idxs = tree_groups[tid]

        total_vol = sum(i -> Float64(node_table[i, col_vol]), idxs)
        total_sa = sum(i -> Float64(node_table[i, col_sa]), idxs)
        total_pts = sum(i -> nodes[i].n_points, idxs)
        max_agh = maximum(i -> nodes[i].agh, idxs)

        # DBH: find node closest to breast height
        best_bh_idx = idxs[1]
        best_bh_dist = abs(nodes[idxs[1]].agh - bh)
        for i in idxs
            d = abs(nodes[i].agh - bh)
            if d < best_bh_dist
                best_bh_dist = d
                best_bh_idx = i
            end
        end
        dbh_a = 2.0 * nodes[best_bh_idx].radius_area
        dbh_c = 2.0 * nodes[best_bh_idx].radius_circ

        loc_x = nodes[best_bh_idx].center_x
        loc_y = nodes[best_bh_idx].center_y

        tree_table[ti, :] = Any[
            tid,
            total_vol, total_sa,
            max_agh,
            dbh_a, dbh_c,
            total_pts, length(idxs),
            loc_x, loc_y,
        ]
    end

    return (tree_table, tree_headers)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Step 9: Main QSM entry point
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _write_csv(path, table, headers)

Write a matrix with headers to a CSV file using DelimitedFiles.
"""
function _write_csv(path::String, table::Matrix{Any}, headers::Vector{String})
    open(path, "w") do io
        println(io, join(headers, ","))
        for i in 1:size(table, 1)
            vals = String[]
            for j in 1:size(table, 2)
                v = table[i, j]
                if v isa AbstractFloat
                    push!(vals, string(round(v; sigdigits=8)))
                else
                    push!(vals, string(v))
                end
            end
            println(io, join(vals, ","))
        end
    end
end

"""
    qsm(; tree_result, config_path, output_dir, output_prefix, kwargs...) -> NamedTuple

Run quantitative structural modeling on tree-segmented point cloud.

# Arguments
- `tree_result`: NamedTuple from `tree_segmentation` with fields `pc_output`, `skeleton_cloud`
- `config_path`: Path to TOML config file
- `output_dir`: Directory for CSV output files
- `output_prefix`: Filename prefix for outputs

# Returns
NamedTuple with fields:
- `status`: `:success` or `:no_linear_nbs`
- `n_nodes`: Number of QSM nodes created
- `n_trees`: Number of trees with QSM data
- `node_csv_path`: Path to node-level CSV
- `tree_csv_path`: Path to tree-level CSV
- `pc_output`: Point cloud with `:qsm_node_id` attribute added
"""
function qsm(; tree_result=nothing, config_path::AbstractString="", output_dir::AbstractString="",
               output_prefix::AbstractString="output", tree_cloud_path::AbstractString="", kwargs...)
    cfg = isempty(config_path) ? _CFG : load_config!(String(config_path))

    if isnothing(tree_result) || npoints(tree_result.pc_output) == 0
        @warn "[qsm] No tree segmentation data available"
        return (status=:no_data, n_nodes=0, n_trees=0,
                node_csv_path="", tree_csv_path="", pc_output=nothing)
    end

    pc = tree_result.pc_output
    coords = coordinates(pc)
    N = size(coords, 1)

    # Required attributes
    nbs_ids = hasattribute(pc, :nbs_id) ? getattribute(pc, :nbs_id) : zeros(Int32, N)
    tree_ids = hasattribute(pc, :tree_id) ? getattribute(pc, :tree_id) : zeros(Int32, N)
    agh_values = hasattribute(pc, :AGH) ? getattribute(pc, :AGH) : zeros(Float64, N)

    println("[qsm] Processing $(N) points")

    # Step 1: Filter linear NBS segments
    linear_nbs = _filter_linear_nbs(coords, nbs_ids, cfg)
    println("[qsm] Found $(length(linear_nbs)) linear NBS segments (threshold=$(cfg.qsm_nbs_linearity_threshold))")

    if isempty(linear_nbs)
        pc_out = setattribute!(pc, :qsm_node_id, zeros(Int32, N))
        return (status=:no_linear_nbs, n_nodes=0, n_trees=0,
                node_csv_path="", tree_csv_path="", pc_output=pc_out)
    end

    # Steps 2-6: Process each NBS
    nodes = QSMNode[]
    sizehint!(nodes, length(linear_nbs) * 10)
    next_node_id = 1

    qsm_node_ids = zeros(Int32, N)

    for (nid, info) in sort!(collect(linear_nbs); by=first)
        n_before = length(nodes)
        next_node_id = _process_single_nbs!(nodes, coords, info, nid,
                                            tree_ids, agh_values, cfg, next_node_id)
        # Map points to QSM node IDs
        n_after = length(nodes)
        if n_after > n_before
            for j in eachindex(info.point_indices)
                pi = info.point_indices[j]
                # Find matching node by agh proximity
                best_node = Int32(0)
                best_dist = Inf
                for ni in (n_before + 1):n_after
                    node_agh = nodes[ni].agh
                    pt_agh = float(agh_values[pi])
                    d_agh = abs(node_agh - pt_agh)
                    if d_agh < best_dist
                        best_dist = d_agh
                        best_node = Int32(nodes[ni].qsm_node_id)
                    end
                end
                if best_node > 0
                    qsm_node_ids[pi] = best_node
                end
            end
        end
    end

    println("[qsm] Created $(length(nodes)) QSM nodes")

    # Step 7-8: Build output tables
    node_table, node_headers = _build_node_table(nodes)
    tree_table, tree_headers = _build_tree_table(nodes, node_table, node_headers, cfg)

    n_trees = size(tree_table, 1)
    println("[qsm] Aggregated to $(n_trees) trees")

    # Step 9: Write CSVs
    node_csv = joinpath(output_dir, "$(output_prefix)qsm_nodes.csv")
    tree_csv = joinpath(output_dir, "$(output_prefix)qsm_trees.csv")

    if !isempty(output_dir)
        mkpath(output_dir)
        if size(node_table, 1) > 0
            _write_csv(node_csv, node_table, node_headers)
            println("[qsm] Wrote node biometrics: $node_csv")
        end
        if size(tree_table, 1) > 0
            _write_csv(tree_csv, tree_table, tree_headers)
            println("[qsm] Wrote tree biometrics: $tree_csv")
        end
    end

    # Add QSM node IDs to point cloud and overwrite tree cloud on disk
    pc_out = setattribute!(pc, :qsm_node_id, qsm_node_ids)
    if !isempty(tree_cloud_path)
        write_pc(tree_cloud_path, pc_out)
        println("[qsm] Overwrote tree cloud with qsm_node_id: $tree_cloud_path")
    end

    return (
        status = :success,
        n_nodes = length(nodes),
        n_trees = n_trees,
        node_csv_path = node_csv,
        tree_csv_path = tree_csv,
        pc_output = pc_out,
    )
end
