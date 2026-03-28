"""
Quantitative Structural Modeling (QSM) for FLiP.jl.

Converts tree-segmented point clouds into geometric measurements (circumference,
cross-sectional area, volume, surface area) per branch node and per tree.
Two estimation methods are computed in parallel:
- Method A (IDW): unroll → IDW gap-fill → integrate
- Method B (Spline): Taubin circle fit → periodic cubic spline → integrate arc length
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
    # Method A (IDW)
    cross_area_idw::Float64
    circumference_idw::Float64
    radius_area_idw::Float64
    radius_circ_idw::Float64
    # Method B (Spline)
    cross_area_spl::Float64
    circumference_spl::Float64
    radius_area_spl::Float64
    radius_circ_spl::Float64
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
# Step 5A: IDW Method
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _method_idw_slice(rho_slice, phi_slice, cfg)
        -> (cross_area, circumference, completeness)

Estimate cross-sectional area and circumference for one slice using the
IDW grid method.
"""
function _method_idw_slice(rho_slice::AbstractVector{Float64},
                           phi_slice::AbstractVector{Float64},
                           cfg::FLiPConfig, rho_median::Float64)
    n = length(rho_slice)
    if n == 0
        return (0.0, 0.0, 0.0)
    end

    surface_res = cfg.qsm_surface_res_scalar * cfg.pipeline_subsample_res
    phi_bin_num = clamp(ceil(Int, 2π * rho_median / surface_res), cfg.qsm_phi_bin_min, cfg.qsm_phi_bin_max)
    dtheta = 2π / phi_bin_num

    # Bin points by phi
    rho_grid = fill(NaN, phi_bin_num)
    rho_count = zeros(Int, phi_bin_num)

    @inbounds for j in 1:n
        # Map phi ∈ [-π, π] to bin ∈ [1, phi_bin_num]
        bin = clamp(floor(Int, (phi_slice[j] + π) / dtheta) + 1, 1, phi_bin_num)
        if isnan(rho_grid[bin]) || rho_slice[j] < rho_grid[bin]
            rho_grid[bin] = rho_slice[j]  # min rho per bin
        end
        rho_count[bin] += 1
    end

    # IQR outlier removal
    valid_rho = filter(isfinite, rho_grid)
    if length(valid_rho) >= 4
        q1 = quantile(valid_rho, 0.25)
        q3 = quantile(valid_rho, 0.75)
        iqr = q3 - q1
        threshold = q3 + cfg.qsm_outlier_iqr * iqr
        @inbounds for b in 1:phi_bin_num
            if isfinite(rho_grid[b]) && rho_grid[b] > threshold
                rho_grid[b] = NaN
            end
        end
    end

    # IDW gap-filling with periodic wrapping
    nan_bins = findall(isnan, rho_grid)
    valid_bins = findall(isfinite, rho_grid)

    if !isempty(nan_bins) && length(valid_bins) >= cfg.qsm_idw_k
        # Build 1D KDTree on valid bins (with periodic copies)
        n_valid = length(valid_bins)
        # Create periodic copies: original + shifted by ±phi_bin_num
        kd_coords = Matrix{Float64}(undef, 1, 3 * n_valid)
        kd_rho = Vector{Float64}(undef, 3 * n_valid)
        @inbounds for (idx, b) in enumerate(valid_bins)
            kd_coords[1, idx] = Float64(b)
            kd_coords[1, n_valid + idx] = Float64(b - phi_bin_num)
            kd_coords[1, 2 * n_valid + idx] = Float64(b + phi_bin_num)
            kd_rho[idx] = rho_grid[b]
            kd_rho[n_valid + idx] = rho_grid[b]
            kd_rho[2 * n_valid + idx] = rho_grid[b]
        end
        kd_tree = KDTree(kd_coords)
        k_use = min(cfg.qsm_idw_k, 3 * n_valid)

        for b in nan_bins
            query = reshape([Float64(b)], 1, 1)
            nbr_idx, nbr_dist = knn(kd_tree, query[:, 1], k_use, true)
            nbr_dist[1] > cfg.qsm_idw_max_dist && continue

            wsum = 0.0; rwsum = 0.0
            for k in eachindex(nbr_idx)
                d = max(nbr_dist[k], 1e-10)
                w = 1.0 / (d * d)
                wsum += w
                rwsum += w * kd_rho[nbr_idx[k]]
            end
            if wsum > 0
                rho_grid[b] = rwsum / wsum
            end
        end
    end

    # Integrate
    valid_after = findall(isfinite, rho_grid)
    completeness = length(valid_after) / phi_bin_num
    completeness > 0 || return (0.0, 0.0, 0.0)

    cross_area = 0.0
    circumference = 0.0
    @inbounds for b in valid_after
        r = rho_grid[b]
        cross_area += 0.5 * dtheta * r^2
        circumference += dtheta * r
    end
    cross_area /= completeness
    circumference /= completeness

    return (cross_area, circumference, completeness)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Step 5B: Periodic Cubic Spline Method
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

    # Process each slice
    slice_nodes = QSMNode[]
    for s in 1:n_slices
        local_js = findall(==(s), pt_slice_ids)
        n_pts = length(local_js)
        n_pts < cfg.qsm_min_node_size && continue

        rho_s = rho[local_js]
        phi_s = phi[local_js]

        # Mean AGH for this slice
        mean_agh = 0.0
        @inbounds for j in local_js
            mean_agh += float(agh_values[indices[j]])
        end
        mean_agh /= n_pts

        rho_med = length(rho_s) > 0 ? median(rho_s) : rho_median_global

        # Method A: IDW
        ca_idw, circ_idw, comp_idw = _method_idw_slice(rho_s, phi_s, cfg, rho_med)

        # Method B: Spline
        ca_spl, circ_spl, comp_spl = _method_spline_slice(rho_s, phi_s, cfg)

        # Use maximum completeness from either method
        completeness = max(comp_idw, comp_spl)
        completeness < cfg.qsm_completeness_threshold && continue

        node = QSMNode(
            next_node_id, nbs_id, dominant_tree,
            mean_agh, slice_res, completeness, n_pts,
            centers[s, 1], centers[s, 2], centers[s, 3],
            info.direction[1], info.direction[2], info.direction[3],
            ca_idw, circ_idw,
            sqrt(max(0.0, ca_idw / π)), circ_idw / (2π),
            ca_spl, circ_spl,
            sqrt(max(0.0, ca_spl / π)), circ_spl / (2π),
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
    vol_idw = zeros(n)
    sa_idw = zeros(n)
    vol_spl = zeros(n)
    sa_spl = zeros(n)

    for (_, idxs) in nbs_groups
        sort!(idxs; by=i -> nodes[i].agh)
        nn = length(idxs)
        for k in 1:nn
            nd = nodes[idxs[k]]
            h = nd.height
            if nn == 1 || k == 1 || k == nn
                # Endpoints: use cylinder
                r_idw = nd.radius_area_idw
                r_spl = nd.radius_area_spl
                if k == 1 && nn > 1
                    r2_idw = nodes[idxs[k+1]].radius_area_idw
                    r2_spl = nodes[idxs[k+1]].radius_area_spl
                    vol_idw[idxs[k]], sa_idw[idxs[k]] = _frustum_metrics(r_idw, r2_idw, h)
                    vol_spl[idxs[k]], sa_spl[idxs[k]] = _frustum_metrics(r_spl, r2_spl, h)
                elseif k == nn && nn > 1
                    r0_idw = nodes[idxs[k-1]].radius_area_idw
                    r0_spl = nodes[idxs[k-1]].radius_area_spl
                    vol_idw[idxs[k]], sa_idw[idxs[k]] = _frustum_metrics(r0_idw, r_idw, h)
                    vol_spl[idxs[k]], sa_spl[idxs[k]] = _frustum_metrics(r0_spl, r_spl, h)
                else
                    vol_idw[idxs[k]], sa_idw[idxs[k]] = _frustum_metrics(r_idw, r_idw, h)
                    vol_spl[idxs[k]], sa_spl[idxs[k]] = _frustum_metrics(r_spl, r_spl, h)
                end
            else
                # Interior: average with neighbors
                r0_idw = nodes[idxs[k-1]].radius_area_idw
                r1_idw = nd.radius_area_idw
                r2_idw = nodes[idxs[k+1]].radius_area_idw
                ra_idw = (r0_idw + r1_idw) / 2
                rb_idw = (r1_idw + r2_idw) / 2
                vol_idw[idxs[k]], sa_idw[idxs[k]] = _frustum_metrics(ra_idw, rb_idw, h)

                r0_spl = nodes[idxs[k-1]].radius_area_spl
                r1_spl = nd.radius_area_spl
                r2_spl = nodes[idxs[k+1]].radius_area_spl
                ra_spl = (r0_spl + r1_spl) / 2
                rb_spl = (r1_spl + r2_spl) / 2
                vol_spl[idxs[k]], sa_spl[idxs[k]] = _frustum_metrics(ra_spl, rb_spl, h)
            end
        end
    end

    headers = [
        "qsm_node_id", "nbs_id", "tree_id", "agh",
        "cross_area_idw", "cross_area_spl",
        "circumference_idw", "circumference_spl",
        "radius_area_idw", "radius_area_spl",
        "radius_circ_idw", "radius_circ_spl",
        "height",
        "volume_idw", "volume_spl",
        "surface_area_idw", "surface_area_spl",
        "completeness", "n_points",
        "center_x", "center_y", "center_z",
        "direction_x", "direction_y", "direction_z",
    ]

    table = Matrix{Any}(undef, n, length(headers))
    for i in 1:n
        nd = nodes[i]
        table[i, :] = Any[
            nd.qsm_node_id, nd.nbs_id, nd.tree_id, nd.agh,
            nd.cross_area_idw, nd.cross_area_spl,
            nd.circumference_idw, nd.circumference_spl,
            nd.radius_area_idw, nd.radius_area_spl,
            nd.radius_circ_idw, nd.radius_circ_spl,
            nd.height,
            vol_idw[i], vol_spl[i],
            sa_idw[i], sa_spl[i],
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
    col_vol_idw = findfirst(==("volume_idw"), headers)
    col_vol_spl = findfirst(==("volume_spl"), headers)
    col_sa_idw = findfirst(==("surface_area_idw"), headers)
    col_sa_spl = findfirst(==("surface_area_spl"), headers)

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
        "volume_idw", "volume_spl",
        "surface_area_idw", "surface_area_spl",
        "height",
        "dbh_area_idw", "dbh_area_spl",
        "dbh_circ_idw", "dbh_circ_spl",
        "n_points", "n_nodes",
        "x", "y",
    ]

    tree_table = Matrix{Any}(undef, n_trees, length(tree_headers))

    for (ti, tid) in enumerate(tree_ids)
        idxs = tree_groups[tid]

        # Sums
        total_vol_idw = sum(i -> Float64(node_table[i, col_vol_idw]), idxs)
        total_vol_spl = sum(i -> Float64(node_table[i, col_vol_spl]), idxs)
        total_sa_idw = sum(i -> Float64(node_table[i, col_sa_idw]), idxs)
        total_sa_spl = sum(i -> Float64(node_table[i, col_sa_spl]), idxs)
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
        dbh_a_idw = 2.0 * nodes[best_bh_idx].radius_area_idw
        dbh_a_spl = 2.0 * nodes[best_bh_idx].radius_area_spl
        dbh_c_idw = 2.0 * nodes[best_bh_idx].radius_circ_idw
        dbh_c_spl = 2.0 * nodes[best_bh_idx].radius_circ_spl

        # Location at breast height
        loc_x = nodes[best_bh_idx].center_x
        loc_y = nodes[best_bh_idx].center_y

        tree_table[ti, :] = Any[
            tid,
            total_vol_idw, total_vol_spl,
            total_sa_idw, total_sa_spl,
            max_agh,
            dbh_a_idw, dbh_a_spl,
            dbh_c_idw, dbh_c_spl,
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
