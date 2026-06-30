"""
QSM surface point-cloud generation from a fitted 2D rho-surface (a reporting artifact built
from per-NBS fitting internals during modeling).
"""

"""
    _generate_surface_points(surface, centers, e1, e2, slice_res, gen_res, T=Float64)
        -> (Matrix{T}, Vector{T})  # (M×3 coords, M surface rho values)

Convert a smoothed 2D rho surface (phi × z-slice) to 3D xyz points.
Linearly interpolates between z-slices to achieve approximately `gen_res`
axial spacing. Returns coordinates and the per-point surface radius (rho),
which equals the smoothed surface value at each generated point (slice points
take `surface[b, s]`; inter-slice points take the linear blend between
slices s and s+1). Geometry is computed in Float64 and stored as `T` (the
cloud's coordinate precision), so the surface cloud matches the input cloud.
"""
function _generate_surface_points(surface::Matrix{Float64},
                                   centers::Matrix{Float64},
                                   e1::NTuple{3,Float64},
                                   e2::NTuple{3,Float64},
                                   slice_res::Float64,
                                   gen_res::Float64,
                                   ::Type{T}=Float64) where {T}
    nphi, nslices = size(surface)
    dphi = 2π / nphi
    n_zsub = max(1, ceil(Int, slice_res / gen_res))

    # Pass 1: count emitted points exactly (predicates mirror the emission below),
    # so we allocate the precise output size — no upper-bound over-allocation and
    # no trailing slice-copy to trim it.
    n_count = 0
    @inbounds for s in 1:nslices
        for b in 1:nphi
            r = surface[b, s]
            (isfinite(r) && r > 0) && (n_count += 1)
        end
        if s < nslices && n_zsub > 1
            for _ in 1:(n_zsub - 1), b in 1:nphi
                r1 = surface[b, s]; r2 = surface[b, s+1]
                (isfinite(r1) && isfinite(r2) && r1 > 0 && r2 > 0) && (n_count += 1)
            end
        end
    end

    pts = Matrix{T}(undef, n_count, 3)
    rho_vals = Vector{T}(undef, n_count)
    idx = 0

    @inbounds for s in 1:nslices
        # Points at slice center
        for b in 1:nphi
            rho = surface[b, s]
            (isfinite(rho) && rho > 0) || continue
            phi_val = -π + (b - 0.5) * dphi
            u = rho * cos(phi_val)
            v = rho * sin(phi_val)
            idx += 1
            pts[idx, 1] = T(centers[s, 1] + u * e1[1] + v * e2[1])
            pts[idx, 2] = T(centers[s, 2] + u * e1[2] + v * e2[2])
            pts[idx, 3] = T(centers[s, 3] + u * e1[3] + v * e2[3])
            rho_vals[idx] = T(rho)
        end

        # Interpolated points between slice s and s+1
        if s < nslices && n_zsub > 1
            for k in 1:(n_zsub - 1)
                frac = k / n_zsub
                icx = centers[s, 1] + frac * (centers[s+1, 1] - centers[s, 1])
                icy = centers[s, 2] + frac * (centers[s+1, 2] - centers[s, 2])
                icz = centers[s, 3] + frac * (centers[s+1, 3] - centers[s, 3])
                for b in 1:nphi
                    rho1 = surface[b, s]
                    rho2 = surface[b, s+1]
                    (isfinite(rho1) && isfinite(rho2) && rho1 > 0 && rho2 > 0) || continue
                    rho = rho1 + frac * (rho2 - rho1)
                    phi_val = -π + (b - 0.5) * dphi
                    u = rho * cos(phi_val)
                    v = rho * sin(phi_val)
                    idx += 1
                    pts[idx, 1] = T(icx + u * e1[1] + v * e2[1])
                    pts[idx, 2] = T(icy + u * e1[2] + v * e2[2])
                    pts[idx, 3] = T(icz + u * e1[3] + v * e2[3])
                    rho_vals[idx] = T(rho)
                end
            end
        end
    end

    return (pts, rho_vals)
end

# ───────────────────────────────────────────────────────────────────────────────
# 2e. Frustum geometry
# ───────────────────────────────────────────────────────────────────────────────


"""
    assemble_surface_cloud(parts) -> PointCloud

Merge the per-NBS surface parts from `model_nbs(emit_surface=true)` (a NamedTuple
`(coords, nbs, rho, T)`) into a single surface point cloud carrying `:tree_nbs_id` and `:rho`.
Returns an empty cloud when there are no parts.
"""
function assemble_surface_cloud(parts)
    Tc = parts.T
    isempty(parts.coords) && return PointCloud(Matrix{Tc}(undef, 0, 3), Dict{Symbol,Vector}())
    S = sum(p -> size(p, 1), parts.coords)
    surf_coords = Matrix{Tc}(undef, S, 3)
    surf_nbs = Vector{Int32}(undef, S); surf_rho = Vector{Tc}(undef, S)
    off = 0
    @inbounds for k in eachindex(parts.coords)
        m = size(parts.coords[k], 1)
        copyto!(view(surf_coords, off+1:off+m, :), parts.coords[k])
        copyto!(view(surf_nbs,    off+1:off+m),    parts.nbs[k])
        copyto!(view(surf_rho,    off+1:off+m),    parts.rho[k])
        off += m
    end
    return PointCloud(surf_coords, Dict{Symbol,Vector}(:tree_nbs_id => surf_nbs, :rho => surf_rho))
end
