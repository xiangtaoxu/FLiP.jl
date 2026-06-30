"""
Finite-cylinder geometry: point-in-cylinder, AABB, AABB overlap, deterministic voxelized
cylinder-union volume (fixed global lattice), plus the `Cyl` type + refinement helpers.
"""

"""
    point_in_cylinder(p, center, axis, radius, half_height) -> Bool

Return `true` iff point `p` lies inside the finite cylinder with midpoint
`center`, unit `axis`, `radius`, and `half_height` (= height / 2). Projects the
offset `p - center` onto `axis` (axial cut at `±half_height`), then tests the
squared radial distance. Returns `false` for non-positive / non-finite geometry.
All arguments are `NTuple{3,Float64}` / `Float64`; `axis` is assumed unit-length.
"""
@inline function point_in_cylinder(p::NTuple{3,Float64}, center::NTuple{3,Float64},
                                   axis::NTuple{3,Float64}, radius::Float64, half_height::Float64)
    (isfinite(radius) && radius > 0 && isfinite(half_height) && half_height > 0) || return false
    wx = p[1] - center[1]; wy = p[2] - center[2]; wz = p[3] - center[3]
    t = wx * axis[1] + wy * axis[2] + wz * axis[3]
    abs(t) > half_height && return false
    r2 = (wx * wx + wy * wy + wz * wz) - t * t
    return r2 <= radius * radius
end

"""
    cylinder_aabb(center, axis, radius, half_height) -> NTuple{6,Float64}

Tight axis-aligned bounding box `(xmin,xmax,ymin,ymax,zmin,zmax)` of a finite
cylinder. The half-extent along world axis `k` is
`half_height·|axis_k| + radius·√(1 − axis_k²)` (the axial projection plus the
radius of the end-cap disk projected onto that axis).
"""
function cylinder_aabb(center::NTuple{3,Float64}, axis::NTuple{3,Float64},
                       radius::Float64, half_height::Float64)
    ex = half_height * abs(axis[1]) + radius * sqrt(max(0.0, 1.0 - axis[1]^2))
    ey = half_height * abs(axis[2]) + radius * sqrt(max(0.0, 1.0 - axis[2]^2))
    ez = half_height * abs(axis[3]) + radius * sqrt(max(0.0, 1.0 - axis[3]^2))
    return (center[1] - ex, center[1] + ex,
            center[2] - ey, center[2] + ey,
            center[3] - ez, center[3] + ez)
end

"""
    aabbs_overlap(a, b) -> Bool

Whether two AABBs (each `(xmin,xmax,ymin,ymax,zmin,zmax)`) intersect. Cheap
candidate reject before any volume work.
"""
@inline aabbs_overlap(a::NTuple{6,Float64}, b::NTuple{6,Float64}) =
    a[1] <= b[2] && b[1] <= a[2] &&
    a[3] <= b[4] && b[3] <= a[4] &&
    a[5] <= b[6] && b[5] <= a[6]

"""
    voxelized_cylinder_volume(cyls, box, voxel_res) -> Float64

Deterministic volume of the **union** of finite cylinders `cyls`, estimated by
counting global-lattice voxel centers that fall inside any cylinder, × `voxel_res³`.

`box = (xmin,xmax,ymin,ymax,zmin,zmax)` bounds the scan; only voxel cells in that
range are visited. Voxel centers are anchored globally at `(k + 0.5)·voxel_res`
(integer `k`), so the SAME cells are tested for any `box` — this is what makes a
segment's self-volume and a pair's intersection (scanned over a sub-box) use
identical cells, guaranteeing `intersection ≤ min(self volumes)` and a
reproducible result independent of thread scheduling.

Each element of `cyls` must expose fields `center::NTuple{3,Float64}`,
`axis::NTuple{3,Float64}`, `radius::Float64`, `half_height::Float64`. A per-cylinder
AABB pre-filter skips the radial test for cells a cylinder cannot contain.
"""
function voxelized_cylinder_volume(cyls, box::NTuple{6,Float64}, voxel_res::Float64)
    (isempty(cyls) || !(voxel_res > 0)) && return 0.0
    caabb = NTuple{6,Float64}[cylinder_aabb(c.center, c.axis, c.radius, c.half_height) for c in cyls]
    kx0 = floor(Int, box[1] / voxel_res); kx1 = ceil(Int, box[2] / voxel_res) - 1
    ky0 = floor(Int, box[3] / voxel_res); ky1 = ceil(Int, box[4] / voxel_res) - 1
    kz0 = floor(Int, box[5] / voxel_res); kz1 = ceil(Int, box[6] / voxel_res) - 1
    cnt = 0
    @inbounds for kx in kx0:kx1
        cx = (kx + 0.5) * voxel_res
        for ky in ky0:ky1
            cy = (ky + 0.5) * voxel_res
            for kz in kz0:kz1
                cz = (kz + 0.5) * voxel_res
                p = (cx, cy, cz)
                for m in eachindex(cyls)
                    a = caabb[m]
                    (cx >= a[1] && cx <= a[2] && cy >= a[3] && cy <= a[4] && cz >= a[5] && cz <= a[6]) || continue
                    c = cyls[m]
                    if point_in_cylinder(p, c.center, c.axis, c.radius, c.half_height)
                        cnt += 1
                        break
                    end
                end
            end
        end
    end
    return cnt * voxel_res^3
end


# ── Cylinder convenience type + refinement geometry helpers (promoted from refine) ──

# A finite cylinder from one trial-QSM node: midpoint, unit axis, radius, half-height.
const Cyl = NamedTuple{(:center, :axis, :radius, :half_height),
                       Tuple{NTuple{3,Float64}, NTuple{3,Float64}, Float64, Float64}}

@inline function _unit3(x::Float64, y::Float64, z::Float64)
    n = sqrt(x * x + y * y + z * z)
    return n > 0 ? (x / n, y / n, z / n) : (0.0, 0.0, 1.0)
end

@inline _aabb_intersection(a::NTuple{6,Float64}, b::NTuple{6,Float64}) =
    (max(a[1], b[1]), min(a[2], b[2]),
     max(a[3], b[3]), min(a[4], b[4]),
     max(a[5], b[5]), min(a[6], b[6]))

@inline function _point_in_any(p::NTuple{3,Float64}, cyls::Vector{Cyl})
    @inbounds for c in cyls
        point_in_cylinder(p, c.center, c.axis, c.radius, c.half_height) && return true
    end
    return false
end

"""
    _voxel_intersection_volume(cyls_a, cyls_b, box, voxel_res) -> Float64

Deterministic volume of the 3-D intersection of two cylinder unions, restricted to
`box`, on the same global lattice as `voxelized_cylinder_volume` (a voxel counts iff its
center is inside some cylinder of `cyls_a` AND some cylinder of `cyls_b`). The shared
lattice makes the result ≤ each side's self-volume, so the per-node overlap ratio is ≤ 1.
"""
function _voxel_intersection_volume(cyls_a::Vector{Cyl}, cyls_b::Vector{Cyl},
                                    box::NTuple{6,Float64}, voxel_res::Float64)
    (!(voxel_res > 0) || isempty(cyls_a) || isempty(cyls_b)) && return 0.0
    kx0 = floor(Int, box[1] / voxel_res); kx1 = ceil(Int, box[2] / voxel_res) - 1
    ky0 = floor(Int, box[3] / voxel_res); ky1 = ceil(Int, box[4] / voxel_res) - 1
    kz0 = floor(Int, box[5] / voxel_res); kz1 = ceil(Int, box[6] / voxel_res) - 1
    cnt = 0
    @inbounds for kx in kx0:kx1
        cx = (kx + 0.5) * voxel_res
        for ky in ky0:ky1
            cy = (ky + 0.5) * voxel_res
            for kz in kz0:kz1
                cz = (kz + 0.5) * voxel_res
                p = (cx, cy, cz)
                (_point_in_any(p, cyls_a) && _point_in_any(p, cyls_b)) && (cnt += 1)
            end
        end
    end
    return cnt * voxel_res^3
end

