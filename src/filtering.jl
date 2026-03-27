"""
Filtering algorithms for point cloud noise removal and segmentation.
"""

"""
    statistical_filter_indices(points::AbstractMatrix{<:Real}, k_neighbors::Int=6, 
                               n_sigma::Real=1) -> Vector{Int}

Statistical outlier removal filter returning indices of inlier points.

For each point, compute the mean distance to its K nearest neighbors. Points whose
mean distance exceeds `mean + n_sigma * std` across all points are considered outliers.

# Arguments
- `points`: N×3 matrix of XYZ coordinates
- `k_neighbors`: Number of nearest neighbors to consider (default: 6)
- `n_sigma`: Number of standard deviations for threshold (default: 1)

# Returns
- `Vector{Int}`: Indices of inlier points

# Example
```julia
coords = rand(10000, 3)
indices = statistical_filter_indices(coords, 6, 1)
filtered = coords[indices, :]
```
"""
function statistical_filter_indices(points::AbstractMatrix{<:Real},
                                    k_neighbors::Int=_CFG.statistical_filter_k_neighbors,
                                    n_sigma::Real=_CFG.statistical_filter_n_sigma)
    size(points, 2) == 3 || throw(ArgumentError("points must be N×3 matrix"))
    k_neighbors > 0 || throw(ArgumentError("k_neighbors must be > 0"))
    n_sigma > 0 || throw(ArgumentError("n_sigma must be > 0"))
    
    n = size(points, 1)
    
    # Need at least k+1 points for k-nearest neighbors
    if n <= k_neighbors
        return collect(1:n)
    end
    
    # Build KDTree for efficient nearest neighbor queries
    tree = KDTree(points')  # NearestNeighbors expects D×N matrix
    
    # Compute mean distance to k nearest neighbors for each point
    mean_dists = Vector{Float64}(undef, n)
    
    @inbounds for i in 1:n
        # Query k+1 neighbors (includes the point itself)
        idxs, dists = knn(tree, points[i, :], k_neighbors + 1)
        
        # Skip the first neighbor (the point itself with distance 0)
        neighbor_dists = dists[2:end]
        mean_dists[i] = mean(neighbor_dists)
    end
    
    # Compute statistics
    μ = mean(mean_dists)
    σ = std(mean_dists)
    threshold = μ + n_sigma * σ
    
    # Filter points below threshold
    return findall(d -> d <= threshold, mean_dists)
end

"""
    statistical_filter(pc::PointCloud, k_neighbors::Int=6, n_sigma::Real=1) -> PointCloud

Statistical outlier removal filter for point clouds.

# Arguments
- `pc`: Input PointCloud
- `k_neighbors`: Number of nearest neighbors to consider (default: 6)
- `n_sigma`: Number of standard deviations for threshold (default: 1)

# Returns
- `PointCloud`: Filtered point cloud with outliers removed

# Example
```julia
pc = read_las("input.laz")
pc_clean = statistical_filter(pc, 6, 1)
```
"""
function statistical_filter(pc::PointCloud,
                            k_neighbors::Int=_CFG.statistical_filter_k_neighbors,
                            n_sigma::Real=_CFG.statistical_filter_n_sigma)
    indices = statistical_filter_indices(coordinates(pc), k_neighbors, n_sigma)
    return pc[indices]
end

"""
    grid_zmin_filter_indices(points::AbstractMatrix{<:Real}, grid_size::Real) -> Vector{Int}

Grid minimum-z filter returning one index per XY cell.

The XY plane is partitioned into square cells of size `grid_size`, and only the point
with minimum z value in each cell is kept.

# Arguments
- `points`: N×3 matrix of XYZ coordinates
- `grid_size`: XY grid size (must be > 0)

# Returns
- `Vector{Int}`: Indices of kept points (sorted ascending)

# Example
```julia
coords = rand(10000, 3)
indices = grid_zmin_filter_indices(coords, 1.0)
filtered = coords[indices, :]
```
"""
function grid_zmin_filter_indices(points::AbstractMatrix{<:Real}, grid_size::Real)
    size(points, 2) == 3 || throw(ArgumentError("points must be N×3 matrix"))
    grid_size > 0 || throw(ArgumentError("grid_size must be > 0"))

    n = size(points, 1)
    n == 0 && return Int[]

    inv_grid_size = 1.0 / float(grid_size)
    zmin_by_cell = Dict{NTuple{2, Int}, Tuple{Float64, Int}}()

    @inbounds for i in 1:n
        cx = floor(Int, float(points[i, 1]) * inv_grid_size)
        cy = floor(Int, float(points[i, 2]) * inv_grid_size)
        zi = float(points[i, 3])
        cell = (cx, cy)

        current = get(zmin_by_cell, cell, (Inf, typemax(Int)))
        if zi < current[1] || (zi == current[1] && i < current[2])
            zmin_by_cell[cell] = (zi, i)
        end
    end

    keep = [v[2] for v in values(zmin_by_cell)]
    sort!(keep)
    return keep
end

# --- Union-Find helpers for voxel_connected_component_filter_indices ---
function _uf_find!(parent::Vector{Int}, i::Int)
    @inbounds while parent[i] != i
        parent[i] = parent[parent[i]]  # path halving
        i = parent[i]
    end
    return i
end

function _uf_union!(parent::Vector{Int}, rnk::Vector{Int}, a::Int, b::Int)
    ra = _uf_find!(parent, a)
    rb = _uf_find!(parent, b)
    ra == rb && return
    if rnk[ra] < rnk[rb]; ra, rb = rb, ra; end
    parent[rb] = ra
    rnk[ra] += (rnk[ra] == rnk[rb])
end

"""
    voxel_connected_component_filter_indices(points::AbstractMatrix{<:Real}, voxel_size::Real;
                                             min_cc_size::Int=1) -> Vector{Int}

Fast voxel-occupancy connected-component filter returning indices of points in large components.

Points are first snapped to 3D voxels of side `voxel_size`. Occupied voxels are merged
into components using union-find over 26-connected neighbors. Because union-find operates
on voxels (V ≪ N) rather than individual points, this is substantially faster than the
KDTree + BFS approach for large dense clouds.

**Approximation**: connectivity is at voxel resolution. Two points in the same voxel
are always considered connected; two points in adjacent voxels may be up to
`voxel_size * √3` apart but still grouped. This is acceptable for removing small
isolated fragments but is not exact Euclidean connectivity.

# Arguments
- `points`: N×3 matrix of XYZ coordinates
- `voxel_size`: Voxel cell side length (must be > 0); controls the connectivity scale
- `min_cc_size`: Minimum component size in *points* to keep (must be >= 1, default: 1)

# Returns
- `Vector{Int}`: Indices of kept points (sorted ascending)

# Example
```julia
coords = rand(10000, 3)
indices = voxel_connected_component_filter_indices(coords, 0.5, min_cc_size=100)
filtered = coords[indices, :]
```
"""
function voxel_connected_component_filter_indices(points::AbstractMatrix{<:Real}, voxel_size::Real;
                                                  min_cc_size::Int=_CFG.voxel_cc_filter_min_cc_size)
    size(points, 2) == 3 || throw(ArgumentError("points must be N×3 matrix"))
    voxel_size > 0 || throw(ArgumentError("voxel_size must be > 0"))
    min_cc_size >= 1 || throw(ArgumentError("min_cc_size must be >= 1"))

    n = size(points, 1)
    n == 0 && return Int[]

    inv_vs = 1.0 / float(voxel_size)
    VK = NTuple{3, Int}

    # --- Step 1: Bin every point into a voxel cell ---
    voxel_of = Vector{VK}(undef, n)
    voxel_points = Dict{VK, Vector{Int}}()

    @inbounds for i in 1:n
        vk = (floor(Int, float(points[i, 1]) * inv_vs),
              floor(Int, float(points[i, 2]) * inv_vs),
              floor(Int, float(points[i, 3]) * inv_vs))
        voxel_of[i] = vk
        bucket = get!(voxel_points, vk, Int[])
        push!(bucket, i)
    end

    # --- Step 2: Assign a compact integer ID to each occupied voxel ---
    voxel_keys = collect(keys(voxel_points))
    nv = length(voxel_keys)
    voxel_id = Dict{VK, Int}()
    sizehint!(voxel_id, nv)
    for (id, vk) in enumerate(voxel_keys)
        voxel_id[vk] = id
    end

    # --- Step 3: Union adjacent occupied voxels (26-connectivity) ---
    parent = collect(1:nv)
    rnk    = zeros(Int, nv)

    for vk in voxel_keys
        va = voxel_id[vk]
        cx, cy, cz = vk
        for dz in -1:1, dy in -1:1, dx in -1:1
            (dx == 0 && dy == 0 && dz == 0) && continue
            nb = get(voxel_id, (cx + dx, cy + dy, cz + dz), 0)
            nb != 0 && _uf_union!(parent, rnk, va, nb)
        end
    end

    # --- Step 4: Sum point counts per component (keyed by root voxel ID) ---
    comp_npts = Dict{Int, Int}()
    for vk in voxel_keys
        root = _uf_find!(parent, voxel_id[vk])
        comp_npts[root] = get(comp_npts, root, 0) + length(voxel_points[vk])
    end

    # --- Step 5: Keep points whose component meets the size threshold ---
    keep = falses(n)
    @inbounds for i in 1:n
        root = _uf_find!(parent, voxel_id[voxel_of[i]])
        if get(comp_npts, root, 0) >= min_cc_size
            keep[i] = true
        end
    end

    return findall(keep)
end

"""
    upward_conic_filter_indices(points::AbstractMatrix{<:Real}, cone_theta_deg::Real;
                                max_search_delta_z::Real=5.0) -> Vector{Int}

Single-pass global upward-conic filter returning indices of kept points.

Each kept point acts as a cone anchor and suppresses higher points in its local XY
neighborhood (within `max_search_radius`) that lie inside its upward cone:
- `Δz = zj - zi > 0`
- `r_xy <= Δz * tan(cone_theta_deg)`

# Arguments
- `points`: N×3 matrix of XYZ coordinates
- `cone_theta_deg`: Cone half-angle in degrees (must satisfy 0 < angle < 90)
- `max_search_delta_z`: Vertical search span (meters) used to cap horizontal
    search radius, where `max_search_radius = max_search_delta_z * tan(cone_theta_deg)`
    (default: 5.0)

# Returns
- `Vector{Int}`: Indices of kept points (sorted ascending)

# Example
```julia
coords = rand(10000, 3)
indices = upward_conic_filter_indices(coords, 45.0)
filtered = coords[indices, :]
```
"""
function upward_conic_filter_indices(points::AbstractMatrix{<:Real}, cone_theta_deg::Real;
                                     max_search_delta_z::Real=_CFG.upward_conic_filter_max_search_delta_z)
    size(points, 2) == 3 || throw(ArgumentError("points must be N×3 matrix"))
    0 < cone_theta_deg < 90 || throw(ArgumentError("cone_theta_deg must satisfy 0 < angle < 90"))
    max_search_delta_z > 0 || throw(ArgumentError("max_search_delta_z must be > 0"))

    n = size(points, 1)
    n == 0 && return Int[]

    x = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)
    z = Vector{Float64}(undef, n)

    @inbounds for i in 1:n
        x[i] = float(points[i, 1])
        y[i] = float(points[i, 2])
        z[i] = float(points[i, 3])
    end

    tan_theta = tan(deg2rad(float(cone_theta_deg)))
    max_search_radius = float(max_search_delta_z) * tan_theta
    max_search_radius2 = max_search_radius * max_search_radius
    keep = trues(n)

    # Process points in input order and scan only local XY bins of future targets.
    # This preserves the previous suppression semantics more closely than a pure
    # candidate-against-active-anchor traversal.
    bin_size = max(max_search_radius / 2.0, eps(Float64))
    inv_bin_size = 1.0 / bin_size
    neighbor_span = max(1, ceil(Int, max_search_radius * inv_bin_size))
    bins = Dict{NTuple{2, Int}, Vector{Int}}()

    @inbounds for idx in 1:n
        cx = floor(Int, x[idx] * inv_bin_size)
        cy = floor(Int, y[idx] * inv_bin_size)
        push!(get!(bins, (cx, cy), Int[]), idx)
    end

    @inbounds for idx in 1:n
        keep[idx] || continue

        xi = x[idx]
        yi = y[idx]
        zi = z[idx]
        cx = floor(Int, xi * inv_bin_size)
        cy = floor(Int, yi * inv_bin_size)

        for dy_cell in -neighbor_span:neighbor_span
            for dx_cell in -neighbor_span:neighbor_span
                bucket = get(bins, (cx + dx_cell, cy + dy_cell), nothing)
                bucket === nothing && continue

                for target in bucket
                    target == idx && continue
                    keep[target] || continue

                    dz = z[target] - zi
                    dz > 0 || continue

                    dx = x[target] - xi
                    dy = y[target] - yi
                    r2 = dx * dx + dy * dy
                    r2 > max_search_radius2 && continue

                    max_r = dz * tan_theta
                    if r2 <= max_r * max_r
                        keep[target] = false
                    end
                end
            end
        end
    end

    return findall(keep)
end

"""
    rnn_filter_indices(points::AbstractMatrix{<:Real}, radius::Real;
                       min_rnn_size::Int=1) -> Vector{Int}

Radius-based neighbor count (RNN) filter returning indices of points with sufficient local density.

For each point, counts the number of neighbors (including itself) within `radius`.
Only points with neighbor count >= `min_rnn_size` are kept.

# Arguments
- `points`: N×3 matrix of XYZ coordinates
- `radius`: Search radius for neighbor queries (must be > 0)
- `min_rnn_size`: Minimum neighbor count to keep a point (must be >= 1, default: 1)

# Returns
- `Vector{Int}`: Indices of points with sufficient local density (sorted ascending)

# Example
```julia
coords = rand(10000, 3)
# Keep only points with at least 20 neighbors within 1.0 meter
indices = rnn_filter_indices(coords, 1.0, min_rnn_size=20)
filtered = coords[indices, :]
```
"""
function rnn_filter_indices(points::AbstractMatrix{<:Real}, radius::Real;
                            min_rnn_size::Int=_CFG.rnn_filter_min_rnn_size)
    size(points, 2) == 3 || throw(ArgumentError("points must be N×3 matrix"))
    radius > 0 || throw(ArgumentError("radius must be > 0"))
    min_rnn_size >= 1 || throw(ArgumentError("min_rnn_size must be >= 1"))

    n = size(points, 1)
    n == 0 && return Int[]

    tree = KDTree(points')
    keep = falses(n)

    @inbounds for i in 1:n
        neighbors = inrange(tree, points[i, :], radius)
        if length(neighbors) >= min_rnn_size
            keep[i] = true
        end
    end

    return findall(keep)
end

"""
    rnn_filter(pc::PointCloud, radius::Real; min_rnn_size::Int=1) -> PointCloud

Radius-based neighbor count filter for point clouds.

# Arguments
- `pc`: Input PointCloud
- `radius`: Search radius for neighbor queries (must be > 0)
- `min_rnn_size`: Minimum neighbor count to keep a point (must be >= 1, default: 1)

# Returns
- `PointCloud`: Filtered point cloud retaining only points with sufficient local density

# Example
```julia
pc = read_las("input.laz")
pc_dense = rnn_filter(pc, 1.0, min_rnn_size=20)
```
"""
function rnn_filter(pc::PointCloud, radius::Real; min_rnn_size::Int=_CFG.rnn_filter_min_rnn_size)
    indices = rnn_filter_indices(coordinates(pc), radius, min_rnn_size=min_rnn_size)
    return pc[indices]
end

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

"""
    buffer_polygon(polygon::AbstractMatrix{<:Real}, buffer::Real) -> Matrix{Float64}

Expand a convex polygon outward by `buffer` meters.

Each vertex is offset along the outward bisector of its two adjacent edges,
scaled so that every edge is displaced by exactly `buffer`.

# Arguments
- `polygon`: M×2 matrix of vertices (ordered, not closed)
- `buffer`: offset distance in meters (must be > 0)

# Returns
- `Matrix{Float64}`: M×2 matrix of buffered polygon vertices
"""
function buffer_polygon(polygon::AbstractMatrix{<:Real}, buffer::Real)
    buffer > 0 || throw(ArgumentError("buffer must be > 0"))
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
        cos_half = max(cos_half, 0.1)  # clamp to avoid extreme spikes
        offset = buffer / cos_half

        result[i, 1] = polygon[i, 1] + bx * offset
        result[i, 2] = polygon[i, 2] + by * offset
    end

    return result
end

# ── XY polygon filter ────────────────────────────────────────────

"""
    XY_polygon_filter_indices(points::AbstractMatrix{<:Real}, polygon::AbstractMatrix{<:Real}) -> Vector{Int}

Return indices of points whose XY projection falls inside a polygon.

Uses the ray-casting algorithm (horizontal ray in +X direction).

# Arguments
- `points`: N×3 (or N×2) coordinate matrix
- `polygon`: M×2 matrix of polygon vertices (ordered, closed implicitly)

# Returns
- `Vector{Int}`: Indices of points inside the polygon

# Example
```julia
poly = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
coords = rand(1000, 3)
inside = XY_polygon_filter_indices(coords, poly)
```
"""
function XY_polygon_filter_indices(points::AbstractMatrix{<:Real}, polygon::AbstractMatrix{<:Real})
    size(points, 2) >= 2 || throw(ArgumentError("points must have at least 2 columns"))
    size(polygon, 2) == 2 || throw(ArgumentError("polygon must be M×2"))

    n = size(points, 1)
    m = size(polygon, 1)
    n == 0 && return Int[]
    m >= 3 || throw(ArgumentError("polygon must have at least 3 vertices"))

    keep = falses(n)

    @inbounds for i in 1:n
        px = Float64(points[i, 1])
        py = Float64(points[i, 2])
        inside = false

        j = m
        for k in 1:m
            vy_j = Float64(polygon[j, 2])
            vy_k = Float64(polygon[k, 2])
            vx_j = Float64(polygon[j, 1])
            vx_k = Float64(polygon[k, 1])

            # Ray casting: check if horizontal ray from (px, py) in +X crosses edge j→k
            if (vy_k > py) != (vy_j > py)
                x_intersect = vx_k + (py - vy_k) / (vy_j - vy_k) * (vx_j - vx_k)
                if px < x_intersect
                    inside = !inside
                end
            end
            j = k
        end

        keep[i] = inside
    end

    return findall(keep)
end

"""
    XY_polygon_filter(pc::PointCloud, polygon::AbstractMatrix{<:Real}) -> PointCloud

Filter a point cloud to points inside a polygon in the XY plane.

# Arguments
- `pc`: Input PointCloud
- `polygon`: M×2 matrix of polygon vertices (ordered, closed implicitly)

# Returns
- `PointCloud`: Filtered point cloud
"""
function XY_polygon_filter(pc::PointCloud, polygon::AbstractMatrix{<:Real})
    indices = XY_polygon_filter_indices(coordinates(pc), polygon)
    return pc[indices]
end

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
