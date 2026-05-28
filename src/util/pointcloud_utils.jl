"""
Point-cloud filtering, subsampling, and connected-component labelling.
Most functions accept an N×3 coordinate matrix and return `Vector{Int}`
indices of points to keep; `connected_component_labels` instead returns
a per-point label vector.

Functions:
- `distance_subsample(points, min_dist)`                       — keep points at least `min_dist` apart
- `statistical_filter(points, k, n_sigma)`                     — drop K-NN outliers beyond mean + n·σ
- `grid_zmin_filter(points, grid_size)`                        — keep min-z point per XY grid cell
- `voxel_connected_component_filter(points, voxel_size; min_cc_size)`
                                                               — drop voxel components smaller than `min_cc_size`
- `upward_conic_filter(points, cone_theta_deg; max_search_delta_z)`
                                                               — drop points inside upward cones from kept points
- `XY_polygon_filter(points, polygon)`                         — keep points inside XY polygon
- `connected_component_labels(points, max_dist; min_cc_size=1)`
                                                               — label exact Euclidean connected components
"""

# ── Subsampling ───────────────────────────────────────────────────

"""
    distance_subsample(points::AbstractMatrix{<:Real}, min_dist::Real) -> Vector{Int}

Subsample points ensuring minimum distance between kept points, returning indices.

Sequential pass: a point is kept only if it is at least `min_dist` away from
all previously kept points. Uses spatial grid hashing for efficient
proximity queries.

# Arguments
- `points`: N×3 matrix of XYZ coordinates
- `min_dist`: Minimum Euclidean distance between kept points (must be > 0)

# Returns
- `Vector{Int}`: Indices of points to keep
"""
function distance_subsample(points::AbstractMatrix{<:Real}, min_dist::Real)
    min_dist > 0 || throw(ArgumentError("min_dist must be > 0"))
    size(points, 2) == 3 || throw(ArgumentError("points must be N×3 matrix"))

    n = size(points, 1)
    inv_d = 1.0 / float(min_dist)
    d2 = float(min_dist)^2

    grid = Dict{NTuple{3, Int}, Vector{Int}}()
    keep = Int[]
    sizehint!(keep, n ÷ 2)  # Estimate to reduce allocations

    @inbounds for i in 1:n
        x = float(points[i, 1])
        y = float(points[i, 2])
        z = float(points[i, 3])

        cx = floor(Int, x * inv_d)
        cy = floor(Int, y * inv_d)
        cz = floor(Int, z * inv_d)

        accept = true

        # Check neighboring grid cells for nearby points
        for dx in -1:1, dy in -1:1, dz in -1:1
            bucket = get(grid, (cx + dx, cy + dy, cz + dz), nothing)
            bucket === nothing && continue

            for j in bucket
                ddx = x - float(points[j, 1])
                ddy = y - float(points[j, 2])
                ddz = z - float(points[j, 3])
                if ddx * ddx + ddy * ddy + ddz * ddz < d2
                    accept = false
                    break
                end
            end
            accept || break
        end

        if accept
            push!(keep, i)
            key = (cx, cy, cz)
            if haskey(grid, key)
                push!(grid[key], i)
            else
                grid[key] = [i]
            end
        end
    end

    return keep
end

# ── Statistical filtering ─────────────────────────────────────────

"""
    statistical_filter(points::AbstractMatrix{<:Real}, k_neighbors::Int=6,
                       n_sigma::Real=1) -> Vector{Int}

Statistical outlier removal filter returning indices of inlier points.

For each point, compute the mean distance to its K nearest neighbors. Points
whose mean distance exceeds `mean + n_sigma * std` across all points are
considered outliers.

# Arguments
- `points`: N×3 matrix of XYZ coordinates
- `k_neighbors`: Number of nearest neighbors to consider (default: from config)
- `n_sigma`: Number of standard deviations for threshold (default: from config)

# Returns
- `Vector{Int}`: Indices of inlier points
"""
function statistical_filter(points::AbstractMatrix{<:Real},
                            k_neighbors::Int=_CFG.statistical_filter.k_neighbors,
                            n_sigma::Real=_CFG.statistical_filter.n_sigma;
                            n_thread::Integer=effective_nthreads())
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

    # Compute mean distance to k nearest neighbors for each point.
    # Per-iteration SVector keeps the query stack-allocated; inline sum avoids
    # allocating `dists[2:end]`. Per-point KNN (rather than batch) keeps peak
    # memory bounded — batching allocates O(N·k) result vectors.
    # Embarrassingly parallel: `tree` is read-only and `mean_dists[i]` slots
    # are disjoint per iteration.
    mean_dists = Vector{Float64}(undef, n)

    _parallel_for(n, n_thread) do i
        @inbounds begin
            q = SVector(points[i, 1], points[i, 2], points[i, 3])
            _, dists = knn(tree, q, k_neighbors + 1)
            s = 0.0
            for j in 2:length(dists)  # skip dists[1] (the point itself)
                s += dists[j]
            end
            mean_dists[i] = s / (length(dists) - 1)
        end
    end

    # Compute statistics
    μ = mean(mean_dists)
    σ = std(mean_dists)
    threshold = μ + n_sigma * σ

    # Filter points below threshold
    keep = Int[]
    sizehint!(keep, n)
    @inbounds for i in 1:n
        mean_dists[i] <= threshold && push!(keep, i)
    end
    return keep
end

# ── Spatial-grid filtering ────────────────────────────────────────

"""
    grid_zmin_filter(points::AbstractMatrix{<:Real}, grid_size::Real) -> Vector{Int}

Grid minimum-z filter returning one index per XY cell.

The XY plane is partitioned into square cells of size `grid_size`, and only
the point with minimum z value in each cell is kept. Ties in z are broken
by minimum point index, so the output is deterministic across runs on the
same input.

# Arguments
- `points`: N×3 matrix of XYZ coordinates
- `grid_size`: XY grid size (must be > 0)

# Returns
- `Vector{Int}`: Indices of kept points (sorted ascending)
"""
function grid_zmin_filter(points::AbstractMatrix{<:Real}, grid_size::Real)
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

"""
    voxel_connected_component_filter(points::AbstractMatrix{<:Real}, voxel_size::Real;
                                     min_cc_size::Int=1) -> Vector{Int}

Fast voxel-occupancy connected-component filter returning indices of points
in large components.

Points are snapped to 3D voxels of side `voxel_size`. Occupied voxels are
merged into components using union-find over 26-connected neighbors. Because
union-find operates on voxels (V ≪ N) rather than individual points, this is
substantially faster than the KDTree + BFS approach for large dense clouds.

**Approximation**: connectivity is at voxel resolution. Two points in the
same voxel are always considered connected; two points in adjacent voxels
may be up to `voxel_size * √3` apart but still grouped. This is acceptable
for removing small isolated fragments but is not exact Euclidean connectivity.

# Arguments
- `points`: N×3 matrix of XYZ coordinates
- `voxel_size`: Voxel cell side length (must be > 0); controls the connectivity scale
- `min_cc_size`: Minimum component size in *points* to keep (must be >= 1)

# Returns
- `Vector{Int}`: Indices of kept points (sorted ascending)
"""
function voxel_connected_component_filter(points::AbstractMatrix{<:Real}, voxel_size::Real;
                                          min_cc_size::Int=1)
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
    ranks  = zeros(Int, nv)

    for vk in voxel_keys
        va = voxel_id[vk]
        cx, cy, cz = vk
        for dz in -1:1, dy in -1:1, dx in -1:1
            (dx == 0 && dy == 0 && dz == 0) && continue
            nb = get(voxel_id, (cx + dx, cy + dy, cz + dz), 0)
            nb != 0 && _uf_union!(parent, ranks, va, nb)
        end
    end

    # --- Step 4: Sum point counts per component (keyed by root voxel ID) ---
    comp_npts = Dict{Int, Int}()
    for vk in voxel_keys
        root = _uf_find!(parent, voxel_id[vk])
        comp_npts[root] = get(comp_npts, root, 0) + length(voxel_points[vk])
    end

    # --- Step 5: Keep points whose component meets the size threshold ---
    keep = Int[]
    sizehint!(keep, n)
    @inbounds for i in 1:n
        root = _uf_find!(parent, voxel_id[voxel_of[i]])
        if get(comp_npts, root, 0) >= min_cc_size
            push!(keep, i)
        end
    end

    return keep
end

# ── Cone-based filtering ──────────────────────────────────────────

"""
    upward_conic_filter(points::AbstractMatrix{<:Real}, cone_theta_deg::Real;
                        max_search_delta_z::Real=5.0) -> Vector{Int}

Single-pass global upward-conic filter returning indices of kept points.

Each kept point acts as a cone anchor and suppresses higher points in its
local XY neighborhood (within `max_search_radius`) that lie inside its
upward cone:
- `Δz = zj - zi > 0`
- `r_xy <= Δz * tan(cone_theta_deg)`

**Order-dependent**: results depend on input row order because the first
point in each region wins as the anchor. LAS files may be in arbitrary
scan order; for fully reproducible output, sort points by z first (which
also tends to give better-quality ground anchors).

# Arguments
- `points`: N×3 matrix of XYZ coordinates
- `cone_theta_deg`: Cone half-angle in degrees (must satisfy 0 < angle < 90)
- `max_search_delta_z`: Vertical search span (meters) used to cap horizontal
    search radius, where `max_search_radius = max_search_delta_z * tan(cone_theta_deg)`

# Returns
- `Vector{Int}`: Indices of kept points (sorted ascending)
"""
function upward_conic_filter(points::AbstractMatrix{<:Real}, cone_theta_deg::Real;
                             max_search_delta_z::Real=5.0)
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
    # BitVector is load-bearing here: the inner loop reads `keep[target]` to
    # short-circuit already-suppressed points. Do not replace with Vector{Int}.
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

# ── Polygon filtering ─────────────────────────────────────────────

"""
    XY_polygon_filter(points::AbstractMatrix{<:Real}, polygon::AbstractMatrix{<:Real}) -> Vector{Int}

Return indices of points whose XY projection falls inside a polygon.

Uses the ray-casting algorithm (horizontal ray in +X direction).

# Arguments
- `points`: N×3 (or N×2) coordinate matrix
- `polygon`: M×2 matrix of polygon vertices (ordered, closed implicitly)

# Returns
- `Vector{Int}`: Indices of points inside the polygon
"""
function XY_polygon_filter(points::AbstractMatrix{<:Real}, polygon::AbstractMatrix{<:Real})
    size(points, 2) >= 2 || throw(ArgumentError("points must have at least 2 columns"))
    size(polygon, 2) == 2 || throw(ArgumentError("polygon must be M×2"))

    n = size(points, 1)
    m = size(polygon, 1)
    n == 0 && return Int[]
    m >= 3 || throw(ArgumentError("polygon must have at least 3 vertices"))

    keep = Int[]
    sizehint!(keep, n)

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

        inside && push!(keep, i)
    end

    return keep
end

# ── Connected-component labelling ─────────────────────────────────

"""
    connected_component_labels(points::AbstractMatrix{<:Real}, max_distance::Real,
                               min_cc_size::Integer=1) -> Vector{Int}

Label exact Euclidean connected components in a point cloud.

Two points are connected when their Euclidean distance is ≤ `max_distance`.
Connectivity is computed exactly using KDTree radius queries plus
union-find over points, without materializing a full graph.

# Arguments
- `points`: N×3 matrix of XYZ coordinates
- `max_distance`: Maximum Euclidean distance for connectivity (must be > 0)
- `min_cc_size`: Minimum component size to keep. Components with fewer
    points are assigned label `0` (must be >= 1)

# Returns
- `Vector{Int}`: Contiguous component labels in `1:k` for retained
    components, where label `1` is the largest retained component and
    labels increase with decreasing component size. Components below
    `min_cc_size` receive `0`.
"""
function connected_component_labels(points::AbstractMatrix{<:Real},
                                    max_distance::Real,
                                    min_cc_size::Integer=1)
    size(points, 2) == 3 || throw(ArgumentError("points must be N×3 matrix"))
    max_distance > 0 || throw(ArgumentError("max_distance must be > 0"))
    min_cc_size >= 1 || throw(ArgumentError("min_cc_size must be >= 1"))

    n = size(points, 1)
    n == 0 && return Int[]
    n == 1 && return (min_cc_size == 1 ? [1] : [0])

    T = eltype(points)
    tree = KDTree(Matrix{T}(transpose(points)))
    radius = float(max_distance)
    parent = collect(1:n)
    ranks  = zeros(Int, n)
    nbr_buf = sizehint!(Int[], 64)

    @inbounds for i in 1:n
        q = SVector(points[i, 1], points[i, 2], points[i, 3])
        empty!(nbr_buf)
        inrange!(nbr_buf, tree, q, radius)
        for j in nbr_buf
            j > i || continue
            _uf_union!(parent, ranks, i, j)
        end
    end

    # Collapse paths so parent[i] == root(i) for every i, then map roots
    # to contiguous frequency-rank labels (≥ min_cc_size; else 0).
    @inbounds for i in eachindex(parent)
        parent[i] = _uf_find!(parent, i)
    end
    return relabel_by_occurrence(parent, Int(min_cc_size))
end
