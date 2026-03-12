"""
Subsampling algorithms for point cloud downsampling.
"""

"""
    distance_subsample_indices(points::AbstractMatrix{<:Real}, min_dist::Real) -> Vector{Int}

Subsample points ensuring minimum distance between kept points, returning indices.

This function processes points sequentially and keeps a point only if it is at
least `min_dist` away from all previously kept points. Uses spatial grid hashing
for efficient proximity queries.

# Arguments
- `points`: N×3 matrix of XYZ coordinates
- `min_dist`: Minimum Euclidean distance between kept points (must be > 0)

# Returns
- `Vector{Int}`: Indices of points to keep

# Example
```julia
coords = rand(10000, 3)
indices = distance_subsample_indices(coords, 0.03)
filtered_coords = coords[indices, :]
```
"""
function distance_subsample_indices(points::AbstractMatrix{<:Real}, min_dist::Real)
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

"""
    distance_subsample(pc::PointCloud, min_dist::Real) -> PointCloud

Subsample a point cloud ensuring minimum distance between kept points.

# Arguments
- `pc`: Input PointCloud
- `min_dist`: Minimum Euclidean distance between kept points (must be > 0)

# Returns
- `PointCloud`: Subsampled point cloud

# Example
```julia
pc = read_las("input.laz")
pc_subsampled = distance_subsample(pc, 0.03)
```
"""
function distance_subsample(pc::PointCloud, min_dist::Real)
    indices = distance_subsample_indices(coordinates(pc), min_dist)
    return pc[indices]
end
