"""
Transformation functions for point cloud coordinate manipulation.
"""

"""
    translate(pc::PointCloud{T}, dx::Real, dy::Real, dz::Real) -> PointCloud{T}

Translate a point cloud by the given offsets.

# Arguments
- `pc`: Input PointCloud
- `dx`, `dy`, `dz`: Translation offsets in X, Y, Z directions

# Returns
- `PointCloud{T}`: Translated point cloud

# Example
```julia
pc = read_las("input.laz")
pc_translated = translate(pc, 100.0, 200.0, 0.0)
```
"""
function translate(pc::PointCloud, dx::Real, dy::Real, dz::Real)
    T = eltype(coordinates(pc))
    new_coords = copy(coordinates(pc))
    new_coords[:, 1] .+= T(dx)
    new_coords[:, 2] .+= T(dy)
    new_coords[:, 3] .+= T(dz)

    return _replace_coordinates(pc, new_coords)
end

"""
    translate!(pc::PointCloud, dx::Real, dy::Real, dz::Real) -> PointCloud

Translate a point cloud in-place by the given offsets.

# Arguments
- `pc`: PointCloud to modify
- `dx`, `dy`, `dz`: Translation offsets in X, Y, Z directions

# Returns
- `PointCloud`: The modified point cloud (same as input)

# Example
```julia
pc = read_las("input.laz")
translate!(pc, 100.0, 200.0, 0.0)
```
"""
function translate!(pc::PointCloud, dx::Real, dy::Real, dz::Real)
    return translate(pc, dx, dy, dz)
end

"""
    scale(pc::PointCloud{T}, factor::Real) -> PointCloud{T}

Uniformly scale a point cloud by the given factor around the origin.

# Arguments
- `pc`: Input PointCloud
- `factor`: Scaling factor (> 0)

# Returns
- `PointCloud{T}`: Scaled point cloud

# Example
```julia
pc = read_las("input.laz")
pc_scaled = scale(pc, 2.0)  # Double the size
```
"""
function scale(pc::PointCloud, factor::Real)
    T = eltype(coordinates(pc))
    factor > 0 || throw(ArgumentError("scale factor must be > 0"))

    new_coords = coordinates(pc) .* T(factor)
    return _replace_coordinates(pc, new_coords)
end

"""
    scale(pc::PointCloud{T}, sx::Real, sy::Real, sz::Real) -> PointCloud{T}

Non-uniformly scale a point cloud by different factors in each axis.

# Arguments
- `pc`: Input PointCloud
- `sx`, `sy`, `sz`: Scaling factors for X, Y, Z axes (all > 0)

# Returns
- `PointCloud{T}`: Scaled point cloud

# Example
```julia
pc = read_las("input.laz")
pc_scaled = scale(pc, 2.0, 2.0, 1.0)  # Double XY, keep Z
```
"""
function scale(pc::PointCloud, sx::Real, sy::Real, sz::Real)
    T = eltype(coordinates(pc))
    sx > 0 && sy > 0 && sz > 0 || throw(ArgumentError("scale factors must be > 0"))

    new_coords = copy(coordinates(pc))
    new_coords[:, 1] .*= T(sx)
    new_coords[:, 2] .*= T(sy)
    new_coords[:, 3] .*= T(sz)

    return _replace_coordinates(pc, new_coords)
end

"""
    rotate(pc::PointCloud{T}, axis, angle::Real) -> PointCloud{T}

Rotate a point cloud around a given axis by the specified angle.

# Arguments
- `pc`: Input PointCloud
- `axis`: Rotation axis as 3-element vector or Symbol (`:x`, `:y`, `:z`)
- `angle`: Rotation angle in radians

# Returns
- `PointCloud{T}`: Rotated point cloud

# Example
```julia
pc = read_las("input.laz")
# Rotate 45 degrees around Z axis
pc_rotated = rotate(pc, :z, π/4)
# Or with explicit axis
pc_rotated = rotate(pc, [0, 0, 1], π/4)
```
"""
function rotate(pc::PointCloud, axis::Symbol, angle::Real)
    # Convert symbol to rotation
    R = if axis == :x
        RotX(angle)
    elseif axis == :y
        RotY(angle)
    elseif axis == :z
        RotZ(angle)
    else
        throw(ArgumentError("axis must be :x, :y, or :z"))
    end
    
    return _apply_rotation(pc, R)
end

function rotate(pc::PointCloud, axis::AbstractVector, angle::Real)
    length(axis) == 3 || throw(ArgumentError("axis must be a 3-element vector"))
    
    # Normalize axis
    axis_norm = normalize(axis)
    
    # Create rotation from axis-angle representation
    R = AngleAxis(angle, axis_norm[1], axis_norm[2], axis_norm[3])
    
    return _apply_rotation(pc, R)
end

"""
    rotate(pc::PointCloud{T}, R::Rotation) -> PointCloud{T}

Rotate a point cloud using a rotation matrix from Rotations.jl.

# Arguments
- `pc`: Input PointCloud
- `R`: Rotation from Rotations.jl (e.g., RotMatrix, RotXYZ, etc.)

# Returns
- `PointCloud{T}`: Rotated point cloud

# Example
```julia
using Rotations
pc = read_las("input.laz")
R = RotXYZ(0.1, 0.2, 0.3)  # Euler angles
pc_rotated = rotate(pc, R)
```
"""
function rotate(pc::PointCloud, R::Rotation)
    return _apply_rotation(pc, R)
end

"""
    _apply_rotation(pc::PointCloud{T}, R::Rotation) -> PointCloud{T}

Internal function to apply a rotation to all points.
"""
function _apply_rotation(pc::PointCloud, R::Rotation)
    T = eltype(coordinates(pc))
    n = length(pc)
    new_coords = Matrix{T}(undef, n, 3)
    coords = coordinates(pc)

    # Apply rotation to each point
    @inbounds for i in 1:n
        p = SVector{3,T}(coords[i, 1], coords[i, 2], coords[i, 3])
        p_rot = R * p
        new_coords[i, 1] = p_rot[1]
        new_coords[i, 2] = p_rot[2]
        new_coords[i, 3] = p_rot[3]
    end

    return _replace_coordinates(pc, new_coords)
end

"""
    center_at_origin(pc::PointCloud) -> PointCloud

Translate point cloud so its centroid is at the origin.

# Arguments
- `pc`: Input PointCloud

# Returns
- `PointCloud`: Centered point cloud

# Example
```julia
pc = read_las("input.laz")
pc_centered = center_at_origin(pc)
```
"""
function center_at_origin(pc::PointCloud)
    c = center(pc)
    return translate(pc, -c[1], -c[2], -c[3])
end

"""
    transform(pc::PointCloud{T}, tfm) -> PointCloud{T}

Apply an arbitrary transformation from CoordinateTransformations.jl.

# Arguments
- `pc`: Input PointCloud
- `tfm`: Transformation from CoordinateTransformations.jl

# Returns
- `PointCloud{T}`: Transformed point cloud

# Example
```julia
using CoordinateTransformations, Rotations
pc = read_las("input.laz")

# Compose transformations
tfm = Translation(10, 20, 30) ∘ LinearMap(RotZ(π/4)) ∘ LinearMap(2.0I)
pc_transformed = transform(pc, tfm)
```
"""
function transform(pc::PointCloud, tfm)
    T = eltype(coordinates(pc))
    n = length(pc)
    new_coords = Matrix{T}(undef, n, 3)
    coords = coordinates(pc)

    # Apply transformation to each point
    @inbounds for i in 1:n
        p = SVector{3,T}(coords[i, 1], coords[i, 2], coords[i, 3])
        p_tfm = tfm(p)
        new_coords[i, 1] = p_tfm[1]
        new_coords[i, 2] = p_tfm[2]
        new_coords[i, 3] = p_tfm[3]
    end

    return _replace_coordinates(pc, new_coords)
end

"""
    apply_transform(coords::AbstractMatrix, tfm) -> Matrix

Apply a transformation to a coordinate matrix.

# Arguments
- `coords`: N×3 matrix of coordinates
- `tfm`: Transformation function or object

# Returns
- `Matrix`: Transformed coordinates

# Example
```julia
coords = rand(100, 3)
tfm = Translation(1, 2, 3)
new_coords = apply_transform(coords, tfm)
```
"""
function apply_transform(coords::AbstractMatrix{T}, tfm) where T
    size(coords, 2) == 3 || throw(ArgumentError("coords must be N×3 matrix"))
    
    n = size(coords, 1)
    new_coords = Matrix{T}(undef, n, 3)
    
    @inbounds for i in 1:n
        p = SVector{3,T}(coords[i, 1], coords[i, 2], coords[i, 3])
        p_tfm = tfm(p)
        new_coords[i, 1] = p_tfm[1]
        new_coords[i, 2] = p_tfm[2]
        new_coords[i, 3] = p_tfm[3]
    end
    
    return new_coords
end

"""
    apply_transform!(coords::AbstractMatrix, tfm) -> Matrix

Apply a transformation to a coordinate matrix in-place.

# Arguments
- `coords`: N×3 matrix of coordinates (modified in place)
- `tfm`: Transformation function or object

# Returns
- `Matrix`: The modified coordinate matrix (same as input)

# Example
```julia
coords = rand(100, 3)
tfm = Translation(1, 2, 3)
apply_transform!(coords, tfm)
```
"""
function apply_transform!(coords::AbstractMatrix{T}, tfm) where T
    size(coords, 2) == 3 || throw(ArgumentError("coords must be N×3 matrix"))
    
    n = size(coords, 1)
    
    @inbounds for i in 1:n
        p = SVector{3,T}(coords[i, 1], coords[i, 2], coords[i, 3])
        p_tfm = tfm(p)
        coords[i, 1] = p_tfm[1]
        coords[i, 2] = p_tfm[2]
        coords[i, 3] = p_tfm[3]
    end
    
    return coords
end

"""
    bounding_box_crop(pc::PointCloud, min_corner, max_corner) -> PointCloud

Crop point cloud to axis-aligned bounding box.

# Arguments
- `pc`: Input PointCloud
- `min_corner`: 3-element vector of minimum [x, y, z] coordinates
- `max_corner`: 3-element vector of maximum [x, y, z] coordinates

# Returns
- `PointCloud`: Cropped point cloud

# Example
```julia
pc = read_las("input.laz")
pc_cropped = bounding_box_crop(pc, [0, 0, 0], [100, 100, 50])
```
"""
function bounding_box_crop(pc::PointCloud, min_corner, max_corner)
    length(min_corner) == 3 || throw(ArgumentError("min_corner must have 3 elements"))
    length(max_corner) == 3 || throw(ArgumentError("max_corner must have 3 elements"))
    
    coords = coordinates(pc)
    n = length(pc)
    keep = Vector{Int}()
    sizehint!(keep, n)
    
    @inbounds for i in 1:n
        if (min_corner[1] <= coords[i, 1] <= max_corner[1] &&
            min_corner[2] <= coords[i, 2] <= max_corner[2] &&
            min_corner[3] <= coords[i, 3] <= max_corner[3])
            push!(keep, i)
        end
    end
    
    return pc[keep]
end
