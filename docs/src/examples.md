# Examples

## Complete Processing Pipelines

### Example 1: Forest Plot Processing

Process terrestrial laser scanning (TLS) data from a forest plot:

```julia
using FLiP

# Load full-resolution scan
pc = read_las("forest_plot.laz")
println("Loaded $(length(pc)) points")

# Subsample to 1cm resolution
pc_sub = voxel_grid_downsample(pc, 0.01)
println("After subsampling: $(length(pc_sub)) points")

# Remove statistical outliers
pc_clean = statistical_filter(pc_sub, k_neighbors=20, n_sigma=1.5)

# Separate ground and vegetation
coords = coordinates(pc_clean)
seed_idx = grid_zmin_filter_indices(coords, 0.5)
ground_local = upward_conic_filter_indices(coords[seed_idx, :], 30.0)
ground_idx = seed_idx[ground_local]
vegetation_idx = sort(setdiff(1:length(pc_clean), ground_idx))
ground = pc_clean[ground_idx]
vegetation = pc_clean[vegetation_idx]
println("Ground points: $(length(ground))")
println("Vegetation points: $(length(vegetation))")

# Filter vegetation by height (keep 0.5m to 30m)
vegetation_filtered = height_filter(vegetation, 0.5, 30.0)

# Save results
write_las("ground.laz", ground)
write_las("vegetation.laz", vegetation_filtered)
```

### Example 2: UAV Point Cloud Registration

Align UAV-based point clouds to a reference coordinate system:

```julia
using FLiP
using CoordinateTransformations, Rotations

# Load UAV point cloud
pc_uav = read_las("uav_scan.laz")

# Center the point cloud at origin for easier manipulation
pc_centered = center_at_origin(pc_uav)

# Apply rotation to align with reference (e.g., North-South)
rotation_angle = deg2rad(45)  # 45 degree correction
pc_rotated = rotate(pc_centered, :z, rotation_angle)

# Translate to reference coordinate system
pc_registered = translate(pc_rotated, 500000.0, 4500000.0, 100.0)

# Subsample for faster processing downstream
pc_final = voxel_grid_downsample(pc_registered, 0.10)  # 10cm voxels

write_las("uav_registered.laz", pc_final)
```

### Example 3: Multi-Scan Merge and Clean

Merge multiple scans and clean the combined data:

```julia
using FLiP

# Load multiple scans
scan_files = ["scan1.laz", "scan2.laz", "scan3.laz"]
scans = [read_las(f) for f in scan_files]

# Merge by concatenating coordinates
all_coords = vcat([coordinates(pc) for pc in scans]...)
merged_pc = PointCloud(all_coords)

println("Merged cloud: $(length(merged_pc)) points")

# Remove overlapping points using voxel grid
pc_unique = voxel_grid_downsample(merged_pc, 0.005)  # 5mm voxels
println("After deduplication: $(length(pc_unique)) points")

# Remove outliers that may come from scan registration errors
pc_clean = statistical_filter(pc_unique, k_neighbors=15, n_sigma=2.0)
println("After cleaning: $(length(pc_clean)) points")

# Save merged and cleaned result
write_las("merged_clean.laz", pc_clean)
```

### Example 4: Point Cloud Quality Assessment

Analyze and filter based on point density:

```julia
using FLiP
using NearestNeighbors

# Load point cloud
pc = read_las("input.laz")

# Compute local point density
coords = coordinates(pc)
tree = KDTree(coords')
densities = zeros(length(pc))

for i in 1:length(pc)
    # Count neighbors within 0.1m radius
    idxs = inrange(tree, coords[i, :], 0.1)
    densities[i] = length(idxs) - 1  # Exclude the point itself
end

# Add density as attribute
setattribute!(pc, :density, densities)

# Filter low-density regions (potential noise)
high_density_indices = findall(d -> d >= 10, densities)
pc_filtered = pc[high_density_indices]

println("Removed $(length(pc) - length(pc_filtered)) low-density points")

write_las("high_density.laz", pc_filtered)
```

### Example 5: Terrain Analysis

Extract and analyze terrain from aerial LiDAR:

```julia
using FLiP

# Load aerial LiDAR data
pc = read_las("aerial_lidar.laz")

# Aggressive subsampling for terrain (aerial data is dense)
pc_sub = voxel_grid_downsample(pc, 0.5)  # 50cm voxels

# Extract ground points
coords = coordinates(pc_sub)
seed_idx = grid_zmin_filter_indices(coords, 2.0)
ground_local = upward_conic_filter_indices(coords[seed_idx, :], 15.0)
ground_idx = seed_idx[ground_local]
non_ground_idx = sort(setdiff(1:length(pc_sub), ground_idx))
ground = pc_sub[ground_idx]
non_ground = pc_sub[non_ground_idx]
println("Ground points: $(length(ground))")

# Further clean ground points
ground_clean = statistical_filter(ground, k_neighbors=8, n_sigma=2.5)

# Compute bounding box
bbox = bounds(ground_clean)
println("Terrain extent:")
println("  X: $(bbox[1]) to $(bbox[2])")
println("  Y: $(bbox[3]) to $(bbox[4])")
println("  Z: $(bbox[5]) to $(bbox[6])")

# Crop to area of interest
ground_aoi = bounding_box_crop(ground_clean, 
                                [500, 500, bbox[5]], 
                                [1500, 1500, bbox[6]])

write_las("terrain.laz", ground_aoi)
```

### Example 6: Custom Transformation Pipeline

Apply complex transformations for data integration:

```julia
using FLiP
using CoordinateTransformations, Rotations, LinearAlgebra

# Load point cloud in local coordinate system
pc_local = read_las("local_scan.laz")

# Define transformation: rotate, scale, then translate
R = RotXYZ(deg2rad(5), deg2rad(-2), deg2rad(45))  # Euler angles
S = 1.05  # 5% scaling correction
T = [1000.0, 2000.0, 50.0]  # Translation vector

# Create composite transformation
transform_composite = Translation(T...) ∘ LinearMap(S * I) ∘ LinearMap(R)

# Apply transformation
pc_transformed = transform(pc_local, transform_composite)

# Crop to region of interest after transformation
pc_cropped = bounding_box_crop(pc_transformed,
                                [0, 0, 0],
                                [3000, 3000, 100])

# Final quality control
pc_final = statistical_filter(pc_cropped, k_neighbors=10, n_sigma=2.0)

write_las("integrated.laz", pc_final)
```

### Example 7: Format Conversion with Processing

Convert between formats while applying processing:

```julia
using FLiP

# Read PTX file from Leica scanner
pc = read_ptx("leica_scan.ptx")
println("Loaded PTX with $(length(pc)) points")

# Check for RGB data
if hasattribute(pc, :r)
    println("Point cloud has RGB data")
end

# Subsample (PTX files are often very dense)
pc_sub = distance_subsample(pc, 0.02)  # 2cm minimum distance

# Remove isolated points
pc_filtered = radius_filter(pc_sub, radius=0.05, min_neighbors=3)

# Save as compressed LAZ (more efficient storage)
write_las("converted.laz", pc_filtered)

# Also save as PCD for use with other tools
write_pcd("converted.pcd", pc_filtered)
```

### Example 8: Batch Processing

Process multiple files in a directory:

```julia
using FLiP

function process_scan(input_path, output_dir)
    # Load
    pc = read_las(input_path)
    
    # Process
    pc = voxel_grid_downsample(pc, 0.05)
    pc = statistical_filter(pc, 10, 2.0)
    coords = coordinates(pc)
    seed_idx = grid_zmin_filter_indices(coords, 1.0)
    ground_local = upward_conic_filter_indices(coords[seed_idx, :], 45.0)
    ground_idx = seed_idx[ground_local]
    vegetation_idx = sort(setdiff(1:length(pc), ground_idx))
    ground = pc[ground_idx]
    vegetation = pc[vegetation_idx]
    
    # Save
    basename_no_ext = splitext(basename(input_path))[1]
    write_las(joinpath(output_dir, "$(basename_no_ext)_ground.laz"), ground)
    write_las(joinpath(output_dir, "$(basename_no_ext)_vegetation.laz"), vegetation)
    
    return length(ground), length(vegetation)
end

# Process all files
input_dir = "raw_scans"
output_dir = "processed_scans"
mkpath(output_dir)

for file in readdir(input_dir)
    if endswith(file, ".laz") || endswith(file, ".las")
        println("Processing $file...")
        input_path = joinpath(input_dir, file)
        n_ground, n_veg = process_scan(input_path, output_dir)
        println("  Ground: $n_ground, Vegetation: $n_veg")
    end
end
```

## Tips and Tricks

### Efficient Memory Usage

```julia
# Use index functions for multi-step filtering
coords = coordinates(pc)
idx1 = voxel_grid_downsample_indices(coords, 0.05)
idx2 = statistical_filter_indices(coords[idx1, :], 10, 2.0)
final_indices = idx1[idx2]
pc_filtered = pc[final_indices]
```

### Progressive Subsampling

```julia
# Coarse subsample first, then fine-tune
pc_coarse = voxel_grid_downsample(pc, 0.1)      # Fast, remove bulk
pc_medium = voxel_grid_downsample(pc_coarse, 0.05)  # Refine
pc_final = statistical_filter(pc_medium, 10, 2.0)   # Clean
```

### Preserve Original Data

```julia
# Original stays unchanged
pc_original = read_las("important_scan.laz")

# Clone for processing
coords_copy = copy(coordinates(pc_original))
pc_working = PointCloud(coords_copy)

# Process the copy
pc_processed = voxel_grid_downsample(pc_working, 0.05)

# Original is still available
```
