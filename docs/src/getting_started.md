# Getting Started

## Installation

FLiP.jl requires Julia 1.9 or later. Install it using Julia's package manager:

```julia
using Pkg
Pkg.add("FLiP")
```

For development or to use the latest features:

```julia
using Pkg
Pkg.develop(url="https://github.com/xiangtaoxu/FLiP.jl")
```

## Basic Workflow

A typical point cloud processing workflow with FLiP.jl involves:

1. **Loading data**: Read point clouds from files
2. **Preprocessing**: Subsample or filter to reduce noise
3. **Processing**: Apply transformations or extract features
4. **Saving results**: Write processed data back to files

### Example: Basic Processing Pipeline

```julia
using FLiP

# 1. Load point cloud
pc = read_las("input.laz")
println("Loaded $(length(pc)) points")

# 2. Subsample to reduce data size
pc_sub = voxel_grid_downsample(pc, 0.05)  # 5cm voxel size
println("After subsampling: $(length(pc_sub)) points")

# 3. Remove noise
pc_clean = statistical_filter(pc_sub, k_neighbors=10, n_sigma=2.0)
println("After filtering: $(length(pc_clean)) points")

# 4. Save result
write_las("output.laz", pc_clean)
println("Saved to output.laz")
```

## The PointCloud Type

FLiP.jl uses the `PointCloud{T}` type to represent 3D point data:

```julia
# Create from coordinates
coords = rand(1000, 3)
pc = PointCloud(coords)

# Add attributes
pc_with_intensity = PointCloud(coords, 
    Dict(:intensity => rand(1000)))

# Access properties
n = length(pc)              # Number of points
xyz = coordinates(pc)       # Get coordinate matrix
bbox = bounds(pc)           # Bounding box
centroid = center(pc)       # Geometric center

# Index and slice
point = pc[100]             # Get 100th point
subset = pc[1:10]           # First 10 points
```

## Supported Formats

FLiP.jl currently supports:

- **LAS/LAZ**: Industry-standard LiDAR format (via PointClouds.jl)
- **PCD**: Point Cloud Data format (ASCII mode)
- **PTX**: Leica Cyclone PTX format

```julia
# Read different formats
pc_las = read_las("scan.laz")
pc_pcd = read_pcd("scan.pcd")
pc_ptx = read_ptx("scan.ptx")

# Write
write_las("output.laz", pc)
write_pcd("output.pcd", pc)
```

## Key Operations

### Subsampling

Reduce point cloud density while preserving structure:

```julia
# Voxel grid downsampling (fastest)
pc_voxel = voxel_grid_downsample(pc, 0.05)

# Minimum distance subsampling (better distribution)
pc_dist = distance_subsample(pc, 0.03)
```

### Filtering

Remove outliers and noise:

```julia
# Statistical outlier removal
pc_clean = statistical_filter(pc, k_neighbors=10, n_sigma=2.0)

# Radius filtering
pc_clean2 = radius_filter(pc, radius=0.05, min_neighbors=5)

# Ground filtering with index utilities
coords = coordinates(pc)
ground_seed_idx = grid_zmin_filter_indices(coords, 1.0)
ground_seed = coords[ground_seed_idx, :]
ground_idx_local = upward_conic_filter_indices(ground_seed, 45.0)
ground_idx = ground_seed_idx[ground_idx_local]
ground = pc[ground_idx]
vegetation_idx = sort(setdiff(1:length(pc), ground_idx))
vegetation = pc[vegetation_idx]

# Height filtering
pc_subset = height_filter(pc, min_z=0.5, max_z=20.0)
```

### Transformations

Manipulate point cloud coordinates:

```julia
# Translation
pc_moved = translate(pc, 100.0, 200.0, 0.0)

# Rotation (45° around Z-axis)
pc_rotated = rotate(pc, :z, π/4)

# Scaling
pc_scaled = scale(pc, 2.0)  # Double size

# Center at origin
pc_centered = center_at_origin(pc)

# Crop to bounding box
pc_cropped = bounding_box_crop(pc, [0, 0, 0], [100, 100, 50])
```

## Working with Attributes

Point clouds can have additional per-point attributes:

```julia
# Check for attributes
if hasattribute(pc, :intensity)
    intensity = getattribute(pc, :intensity)
end

# Add new attribute
labels = classify_points(pc)  # Your classification function
setattribute!(pc, :label, labels)

# Attributes are preserved through operations
pc_sub = voxel_grid_downsample(pc, 0.05)
# pc_sub still has the :label attribute
```

## Performance Tips

1. **Choose appropriate voxel size**: Larger voxels = faster processing but less detail
2. **Subsample before filtering**: Reduce data size first for faster filtering
3. **Use index functions for custom workflows**: 
   ```julia
   indices = voxel_grid_downsample_indices(coords, 0.05)
   # Apply indices to multiple arrays efficiently
   ```
4. **Process in chunks for very large datasets**: Load and process subsets iteratively

## Next Steps

- Explore the [User Guide](guide/types.md) for detailed information
- Check out [Examples](examples.md) for complete workflows
- Read the [API Reference](api.md) for all available functions
