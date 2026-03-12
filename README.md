# FLiP.jl - Forest Lidar Processing

[![CI](https://github.com/xiangtaoxu/FLiP.jl/workflows/CI/badge.svg)](https://github.com/xiangtaoxu/FLiP.jl/actions)
[![Coverage](https://codecov.io/gh/xiangtaoxu/FLiP.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/xiangtaoxu/FLiP.jl)

A high-performance Julia package for processing 3D point cloud data from LiDAR and other sensors.

## Features

- **Format Support**: Read and write LAS and LAZ files
- **Efficient Subsampling**: Voxel grid and minimum distance algorithms
- **Noise Filtering**: Statistical outlier removal and radius-based filtering
- **Ground Filtering**: Grid z-min selection and upward conic suppression
- **Transformations**: Translation, rotation, scaling, and arbitrary affine transforms
- **CRS Support**: Coordinate reference system handling and reprojection

## Installation

```julia
using Pkg
Pkg.add("FLiP")
```

Or for development:

```julia
using Pkg
Pkg.develop(url="https://github.com/xiangtaoxu/FLiP.jl")
```

## Quick Start

```julia
using FLiP

# Read a point cloud
pc = read_las("input.laz")

# Subsample using voxel grid (5cm voxels)
pc_sub = voxel_grid_downsample(pc, 0.05)

# Remove statistical outliers
pc_clean = statistical_filter(pc_sub, k_neighbors=10, n_sigma=2.0)

# Apply transformation
pc_transformed = translate(pc_clean, 100.0, 200.0, 0.0)

# Save result
write_las("output.laz", pc_transformed)
```

## Core Functionality

### I/O Operations

```julia
# LAS/LAZ format
pc = read_las("file.las")
write_las("output.laz", pc)


```

### Subsampling

```julia
# Voxel grid downsampling
pc_voxel = voxel_grid_downsample(pc, voxel_size=0.05)

# Minimum distance subsampling
pc_dist = distance_subsample(pc, min_distance=0.03)

# Get indices only (for custom processing)
indices = voxel_grid_downsample_indices(coordinates(pc), 0.05)
pc_filtered = pc[indices]
```

### Filtering

```julia
# Statistical outlier removal
pc_clean = statistical_filter(pc, k_neighbors=10, n_sigma=2.0)

# Radius outlier removal
pc_clean = radius_filter(pc, radius=0.05, min_neighbors=5)

# Grid + cone ground filtering (index-based)
coords = coordinates(pc)
seed_idx = grid_zmin_filter_indices(coords, 1.0)
ground_local = upward_conic_filter_indices(coords[seed_idx, :], 45.0)
ground_idx = seed_idx[ground_local]
nonground_idx = sort(setdiff(1:length(pc), ground_idx))
ground = pc[ground_idx]
nonground = pc[nonground_idx]
```

### Transformations

```julia
# Translation
pc_translated = translate(pc, 10.0, 20.0, 5.0)

# Rotation (around Z-axis by 45 degrees)
pc_rotated = rotate(pc, [0, 0, 1], π/4)

# Scaling
pc_scaled = scale(pc, 2.0)  # uniform scaling
pc_scaled = scale(pc, 2.0, 2.0, 1.0)  # non-uniform

# Arbitrary affine transformation
using CoordinateTransformations
tfm = Translation(10, 20, 30) ∘ LinearMap(RotZ(π/4))
pc_transformed = transform(pc, tfm)
```

## Data Structure

The `PointCloud{T}` type stores 3D coordinates and optional attributes:

```julia
# Create from coordinates
coords = rand(Float64, 1000, 3)  # N×3 matrix
pc = PointCloud(coords)

# Add attributes
pc_with_attrs = PointCloud(coords, 
    Dict(:intensity => rand(1000),
         :label => rand(1:5, 1000)))

# Access properties
n_points = length(pc)
coords = coordinates(pc)
bbox = bounds(pc)
centroid = center(pc)

# Indexing
point = pc[100]  # Get 100th point
subset = pc[1:10]  # Get first 10 points
```

## Performance

FLiP.jl is designed for high performance on large point clouds:

- Efficient spatial indexing using NearestNeighbors.jl
- Memory-efficient voxel grid hashing
- Type-stable implementations
- Optional parallel processing for filtering operations

Typical performance on a modern laptop:
- Process 1M points in <2 seconds
- Voxel grid downsampling: ~0.5s for 10M points
- Statistical filtering: ~1-2s for 1M points

## Dependencies

- [PointClouds.jl](https://github.com/fugro-analytics/PointClouds.jl) - LAS/LAZ I/O
- [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl) - Spatial queries
- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) - Efficient fixed-size arrays
- [CoordinateTransformations.jl](https://github.com/JuliaGeometry/CoordinateTransformations.jl) - Transformations
- [Rotations.jl](https://github.com/JuliaGeometry/Rotations.jl) - Rotation representations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use FLiP.jl in your research, please cite:

```bibtex
@software{flip_jl,
  author = {Xu, Xiangtao},
  title = {FLiP.jl: Forest Lidar Processing in Julia},
  year = {2026},
  url = {https://github.com/xiangtaoxu/FLiP.jl}
}
```

## Related Projects

- [ForestLidarPackage](https://github.com/xiangtaoxu/ForestLidarPackage) - Python package for forest point cloud processing
- [PointClouds.jl](https://github.com/fugro-analytics/PointClouds.jl) - Julia point cloud library
- [CloudCompare](https://www.cloudcompare.org/) - 3D point cloud processing software
