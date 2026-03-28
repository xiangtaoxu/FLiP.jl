# FLiP.jl - Forest Lidar Processing

[![CI](https://github.com/xiangtaoxu/FLiP.jl/workflows/CI/badge.svg)](https://github.com/xiangtaoxu/FLiP.jl/actions)
[![Coverage](https://codecov.io/gh/xiangtaoxu/FLiP.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/xiangtaoxu/FLiP.jl)

A high-performance Julia package for processing 3D point cloud data from LiDAR and other sensors.

## Features

- **Format Support**: Read and write LAS, LAZ, and E57 files with auto-dispatch via `read_pc`/`write_pc`
- **Subsampling**: Minimum-distance subsampling with spatial grid hashing
- **Noise Filtering**: Statistical outlier removal, radius-neighbor filtering, and voxel connected-component filtering
- **Ground Segmentation**: Two-stage pipeline (grid z-min + upward conic filter) with above-ground height calculation
- **Tree Segmentation**: Non-Branching Segments (NBS) and Longest Connected Segments (LCS) algorithms for individual tree extraction
- **Graph Algorithms**: Radius graphs, connected components, quotient graphs, and shortest-path slicing
- **Mesh Operations**: Delaunay triangulation and cloud-to-mesh distance computation
- **Transformations**: Translation, rotation, scaling, arbitrary affine transforms, and bounding box crop
- **Pipeline**: TOML-configurable end-to-end processing via `run_pipeline`

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
pc = read_laz("input.laz")

# Subsample using minimum distance (5cm)
pc_sub = distance_subsample(pc, 0.05)

# Remove statistical outliers
pc_clean = statistical_filter(pc_sub, 10, 2.0)

# Apply transformation
pc_transformed = translate(pc_clean, 100.0, 200.0, 0.0)

# Save result
write_laz("output.laz", pc_transformed)
```

## Core Functionality

### I/O Operations

```julia
# LAS/LAZ format
pc = read_las("file.las")
pc = read_laz("file.laz")
write_las("output.las", pc)
write_laz("output.laz", pc)

# E57 format
pc = read_e57("scan.e57")
write_e57("output.e57", pc)

# Auto-dispatch by file extension
pc = read_pc("file.laz")
write_pc("output.e57", pc)

# Read metadata without loading point data
meta = read_laz_metadata("file.laz")
```

### Subsampling

```julia
# Minimum distance subsampling
pc_sub = distance_subsample(pc, 0.03)

# Get indices only (for custom processing)
indices = distance_subsample_indices(coordinates(pc), 0.03)
pc_sub = pc[indices]
```

### Filtering

```julia
# Statistical outlier removal
pc_clean = statistical_filter(pc, 10, 2.0)

# Radius-neighbor noise filter
pc_clean = rnn_filter(pc, 0.05, min_rnn_size=5)

# Low-level ground filtering (index-based)
coords = coordinates(pc)
seed_idx = grid_zmin_filter_indices(coords, 1.0)
ground_local = upward_conic_filter_indices(coords[seed_idx, :], 45.0)
ground_idx = seed_idx[ground_local]
nonground_idx = sort(setdiff(1:npoints(pc), ground_idx))
```

### Transformations

```julia
# Translation
pc_translated = translate(pc, 10.0, 20.0, 5.0)

# Rotation (axis-angle or symbol shorthand)
pc_rotated = rotate(pc, [0, 0, 1], π/4)
pc_rotated = rotate(pc, :z, π/4)

# Scaling
pc_scaled = scale(pc, 2.0)              # uniform
pc_scaled = scale(pc, 2.0, 2.0, 1.0)   # non-uniform

# Arbitrary affine transformation
using CoordinateTransformations
tfm = Translation(10, 20, 30) ∘ LinearMap(RotZ(π/4))
pc_transformed = transform(pc, tfm)

# Bounding box crop
pc_cropped = bounding_box_crop(pc, [0, 0, 0], [10, 10, 10])
```

### Ground Segmentation

```julia
# Two-stage ground segmentation + above-ground height
result = ground_segmentation(pc)
# result.ground_points  — ground point cloud
# result.agh_cloud      — input cloud with :AGH attribute added
# result.ground_area    — area of ground polygon (m²)

# Or use the lower-level API
ground_pc = segment_ground(pc, grid_size=1.0, cone_theta_deg=45.0)
agh = calculate_aboveground_height(pc, ground_pc, xy_resolution=0.5)
```

### Tree Segmentation

```julia
# Requires :AGH attribute (from ground_segmentation)
result_pc = tree_segmentation(result.agh_cloud)
# Returns PointCloud with :tree_id and :segment_id attributes

# Create skeleton point cloud from segmentation result
skeleton_pc = create_skeleton_cloud(result_pc)
```

### Pipeline

```julia
# Run the full processing pipeline from a TOML config file
run_pipeline("my_config.toml")
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
n = npoints(pc)
coords = coordinates(pc)
bbox = bounds(pc)
centroid = center(pc)

# Indexing
subset = pc[1:10]  # Get first 10 points
subset = pc[indices]  # Index with integer vector
```

## Performance

FLiP.jl is designed for high performance on large point clouds:

- Efficient spatial indexing using NearestNeighbors.jl (KD-trees)
- Pre-allocated workspace structs for repeated graph operations
- Type-stable implementations throughout
- Index-based operations to minimize memory allocations

## Dependencies

- [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl) - Spatial queries (KD-trees)
- [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl) - Graph algorithms
- [DelaunayTriangulation.jl](https://github.com/DanielVandH/DelaunayTriangulation.jl) - Mesh generation
- [MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl) - PCA for linearity analysis
- [CoordinateTransformations.jl](https://github.com/JuliaGeometry/CoordinateTransformations.jl) / [Rotations.jl](https://github.com/JuliaGeometry/Rotations.jl) - Geometric transformations
- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) - Efficient fixed-size arrays
- [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl) + CondaPkg - LAS/LAZ I/O via laspy, E57 I/O via pye57

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
- [CloudCompare](https://www.cloudcompare.org/) - 3D point cloud processing software
