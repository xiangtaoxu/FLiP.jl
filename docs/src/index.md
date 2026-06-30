# FLiP.jl Documentation

```@meta
CurrentModule = FLiP
```

FLiP.jl (Forest Lidar Processing in Julia) is a high-performance package for processing 3D point cloud data from LiDAR and other sensors.

## Features

- **Multiple Format Support**: Read and write LAS, LAZ, PCD, PTX, and E57 files
- **Efficient Subsampling**: Voxel grid and minimum distance algorithms
- **Noise Filtering**: Statistical outlier removal and radius-based filtering
- **Ground Filtering**: Cone-based ground point extraction
- **Transformations**: Translation, rotation, scaling, and arbitrary affine transforms
- **CRS Support**: Coordinate reference system handling

## Installation

```julia
using Pkg
Pkg.add("FLiP")
```

For development:

```julia
using Pkg
Pkg.develop(url="https://github.com/xiangtaoxu/FLiP.jl")
```

## Quick Example

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

## Manual Outline

```@contents
Pages = [
    "getting_started.md",
    "guide/types.md",
    "guide/io.md",
    "guide/subsampling.md",
    "guide/filtering.md",
    "guide/transformations.md",
    "examples.md",
    "api.md",
]
Depth = 2
```

## Index

```@index
```
