"""
    FLiP.jl - Forest Lidar Processing

A high-performance Julia package for processing 3D point cloud data from LiDAR and other sensors.

# Features
- Read/write LAS and LAZ point cloud formats
- Efficient subsampling: minimum distance
- Statistical noise filtering
- Coordinate transformations and CRS support

# Example
```julia
using FLiP

# Load point cloud
pc = read_las("input.laz")

# Subsample using minimum distance
pc_sub = distance_subsample(pc, 0.05)

# Remove statistical outliers
pc_clean = statistical_filter(pc_sub, k_neighbors=10, n_sigma=2.0)

# Apply transformation
pc_transformed = translate(pc_clean, 100.0, 200.0, 0.0)

# Save result
write_las("output.laz", pc_transformed)
```
"""
module FLiP

# Standard library imports
using LinearAlgebra
using Statistics

# External dependencies
using DelaunayTriangulation
using Graphs
using NearestNeighbors
using PointClouds
using StaticArrays
using CoordinateTransformations
using Rotations

# Include submodules
include("types.jl")
include("io.jl")
include("config.jl")
include("subsampling.jl")
include("filtering.jl")
include("segmentation.jl")
include("mesh.jl")
include("main_pipeline.jl")
include("transformations.jl")

# Export types
export AbstractPointCloud, PointCloud

# Export utility functions
export npoints, coordinates, hasattribute, getattribute, setattribute!
export addattribute, deleteattribute
export bounds, center

# Export I/O functions
export read_las, read_laz, write_las, write_laz

# Export subsampling functions
export distance_subsample, distance_subsample_indices

# Export config
export FLiPConfig, load_config!

# Export filtering functions
export statistical_filter, statistical_filter_indices
export grid_zmin_filter_indices, upward_conic_filter_indices
export voxel_connected_component_filter_indices
export rnn_filter, rnn_filter_indices
export segment_ground

# Export mesh functions
export delaunay_triangulation_xy, cloud_to_mesh_distance_z

# Export pipeline functions
export calculate_aboveground_height
export run_pipeline

# Export transformation functions
export translate, translate!, rotate, scale, transform
export center_at_origin, apply_transform, apply_transform!
export bounding_box_crop

end # module FLiP
