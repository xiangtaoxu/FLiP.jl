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

# Subsample using minimum distance (filters return indices; subset explicitly)
pc_sub = pc[distance_subsample(coordinates(pc), 0.05)]

# Remove statistical outliers
pc_clean = pc_sub[statistical_filter(coordinates(pc_sub), 10, 2.0)]

# Apply transformation
pc_transformed = translate(pc_clean, 100.0, 200.0, 0.0)

# Save result
write_las("output.laz", pc_transformed)
```
"""
module FLiP

# Standard library imports
using LinearAlgebra
using SparseArrays
using Statistics

# External dependencies
using DelaunayTriangulation
using Graphs
using NearestNeighbors
using StaticArrays
using CoordinateTransformations
using MultivariateStats
using Rotations

# Include submodules
include("types/pointcloud.jl")
include("config.jl")
include("util/logging.jl")
include("util/parallelization.jl")
include("io.jl")
include("util/array_utils.jl")
include("util/geometry_utils.jl")
include("util/pointcloud_utils.jl")
include("util/interpolation.jl")
include("preprocess.jl")
include("ground_segmentation.jl")
include("types/mesh.jl")
include("util/graph_utils.jl")
include("tree_segmentation.jl")
include("qsm.jl")
include("generate_report.jl")
include("main.jl")
include("util/transformations.jl")

# Export types
export PointCloud

# Export utility functions
export npoints, coordinates, hasattribute, getattribute, setattribute!
export addattribute, deleteattribute
export bounds, center

# Export I/O functions
export read_las, read_laz, write_las, write_laz
export read_e57, write_e57
export read_pc, write_pc
export PointCloudMetadata
export read_las_metadata, read_laz_metadata, read_e57_metadata, read_pc_metadata

# Export subsampling functions
export distance_subsample

# Export config
export FLiPConfig, load_config!, coord_type

# Export generic array utilities
export relabel_by_occurrence, group_indices_by_label

# Export filtering functions
export statistical_filter
export grid_zmin_filter, upward_conic_filter
export voxel_connected_component_filter
export segment_ground
export XY_polygon_filter
export convex_hull_2d, buffer_polygon, polygon_area
export crop_by_ground_polygon
export ground_segmentation
export interpolate_idw

# Export mesh functions
export delaunay_triangulation_xy, cloud_to_mesh_distance_z

# Export graph functions
export connected_component_labels
export graph_connected_component_labels
export build_radius_graph
export build_graph
export ConnectedComponentSubsetWorkspace
export ShortestPathSubsetWorkspace
export GreedySearchWorkspace
export connected_component_subset!
export quotient_graph
export shortest_path_distances
export shortest_path_subset!
export slice_by_shortest_path
export generate_proto_nodes_from_slice_label
export greedy_neighborhood_search
export longest_linear_path
export tree_segmentation
export label_non_branching_segments
export create_skeleton_cloud
export assemble_segments

# Export pipeline functions
export preprocess, find_input_files
export calculate_aboveground_height
export qsm
export generate_report
export run_pipeline

# Export transformation functions
export translate, translate!, rotate, scale, transform
export center_at_origin, apply_transform, apply_transform!
export bounding_box_crop

end # module FLiP
