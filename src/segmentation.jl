"""
    segment_ground(pc::PointCloud; grid_size, cone_theta_deg, voxel_size, min_cc_size)
        -> Tuple{PointCloud, XYTriMesh, Vector{Float64}}

Segment ground points from a point cloud using a three-step filtering pipeline:
1. Voxel connected-component filtering to remove small isolated fragments.
2. Grid minimum-z filtering.
3. Upward conic filtering.
4. XY Delaunay triangulation from segmented ground points.
5. Signed z-direction distance from all input points to the ground mesh.

All defaults are read from `FLiP._CFG` (see `flip_config.toml` or `load_config!`).

# Arguments
- `pc`: Input PointCloud
- `voxel_size`: Voxel cell size for `voxel_connected_component_filter_indices`
- `min_cc_size`: Minimum connected-component size in points to keep
- `grid_size`: XY grid size for `grid_zmin_filter_indices`
- `cone_theta_deg`: Cone half-angle in degrees for `upward_conic_filter_indices`

# Returns
- `Tuple{PointCloud, XYTriMesh, Vector{Float64}}`:
    - `ground_points`: Ground-segmented point cloud
    - `ground_mesh`: XY Delaunay mesh from `delaunay_triangulation_xy`
    - `aboveground_height`: Signed z residuals (`z_point - z_ground_mesh`) for all
        points in input cloud; points outside mesh XY hull are `NaN`
"""
function segment_ground(pc::PointCloud;
                        grid_size::Real=_CFG.segment_ground_grid_size,
                        cone_theta_deg::Real=_CFG.segment_ground_cone_theta_deg,
                        voxel_size::Real=_CFG.segment_ground_voxel_size,
                        min_cc_size::Int=_CFG.segment_ground_min_cc_size)
    coords = coordinates(pc)

    idx1 = voxel_connected_component_filter_indices(coords, voxel_size, min_cc_size=min_cc_size)
    idx2_local = grid_zmin_filter_indices(coords[idx1, :], grid_size)
    idx2 = idx1[idx2_local]
    idx3_local = upward_conic_filter_indices(coords[idx2, :], cone_theta_deg)
    idx_final = idx2[idx3_local]
    return pc[idx_final]
end
