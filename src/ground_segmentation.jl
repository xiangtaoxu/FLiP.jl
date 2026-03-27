"""
    segment_ground(pc::PointCloud; grid_size, cone_theta_deg, voxel_size, min_cc_size)
        -> PointCloud

Segment ground points from a point cloud using:
1. Voxel connected-component filtering
2. Grid minimum-z filtering
3. Upward conic filtering
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

"""
    calculate_aboveground_height(pc::PointCloud, ground_points::PointCloud; xy_resolution::Real, idw_k::Int=8, idw_power::Real=2.0)
        -> Vector{Float64}

Interpolate `ground_points` onto an XY lattice using IDW, then compute
`z_point - z_ground_nearest_xy` for all points in `pc`.
"""
function calculate_aboveground_height(pc::PointCloud, ground_points::PointCloud;
                                      xy_resolution::Real,
                                      idw_k::Int=8,
                                      idw_power::Real=2.0)
    npoints(ground_points) >= 3 || throw(ArgumentError(
        "ground_points must contain at least 3 points for interpolation; got $(npoints(ground_points))"))
    xy_resolution > 0 || throw(ArgumentError("xy_resolution must be > 0"))
    idw_k >= 1 || throw(ArgumentError("idw_k must be >= 1"))
    idw_power > 0 || throw(ArgumentError("idw_power must be > 0"))

    ground_coords = coordinates(ground_points)
    size(ground_coords, 2) == 3 || throw(ArgumentError("ground_points coordinates must be N×3 matrix"))

    n_ground = size(ground_coords, 1)
    ground_xy = Matrix{Float64}(undef, 2, n_ground)
    ground_z = Vector{Float64}(undef, n_ground)

    xmin = Inf
    xmax = -Inf
    ymin = Inf
    ymax = -Inf

    @inbounds for i in 1:n_ground
        x = float(ground_coords[i, 1])
        y = float(ground_coords[i, 2])
        z = float(ground_coords[i, 3])
        (isfinite(x) && isfinite(y) && isfinite(z)) ||
            throw(ArgumentError("ground_points contain non-finite values"))

        ground_xy[1, i] = x
        ground_xy[2, i] = y
        ground_z[i] = z

        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
    end

    step = float(xy_resolution)
    nx = max(1, floor(Int, (xmax - xmin) / step) + 1)
    ny = max(1, floor(Int, (ymax - ymin) / step) + 1)
    n_grid = nx * ny

    sampled_ground = Matrix{Float64}(undef, n_grid, 3)
    ground_tree = KDTree(ground_xy)
    k_use = min(idw_k, n_ground)
    p = float(idw_power)

    idx = 1
    @inbounds for iy in 0:(ny - 1)
        y = ymin + iy * step
        for ix in 0:(nx - 1)
            x = xmin + ix * step

            nbr_idx, nbr_dist = knn(ground_tree, SVector(x, y), k_use, true)

            z_interp = NaN
            exact_found = false
            for j in eachindex(nbr_idx)
                if nbr_dist[j] <= eps(Float64)
                    z_interp = ground_z[nbr_idx[j]]
                    exact_found = true
                    break
                end
            end

            if !exact_found
                wsum = 0.0
                zwsum = 0.0
                for j in eachindex(nbr_idx)
                    d = nbr_dist[j]
                    w = 1.0 / (d^p)
                    wsum += w
                    zwsum += w * ground_z[nbr_idx[j]]
                end
                z_interp = wsum > 0 ? (zwsum / wsum) : NaN
            end

            sampled_ground[idx, 1] = x
            sampled_ground[idx, 2] = y
            sampled_ground[idx, 3] = z_interp
            idx += 1
        end
    end

    points = coordinates(pc)
    size(points, 2) == 3 || throw(ArgumentError("points must be N×3 matrix"))
    size(sampled_ground, 2) == 3 || throw(ArgumentError("sampled_ground must be N×3 matrix"))

    n_sampled = size(sampled_ground, 1)
    n_sampled > 0 || throw(ArgumentError("sampled_ground must contain at least one point"))

    sampled_xy = Matrix{Float64}(undef, 2, n_sampled)
    sampled_z = Vector{Float64}(undef, n_sampled)
    @inbounds for i in 1:n_sampled
        x = float(sampled_ground[i, 1])
        y = float(sampled_ground[i, 2])
        z = float(sampled_ground[i, 3])
        (isfinite(x) && isfinite(y) && isfinite(z)) ||
            throw(ArgumentError("sampled_ground contains non-finite values"))
        sampled_xy[1, i] = x
        sampled_xy[2, i] = y
        sampled_z[i] = z
    end

    sampled_tree = KDTree(sampled_xy)
    n = size(points, 1)
    aboveground_height = Vector{Float64}(undef, n)
    max_xy_distance = sqrt(2.0) * float(xy_resolution)

    @inbounds for i in 1:n
        xq = float(points[i, 1])
        yq = float(points[i, 2])
        zq = float(points[i, 3])

        if !(isfinite(xq) && isfinite(yq) && isfinite(zq))
            aboveground_height[i] = NaN
            continue
        end

        idxs, dists = knn(sampled_tree, SVector(xq, yq), 1, true)
        d = dists[1]
        if d > max_xy_distance
            aboveground_height[i] = NaN
            continue
        end

        aboveground_height[i] = zq - sampled_z[idxs[1]]
    end

    return aboveground_height
end

"""
    crop_by_ground_polygon(pc::PointCloud, ground_points::PointCloud;
        buffer::Real=_CFG.ground_polygon_buffer,
        k_neighbors::Int=_CFG.statistical_filter_k_neighbors,
        n_sigma::Real=_CFG.statistical_filter_n_sigma) -> (pc_cropped, ground_area)

Crop a point cloud to the buffered 2D convex hull of ground points.

First applies `statistical_filter` to the ground points to remove outliers,
then computes the convex hull in the XY plane, expands it by `buffer` meters,
and returns only the points inside the buffered polygon.

# Arguments
- `pc`: Point cloud to crop
- `ground_points`: Ground points used to define the cropping polygon
- `buffer`: Outward buffer distance in meters (default: 5.0)
- `k_neighbors`: K for statistical filter on ground points (default from config)
- `n_sigma`: Sigma threshold for statistical filter (default from config)

# Returns
- `NamedTuple` with:
  - `pc_cropped::PointCloud`: Cropped point cloud
  - `ground_area::Float64`: Area of the buffered polygon in m²
"""
function crop_by_ground_polygon(pc::PointCloud, ground_points::PointCloud;
                                buffer::Real=_CFG.ground_polygon_buffer,
                                k_neighbors::Int=_CFG.statistical_filter_k_neighbors,
                                n_sigma::Real=_CFG.statistical_filter_n_sigma)
    # Clean ground points before computing hull
    gnd_clean = statistical_filter(ground_points, k_neighbors, n_sigma)
    if npoints(gnd_clean) < 3
        return (pc_cropped=pc, ground_area=0.0)
    end

    # Compute buffered convex hull
    hull = convex_hull_2d(coordinates(gnd_clean))
    hull_buffered = buffer_polygon(hull, buffer)
    area = polygon_area(hull_buffered)

    # Crop
    pc_cropped = XY_polygon_filter(pc, hull_buffered)
    return (pc_cropped=pc_cropped, ground_area=area)
end

"""
    ground_segmentation(pc::PointCloud; cfg::FLiPConfig=_CFG) -> NamedTuple

Run configured ground segmentation, optional ground polygon cropping,
and optional AGH computation.

Returns a `NamedTuple` with fields:
- `ground_points`: segmented ground `PointCloud`
- `aboveground_height`: `Vector{Float64}` (empty if AGH disabled)
- `agh_cloud`: `PointCloud` with `:AGH` attribute (or uncropped `pc` if both crop and AGH disabled)
- `ground_area`: area of the buffered ground polygon in m² (0.0 if cropping disabled)
"""
function ground_segmentation(pc::PointCloud; cfg::FLiPConfig=_CFG)
    ground_points = segment_ground(
        pc;
        grid_size=cfg.segment_ground_grid_size,
        cone_theta_deg=cfg.segment_ground_cone_theta_deg,
        voxel_size=cfg.segment_ground_voxel_size,
        min_cc_size=cfg.segment_ground_min_cc_size,
    )

    ground_area = 0.0
    pc_use = pc

    if cfg.pipeline_enable_ground_crop
        crop_res = crop_by_ground_polygon(pc, ground_points;
            buffer=cfg.ground_polygon_buffer,
            k_neighbors=cfg.statistical_filter_k_neighbors,
            n_sigma=cfg.statistical_filter_n_sigma)
        pc_use = crop_res.pc_cropped
        ground_area = crop_res.ground_area
        @info "Ground polygon crop: $(npoints(pc)) → $(npoints(pc_use)) points, ground area = $(round(ground_area; digits=2)) m²"
    end

    if cfg.pipeline_enable_agh
        agh = calculate_aboveground_height(
            pc_use,
            ground_points;
            xy_resolution=cfg.pipeline_xy_resolution,
            idw_k=cfg.pipeline_idw_k,
            idw_power=cfg.pipeline_idw_power,
        )
        pc_agh = setattribute!(pc_use, :AGH, agh)
        return (ground_points=ground_points, aboveground_height=agh, agh_cloud=pc_agh, ground_area=ground_area)
    end

    return (ground_points=ground_points, aboveground_height=Float64[], agh_cloud=pc_use, ground_area=ground_area)
end
