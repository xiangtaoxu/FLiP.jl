"""
    calculate_aboveground_height(pc::PointCloud, ground_points::PointCloud; xy_resolution::Real, idw_k::Int=8, idw_power::Real=2.0)
        -> Vector{Float64}

Interpolate `ground_points` onto a regular XY lattice using inverse-distance
weighting (IDW), then compute approximate signed z-direction distance from every
point in `pc` to the nearest interpolated ground-grid point in XY.

# Arguments
- `pc`: Full input point cloud (all returns)
- `ground_points`: Ground-segmented point cloud (e.g. output of `segment_ground`)
- `xy_resolution`: XY spacing for mesh sampling lattice (must be > 0)
- `idw_k`: Number of nearest ground points used for IDW interpolation (must be >= 1)
- `idw_power`: Power parameter for IDW weights `w = 1 / d^idw_power` (must be > 0)

# Returns
- `aboveground_height::Vector{Float64}`: Approximate signed residual
    `z_point - z_sampled_ground_nearest_xy`; non-finite query points and points
    beyond sampled support are `NaN`

# Throws
- `ArgumentError` if `ground_points` has fewer than 3 points
- `ArgumentError` if `xy_resolution <= 0`
- `ArgumentError` if `idw_k < 1` or `idw_power <= 0`
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

@inline function _pipeline_write(path::AbstractString, pc::PointCloud; overwrite::Bool)
    if isfile(path) && !overwrite
        println("[run_pipeline] skip existing output (overwrite_outputs=false): $path")
        return false
    end
    write_las(path, pc)
    println("[run_pipeline] wrote: $path")
    return true
end

"""
    run_pipeline(config_path::AbstractString=_DEFAULT_CONFIG_PATH)

Run a config-driven point cloud processing pipeline:
1. Load pipeline configuration from TOML.
2. Read input point cloud.
3. Optionally subsample the cloud.
4. Segment ground points from the active cloud.
5. Optionally compute AGH and attach `AGH` attribute.
6. Save ground cloud and AGH cloud to output directory with configured prefix.

Output files:
- `{output_prefix}_ground.las`
- `{output_prefix}_agh.las` (when `enable_agh=true`)
"""
function run_pipeline(config_path::AbstractString=_DEFAULT_CONFIG_PATH)
    cfg = load_config!(String(config_path))

    input_path = strip(cfg.pipeline_input_path)
    output_dir = strip(cfg.pipeline_output_dir)
    output_prefix = strip(cfg.pipeline_output_prefix)

    isempty(input_path) && throw(ArgumentError("pipeline.input_path must be set in config"))
    isempty(output_dir) && throw(ArgumentError("pipeline.output_dir must be set in config"))
    isempty(output_prefix) && throw(ArgumentError("pipeline.output_prefix must be set in config"))
    isfile(input_path) || throw(ArgumentError("pipeline input file not found: $input_path"))

    cfg.pipeline_subsample_res > 0 || throw(ArgumentError("pipeline.subsample_res must be > 0"))
    cfg.pipeline_xy_resolution > 0 || throw(ArgumentError("pipeline.xy_resolution must be > 0"))
    cfg.pipeline_idw_k >= 1 || throw(ArgumentError("pipeline.idw_k must be >= 1"))
    cfg.pipeline_idw_power > 0 || throw(ArgumentError("pipeline.idw_power must be > 0"))

    mkpath(output_dir)

    pc_input = read_las(input_path)
    pc_active = cfg.pipeline_enable_subsample ? distance_subsample(pc_input, cfg.pipeline_subsample_res) : pc_input

    ground_points = segment_ground(pc_active)

    ground_path = joinpath(output_dir, "$(output_prefix)_ground.las")
    ground_written = _pipeline_write(ground_path, ground_points; overwrite=cfg.pipeline_overwrite_outputs)

    agh_path = joinpath(output_dir, "$(output_prefix)_agh.las")
    agh_written = false
    agh_count = 0

    if cfg.pipeline_enable_agh
        agh = calculate_aboveground_height(
            pc_active,
            ground_points;
            xy_resolution=cfg.pipeline_xy_resolution,
            idw_k=cfg.pipeline_idw_k,
            idw_power=cfg.pipeline_idw_power,
        )
        agh_count = length(agh)
        pc_agh = setattribute!(pc_active, :AGH, agh)
        agh_written = _pipeline_write(agh_path, pc_agh; overwrite=cfg.pipeline_overwrite_outputs)
    else
        println("[run_pipeline] AGH stage disabled by config (pipeline.enable_agh=false)")
    end

    return (
        input_path=input_path,
        output_dir=output_dir,
        output_prefix=output_prefix,
        used_subsample=cfg.pipeline_enable_subsample,
        agh_enabled=cfg.pipeline_enable_agh,
        n_input=npoints(pc_input),
        n_active=npoints(pc_active),
        n_ground=npoints(ground_points),
        n_agh=agh_count,
        ground_path=ground_path,
        agh_path=agh_path,
        ground_written=ground_written,
        agh_written=agh_written,
    )
end
