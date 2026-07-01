"""
    ground_segmentation(pc::PointCloud; cfg::FLiPConfig=_CFG) -> NamedTuple

Run the ground-segmentation workflow:
1. Always: [`segment_ground`](@ref) extracts ground points using the
   voxel connected-component → grid Z-min → upward conic filter chain
   (parameters from `cfg.segment_ground_*`).
2. If `cfg.segment_ground.enable_ground_crop`: [`crop_by_ground_polygon`](@ref)
   clips the working cloud to the buffered convex hull of the ground
   points.
3. If `cfg.pipeline.enable_agh`: [`calculate_aboveground_height`](@ref)
   pointwise-IDW interpolates ground z at each query XY and stamps `:AGH`
   onto the (possibly cropped) cloud.

Returns a `NamedTuple` with fields:
- `ground_points`: segmented ground `PointCloud`
- `aboveground_height`: `Vector{Float64}` (empty if AGH disabled)
- `agh_cloud`: `PointCloud` with `:AGH` attribute (or uncropped `pc`
  if both crop and AGH are disabled)
- `ground_area`: area of the buffered ground polygon in m² (0.0 if
  cropping disabled)
"""
function ground_segmentation(pc::PointCloud; cfg::FLiPConfig=_CFG)
    ground_points = segment_ground(
        pc;
        grid_size=cfg.segment_ground.grid_size,
        cone_theta_deg=cfg.segment_ground.cone_theta_deg,
        voxel_size=cfg.segment_ground.voxel_size,
        min_cc_size=cfg.segment_ground.min_cc_size,
        verbose=cfg.pipeline.enable_debug_info,
    )

    ground_area = 0.0
    pc_use = pc

    if cfg.segment_ground.enable_ground_crop
        crop_res = crop_by_ground_polygon(pc, ground_points;
            buffer=cfg.segment_ground.polygon_buffer,
            k_neighbors=cfg.statistical_filter.k_neighbors,
            n_sigma=cfg.statistical_filter.n_sigma)
        pc_use = crop_res.pc_cropped
        ground_area = crop_res.ground_area
        @info "$_LOG_PREFIX   polygon crop: $(npoints(pc)) → $(npoints(pc_use)) points, ground area = $(round(ground_area; digits=2)) m²"
    end

    if cfg.pipeline.enable_agh
        agh = calculate_aboveground_height(
            pc_use,
            ground_points;
            xy_resolution=cfg.pipeline.xy_resolution,
            idw_k=cfg.pipeline.idw_k,
            idw_power=cfg.pipeline.idw_power,
            n_thread=effective_nthreads(cfg),
        )
        # addattribute returns a new PointCloud sharing pc_use.coords — avoids
        # leaking :AGH onto the caller's input cloud when crop is disabled.
        pc_use = addattribute(pc_use, :AGH, agh)
        return (ground_points=ground_points, aboveground_height=agh,
                agh_cloud=pc_use, ground_area=ground_area, agh_computed=true)
    end

    # AGH disabled: agh_cloud is the most-processed cloud available (possibly
    # cropped, but without :AGH). `agh_computed=false` lets callers
    # distinguish this from a real AGH run.
    return (ground_points=ground_points, aboveground_height=Float64[],
            agh_cloud=pc_use, ground_area=ground_area, agh_computed=false)
end

# ── Step 1: segment ground points ─────────────────────────────────

"""
    segment_ground(pc::PointCloud; grid_size, cone_theta_deg, voxel_size, min_cc_size)
        -> PointCloud

Segment ground points from a point cloud using:
1. Voxel connected-component filtering
2. Grid minimum-z filtering
3. Upward conic filtering
"""
function segment_ground(pc::PointCloud;
                        grid_size::Real=_CFG.segment_ground.grid_size,
                        cone_theta_deg::Real=_CFG.segment_ground.cone_theta_deg,
                        voxel_size::Real=_CFG.segment_ground.voxel_size,
                        min_cc_size::Int=_CFG.segment_ground.min_cc_size,
                        verbose::Bool=false)
    coords = coordinates(pc)
    n0 = size(coords, 1)

    idx1 = voxel_connected_component_filter(coords, voxel_size, min_cc_size=min_cc_size)
    verbose && @info "$_LOG_PREFIX     voxel CC pre-filter: $n0 → $(length(idx1)) points"
    # @view avoids materializing intermediate copies of `coords`; for a 50M-point
    # cloud at Float64 each copy would be ≈ 1.2 GB.
    idx2_local = grid_zmin_filter(@view(coords[idx1, :]), grid_size)
    idx2 = idx1[idx2_local]
    verbose && @info "$_LOG_PREFIX     grid z-min filter: → $(length(idx2)) seeds"
    idx3_local = upward_conic_filter(@view(coords[idx2, :]), cone_theta_deg)
    idx_final = idx2[idx3_local]
    verbose && @info "$_LOG_PREFIX     upward conic filter: → $(length(idx_final)) ground points"
    return pc[idx_final]
end

# ── Step 2: crop to ground polygon ────────────────────────────────

"""
    crop_by_ground_polygon(pc::PointCloud, ground_points::PointCloud;
        buffer::Real=_CFG.segment_ground.polygon_buffer,
        k_neighbors::Int=_CFG.statistical_filter.k_neighbors,
        n_sigma::Real=_CFG.statistical_filter.n_sigma) -> (pc_cropped, ground_area)

Crop a point cloud to the buffered 2D convex hull of ground points.

First applies `statistical_filter` (in index form) to the ground points
to remove outliers, then computes the convex hull in the XY plane,
expands it by `buffer` meters, and returns only the points inside the
buffered polygon.

# Arguments
- `pc`: Point cloud to crop
- `ground_points`: Ground points used to define the cropping polygon
- `buffer`: Outward buffer distance in meters (default: 0.0)
- `k_neighbors`: K for statistical filter on ground points (default from config)
- `n_sigma`: Sigma threshold for statistical filter (default from config)

# Returns
- `NamedTuple` with:
  - `pc_cropped::PointCloud`: Cropped point cloud
  - `ground_area::Float64`: Area of the buffered polygon in m²
"""
function crop_by_ground_polygon(pc::PointCloud, ground_points::PointCloud;
                                buffer::Real=_CFG.segment_ground.polygon_buffer,
                                k_neighbors::Int=_CFG.statistical_filter.k_neighbors,
                                n_sigma::Real=_CFG.statistical_filter.n_sigma)
    # Clean ground points before computing hull
    gnd_clean = ground_points[statistical_filter(coordinates(ground_points), k_neighbors, n_sigma)]
    if npoints(gnd_clean) < 3
        return (pc_cropped=pc, ground_area=0.0)
    end

    # Compute buffered convex hull
    hull = convex_hull_2d(coordinates(gnd_clean))
    if size(hull, 1) < 3
        # Collinear / near-collinear ground returns (e.g. a single scan line) collapse the
        # hull to < 3 vertices, which buffer_polygon/polygon_area cannot handle. Skip cropping
        # rather than aborting the whole ground stage.
        @warn "$_LOG_PREFIX   polygon crop: degenerate ground hull (< 3 vertices); skipping crop"
        return (pc_cropped=pc, ground_area=0.0)
    end
    hull_buffered = buffer_polygon(hull, buffer)
    area = polygon_area(hull_buffered)

    # Crop
    pc_cropped = pc[XY_polygon_filter(coordinates(pc), hull_buffered)]
    return (pc_cropped=pc_cropped, ground_area=area)
end

# ── Step 3: above-ground height ───────────────────────────────────

"""
    _interpolate_ground_grid(ground_coords; xy_resolution, idw_k, idw_power)
        -> (grid_xy::Matrix{Float64}, grid_z::Vector{Float64}, nx::Int, ny::Int)

Build the regular XY lattice over the ground bounding box at `xy_resolution`
spacing and fill every cell with IDW-interpolated z from the `idw_k` nearest
ground points (one [`interpolate_idw`](@ref) call). Shared by
`calculate_aboveground_height` (snap-to-cell AGH) and `build_ground_mesh`
(dense ground surface) so both use the identical lattice. `grid_xy` is
row-major: `row = iy*nx + ix + 1` for 0-based `ix ∈ 0:nx-1`, `iy ∈ 0:ny-1`.
"""
function _interpolate_ground_grid(ground_coords::AbstractMatrix{<:Real};
                                  xy_resolution::Real, idw_k::Integer, idw_power::Real)
    size(ground_coords, 2) == 3 || throw(ArgumentError("ground coordinates must be N×3 matrix"))
    xy_resolution > 0 || throw(ArgumentError("xy_resolution must be > 0"))
    idw_k >= 1 || throw(ArgumentError("idw_k must be >= 1"))
    idw_power > 0 || throw(ArgumentError("idw_power must be > 0"))

    # Ground XY bbox (single pass; also validates finiteness)
    xmin = Inf; ymin = Inf
    xmax = -Inf; ymax = -Inf
    @inbounds for i in 1:size(ground_coords, 1)
        x = float(ground_coords[i, 1])
        y = float(ground_coords[i, 2])
        z = float(ground_coords[i, 3])
        (isfinite(x) && isfinite(y) && isfinite(z)) ||
            throw(ArgumentError("ground_points contain non-finite values"))
        x < xmin && (xmin = x); x > xmax && (xmax = x)
        y < ymin && (ymin = y); y > ymax && (ymax = y)
    end

    # Regular grid covering the bbox at xy_resolution spacing
    step = float(xy_resolution)
    nx = max(1, floor(Int, (xmax - xmin) / step) + 1)
    ny = max(1, floor(Int, (ymax - ymin) / step) + 1)
    n_grid = nx * ny
    grid_xy = Matrix{Float64}(undef, n_grid, 2)
    @inbounds for iy in 0:(ny - 1), ix in 0:(nx - 1)
        row = iy * nx + ix + 1
        grid_xy[row, 1] = xmin + ix * step
        grid_xy[row, 2] = ymin + iy * step
    end

    # One batched IDW call fills every grid cell from k nearest ground points
    grid_z = interpolate_idw(@view(ground_coords[:, 1:2]),
                             @view(ground_coords[:, 3]),
                             grid_xy;
                             k=idw_k, power=idw_power)

    return (grid_xy=grid_xy, grid_z=grid_z, nx=nx, ny=ny)
end

"""
    calculate_aboveground_height(pc::PointCloud, ground_points::PointCloud;
        xy_resolution::Real, idw_k::Int=8, idw_power::Real=2.0,
        ground_polygon::Union{Nothing,AbstractMatrix}=nothing) -> Vector{Float64}

Compute above-ground height (AGH) for each point in `pc`:
1. Build a regular XY lattice covering the ground bbox at `xy_resolution`
   spacing.
2. Fill the lattice with IDW-interpolated z from the k nearest ground
   points (one [`interpolate_idw`](@ref) call for the whole grid).
3. For each query, snap to the nearest grid cell and return
   `z_query - z_cell`.

Queries whose nearest grid cell is farther than `sqrt(2) * xy_resolution`
(i.e. outside the ground bbox by more than one cell diagonal) get NaN.
Inside the bbox, even sparse-ground regions are interpolated — the grid
provides spatial smoothing and bbox-wide z continuity.

If `ground_polygon` is provided (M×2 vertex matrix), points outside the
polygon also get NaN — useful when the caller did not crop the cloud to
the ground footprint and wants AGH masked off outside it.

# Arguments
- `pc`: query cloud (N points)
- `ground_points`: known ground samples (≥ 3 required)
- `xy_resolution`: grid spacing in meters (must be > 0)
- `idw_k`: IDW neighbors used to fill each grid cell (must be ≥ 1)
- `idw_power`: IDW exponent (must be > 0)
- `ground_polygon`: optional XY polygon for outside-footprint NaN masking
"""
function calculate_aboveground_height(pc::PointCloud, ground_points::PointCloud;
                                      xy_resolution::Real,
                                      idw_k::Int=8,
                                      idw_power::Real=2.0,
                                      ground_polygon::Union{Nothing,AbstractMatrix}=nothing,
                                      n_thread::Integer=effective_nthreads())
    npoints(ground_points) >= 3 || throw(ArgumentError(
        "ground_points must contain at least 3 points for interpolation; got $(npoints(ground_points))"))
    xy_resolution > 0 || throw(ArgumentError("xy_resolution must be > 0"))
    idw_k >= 1 || throw(ArgumentError("idw_k must be >= 1"))
    idw_power > 0 || throw(ArgumentError("idw_power must be > 0"))

    ground_coords = coordinates(ground_points)
    size(ground_coords, 2) == 3 || throw(ArgumentError("ground_points coordinates must be N×3 matrix"))

    points = coordinates(pc)
    size(points, 2) == 3 || throw(ArgumentError("points must be N×3 matrix"))
    n = size(points, 1)

    # 1-3. Build + IDW-fill the dense ground lattice (shared with build_ground_mesh)
    grid = _interpolate_ground_grid(ground_coords;
                                    xy_resolution=xy_resolution, idw_k=idw_k, idw_power=idw_power)
    grid_xy = grid.grid_xy
    grid_z  = grid.grid_z
    n_grid  = size(grid_xy, 1)
    step    = float(xy_resolution)

    # 4. KDTree on grid samples (2 × n_grid layout for NearestNeighbors)
    grid_xy_t = Matrix{Float64}(undef, 2, n_grid)
    @inbounds for i in 1:n_grid
        grid_xy_t[1, i] = grid_xy[i, 1]
        grid_xy_t[2, i] = grid_xy[i, 2]
    end
    grid_tree = KDTree(grid_xy_t)
    max_d = sqrt(2.0) * step

    # 5. Snap each query to its nearest grid cell (threaded; writes disjoint agh[i])
    agh = Vector{Float64}(undef, n)
    _parallel_for(n, n_thread) do i
        @inbounds begin
            xq = float(points[i, 1])
            yq = float(points[i, 2])
            zq = float(points[i, 3])
            if !(isfinite(xq) && isfinite(yq) && isfinite(zq))
                agh[i] = NaN
                return
            end
            idxs, dists = knn(grid_tree, SVector(xq, yq), 1, true)
            agh[i] = dists[1] > max_d ? NaN : zq - grid_z[idxs[1]]
        end
    end

    # 6. Optional: NaN out points outside an explicit ground footprint polygon
    if ground_polygon !== nothing
        inside_idx = XY_polygon_filter(points, ground_polygon)
        inside_mask = falses(n)
        @inbounds for i in inside_idx
            inside_mask[i] = true
        end
        @inbounds for i in 1:n
            inside_mask[i] || (agh[i] = NaN)
        end
    end

    return agh
end

"""
    build_ground_mesh(ground_points::PointCloud;
                      xy_resolution, idw_k=8, idw_power=2.0)
        -> (vertices::Matrix{Float64}, faces::Vector{NTuple{3,Int}})

Build a triangulated ground surface from the dense IDW-interpolated ground
lattice — the same grid [`calculate_aboveground_height`](@ref) uses (via
[`_interpolate_ground_grid`](@ref)). Vertices are the lattice points
`(x, y, z_idw)`; `faces` are the structured two-triangles-per-cell mesh of the
regular `nx × ny` grid, as 1-based vertex-index triples. `faces` is empty if the
lattice is too thin (`nx < 2` or `ny < 2`) to form a triangle.
"""
function build_ground_mesh(ground_points::PointCloud;
                           xy_resolution::Real, idw_k::Integer=8, idw_power::Real=2.0)
    npoints(ground_points) >= 3 || throw(ArgumentError(
        "ground_points must contain at least 3 points to build a mesh; got $(npoints(ground_points))"))

    grid = _interpolate_ground_grid(coordinates(ground_points);
                                    xy_resolution=xy_resolution, idw_k=idw_k, idw_power=idw_power)
    nx, ny = grid.nx, grid.ny
    n_grid = nx * ny

    vertices = Matrix{Float64}(undef, n_grid, 3)
    @inbounds for i in 1:n_grid
        vertices[i, 1] = grid.grid_xy[i, 1]
        vertices[i, 2] = grid.grid_xy[i, 2]
        vertices[i, 3] = grid.grid_z[i]
    end

    # Structured triangulation of the regular lattice (row = iy*nx + ix + 1).
    faces = NTuple{3,Int}[]
    if nx >= 2 && ny >= 2
        sizehint!(faces, 2 * (nx - 1) * (ny - 1))
        @inbounds for iy in 0:(ny - 2), ix in 0:(nx - 2)
            v00 = iy * nx + ix + 1
            v10 = iy * nx + (ix + 1) + 1
            v01 = (iy + 1) * nx + ix + 1
            v11 = (iy + 1) * nx + (ix + 1) + 1
            push!(faces, (v00, v10, v11))
            push!(faces, (v00, v11, v01))
        end
    end

    return (vertices=vertices, faces=faces)
end
