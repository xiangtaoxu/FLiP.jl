"""
Quantitative Structural Modeling (QSM) for FLiP.jl.

Converts tree-segmented point clouds into geometric measurements (circumference,
cross-sectional area, volume, surface area) per branch node and per tree.
Uses 2D periodic surface smoothing (periodic in phi, non-periodic in z) for
cross-section estimation.
"""

using LinearAlgebra: Symmetric, eigen, dot, norm, cross, normalize
using Statistics: mean, median, quantile
using NearestNeighbors: KDTree, knn

# ═══════════════════════════════════════════════════════════════════════════════
# Module-level constants
# ═══════════════════════════════════════════════════════════════════════════════

# 8-way angular partition used in slice angular-coverage checks
const OCTANT_WIDTH = π / 4

# Generalized-eigenvalue positivity tolerance in taubin_circle_fit
const TAUBIN_EIG_TOL = 1e-12

# CC radius (in units of pipeline_subsample_res) for per-slice QC connected-component pass
const QC_CC_RADIUS_SCALAR = 2.0

# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

"""Per-NBS metadata after linearity filtering and PCA."""
struct NBSInfo
    direction::NTuple{3,Float64}   # PC1 unit vector
    center::NTuple{3,Float64}      # mean centroid
    eigenvalues::NTuple{3,Float64} # ascending order
    linearity::Float64
    point_indices::Vector{Int}
end

"""Per-QSM node biometric results (one per z-slice per NBS)."""
mutable struct QSMNode
    qsm_node_id::Int
    nbs_id::Int32
    tree_id::Int32
    tree_nbs_id::Int32
    agh::Float64
    height::Float64
    completeness::Float64
    n_points::Int
    center_x::Float64
    center_y::Float64
    center_z::Float64
    direction_x::Float64
    direction_y::Float64
    direction_z::Float64
    cross_area::Float64
    circumference::Float64
    radius_area::Float64
    radius_circ::Float64
end

# ═══════════════════════════════════════════════════════════════════════════════
# ★ Public entry point: qsm
# ═══════════════════════════════════════════════════════════════════════════════
#
# High-level pipeline:
#   Stage 1 — _filter_linear_nbs   : keep NBSes that are linear enough to QSM
#   Stage 2 — _process_single_nbs! : per-NBS slicing, fitting, surface estimation
#               2a slicing & centerline   2d surface point cloud generation
#               2b unroll & rho stats     2e frustum geometry
#               2c 2D surface smoothing
#   Stage 3 — _build_node_table, _build_tree_table, _write_csv : aggregation & I/O
# ═══════════════════════════════════════════════════════════════════════════════

"""
    qsm(; tree_result, config_path, output_dir, output_prefix, kwargs...) -> NamedTuple

Run quantitative structural modeling on tree-segmented point cloud.

# Arguments
- `tree_result`: NamedTuple from `tree_segmentation` with fields `pc_output`, `skeleton_cloud`
- `config_path`: Path to TOML config file
- `output_dir`: Directory for CSV output files
- `output_prefix`: Filename prefix for outputs

# Returns
NamedTuple with fields:
- `status`: `:success` or `:no_linear_nbs`
- `n_nodes`: Number of QSM nodes created
- `n_trees`: Number of trees with QSM data
- `node_csv_path`: Path to node-level CSV (includes all NBS)
- `tree_csv_path`: Path to tree-level CSV (tree NBS only)
- `pc_output`: Point cloud with `:qsm_node_id` attribute added
- `qsm_surface_cloud`: Point cloud of QSM surface with `:tree_nbs_id` attribute
- `surface_cloud_path`: Path to surface cloud LAZ file
"""
function qsm(; tree_result=nothing, config_path::AbstractString="", output_dir::AbstractString="",
               output_prefix::AbstractString="output", tree_cloud_path::AbstractString="", kwargs...)
    cfg = isempty(config_path) ? _CFG : load_config!(String(config_path))

    if isnothing(tree_result) || npoints(tree_result.pc_output) == 0
        @warn "[qsm] No tree segmentation data available"
        return (status=:no_data, n_nodes=0, n_trees=0,
                node_csv_path="", tree_csv_path="", pc_output=nothing)
    end

    pc = tree_result.pc_output
    coords = coordinates(pc)
    N = size(coords, 1)

    # Required attributes (group QSM by tree_nbs_id — the merged, post-assembly identifier)
    tree_ids = hasattribute(pc, :tree_id) ? getattribute(pc, :tree_id) : zeros(Int32, N)
    tree_nbs_ids = hasattribute(pc, :tree_nbs_id) ? getattribute(pc, :tree_nbs_id) : zeros(Int32, N)
    agh_values = hasattribute(pc, :AGH) ? getattribute(pc, :AGH) : zeros(Float64, N)

    @info "[qsm] processing point cloud" n_points=N

    # Stage 1: Filter linear NBS segments
    linear_nbs = _filter_linear_nbs(coords, tree_nbs_ids, cfg)
    n_linear = count(!isnothing, linear_nbs)
    @info "[qsm] linear NBS filter" n_linear linearity_threshold=cfg.qsm_nbs_linearity_threshold

    if n_linear == 0
        setattribute!(pc, :qsm_node_id, zeros(Int32, N))
        return (status=:no_linear_nbs, n_nodes=0, n_trees=0,
                node_csv_path="", tree_csv_path="", pc_output=pc)
    end

    # Stage 2: Process each NBS through the slicing / surface-fitting pipeline
    nodes = QSMNode[]
    sizehint!(nodes, n_linear * 10)
    next_node_id = 1

    qsm_node_ids = zeros(Int32, N)
    surface_parts = Matrix{Float64}[]
    surface_nbs_parts = Vector{Int32}[]
    surface_rho_parts = Vector{Float64}[]

    for nid in 1:length(linear_nbs)
        info = linear_nbs[nid]
        info === nothing && continue
        nid32 = Int32(nid)
        next_node_id, surf_pts, surf_rho = _process_single_nbs!(nodes, qsm_node_ids,
                                            coords, info, nid32,
                                            tree_ids, agh_values, cfg, next_node_id)

        # Accumulate surface point cloud
        if size(surf_pts, 1) > 0
            push!(surface_parts, surf_pts)
            push!(surface_nbs_parts, fill(nid32, size(surf_pts, 1)))
            push!(surface_rho_parts, surf_rho)
        end
    end

    @info "[qsm] QSM node creation" n_nodes=length(nodes)

    # Stage 3: Build output tables and write CSV / surface cloud
    node_columns, node_headers, vol, sa = _build_node_table(nodes)
    tree_columns, tree_headers = _build_tree_table(nodes, vol, sa, cfg)

    n_nodes_out = isempty(node_columns) ? 0 : length(first(node_columns))
    n_trees     = isempty(tree_columns) ? 0 : length(first(tree_columns))
    @info "[qsm] tree aggregation" n_trees

    node_csv = joinpath(output_dir, "$(output_prefix)qsm_nodes.csv")
    tree_csv = joinpath(output_dir, "$(output_prefix)qsm_trees.csv")

    if !isempty(output_dir)
        mkpath(output_dir)
        if n_nodes_out > 0
            _write_csv(node_csv, node_columns, node_headers)
            @info "[qsm] wrote node biometrics" path=node_csv
        end
        if n_trees > 0
            _write_csv(tree_csv, tree_columns, tree_headers)
            @info "[qsm] wrote tree biometrics" path=tree_csv
        end
    end

    # Build QSM surface point cloud
    surf_cloud_path = joinpath(output_dir, "$(output_prefix)qsm_surface.laz")
    if !isempty(surface_parts)
        surf_coords = vcat(surface_parts...)
        surf_nbs = vcat(surface_nbs_parts...)
        surf_rho = vcat(surface_rho_parts...)
        qsm_surface_cloud = PointCloud(surf_coords, Dict{Symbol,Vector}(:tree_nbs_id => surf_nbs, :rho => surf_rho))
        @info "[qsm] generated QSM surface cloud" n_points=npoints(qsm_surface_cloud)
    else
        qsm_surface_cloud = PointCloud(Matrix{Float64}(undef, 0, 3), Dict{Symbol,Vector}())
    end

    if !isempty(output_dir) && npoints(qsm_surface_cloud) > 0
        write_pc(surf_cloud_path, qsm_surface_cloud)
        @info "[qsm] wrote QSM surface cloud" path=surf_cloud_path
    end

    # Add QSM node IDs to point cloud and overwrite tree cloud on disk
    setattribute!(pc, :qsm_node_id, qsm_node_ids)
    if !isempty(tree_cloud_path)
        write_pc(tree_cloud_path, pc)
        @info "[qsm] overwrote tree cloud with qsm_node_id" path=tree_cloud_path
    end

    return (
        status = :success,
        n_nodes = length(nodes),
        n_trees = n_trees,
        node_csv_path = node_csv,
        tree_csv_path = tree_csv,
        pc_output = pc,
        qsm_surface_cloud = qsm_surface_cloud,
        surface_cloud_path = surf_cloud_path,
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — NBS linearity filtering
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _filter_linear_nbs(coords, nbs_ids, cfg) -> Vector{Union{Nothing, NBSInfo}}

Filter NBS segments by linearity and return PCA results for qualifying ones,
as a vector indexed by NBS id (1..K). Entries are `nothing` for ids with no
points or that failed the linearity test. NBS ids are assumed dense from
`tree_segmentation` (post `relabel_by_occurrence`).

Delegates the geometric PCA + linearity check to `pca_linearity` (in
geometry_utils.jl) and adds the QSM-specific tree-stem orientation: PC1 is
flipped so it points from the lowest-z point to the highest-z point of the NBS.
"""
function _filter_linear_nbs(coords::AbstractMatrix{<:Real},
                            nbs_ids::AbstractVector{<:Integer},
                            cfg::FLiPConfig)
    K = Int(maximum(nbs_ids; init=Int32(0)))
    K == 0 && return Vector{Union{Nothing, NBSInfo}}()

    nbs_groups = [Int[] for _ in 1:K]
    @inbounds for i in eachindex(nbs_ids)
        nid = Int(nbs_ids[i])
        nid > 0 && push!(nbs_groups[nid], i)
    end

    result = Vector{Union{Nothing, NBSInfo}}(nothing, K)
    @inbounds for nid in 1:K
        indices = nbs_groups[nid]
        isempty(indices) && continue
        pca = pca_linearity(coords, indices, cfg.qsm_nbs_linearity_threshold)
        pca === nothing && continue

        # Orient PC1 by z (tree-stem convention: PC1 from low-z to high-z)
        d = pca.direction
        c = pca.center
        z_min_i = indices[1]; z_max_i = indices[1]
        for i in indices
            if coords[i, 3] < coords[z_min_i, 3]; z_min_i = i; end
            if coords[i, 3] > coords[z_max_i, 3]; z_max_i = i; end
        end
        proj_low  = (coords[z_min_i, 1] - c[1]) * d[1] +
                    (coords[z_min_i, 2] - c[2]) * d[2] +
                    (coords[z_min_i, 3] - c[3]) * d[3]
        proj_high = (coords[z_max_i, 1] - c[1]) * d[1] +
                    (coords[z_max_i, 2] - c[2]) * d[2] +
                    (coords[z_max_i, 3] - c[3]) * d[3]
        dvec = proj_high < proj_low ? (-d[1], -d[2], -d[3]) : d

        result[nid] = NBSInfo(dvec, pca.center, pca.eigenvalues, pca.linearity, indices)
    end
    return result
end

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Per-NBS processing
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _process_single_nbs!(nodes, qsm_node_ids, coords, info, nbs_id, tree_ids,
                         agh_values, cfg, next_node_id)
        -> (next_node_id, surface_pts, surface_rho)

Process a single linear NBS through the full QSM pipeline.
Appends `QSMNode` entries to `nodes` and writes the assigned QSM node id of
every point of this NBS into `qsm_node_ids` (indexed in the global point cloud).

Pipeline within this function:
  2a  _slice_and_fit_centers (incl. _finalize_centerline!) (slicing & centerline)
  2b  _unroll_points          →  _filter_rho_outliers        (unroll + per-slice rho filter)
  2c  _method_spline_2d                                      (2D surface integration)
       └ writes QSMNodes + maps point→node →
  2d  _generate_surface_points                                (3D surface cloud)
"""
function _process_single_nbs!(nodes::Vector{QSMNode},
                              qsm_node_ids::AbstractVector{Int32},
                              coords::AbstractMatrix{<:Real},
                              info::NBSInfo,
                              nbs_id::Int32,
                              tree_ids::AbstractVector{<:Integer},
                              agh_values::AbstractVector{<:Real},
                              cfg::FLiPConfig,
                              next_node_id::Int)
    slice_res = cfg.qsm_slice_height_scalar * cfg.pipeline_subsample_res

    # 2a: Slice, fit centers, interpolate, smooth. Per-slice QC inside
    # _slice_and_fit_centers may drop points; `indices` is rebound here to the
    # compacted survivor list (positions into `coords`), and pt_slice_ids /
    # slice_point_indices are renumbered to match. The returned `centers`
    # already has NaN slices interpolated and moving-window smoothing applied
    # (both done inside `_finalize_centerline!`). Dropped points keep
    # qsm_node_id = 0 by default.
    centers, slice_point_indices, _, _, pt_slice_ids, e1, e2, indices =
        _slice_and_fit_centers(coords, info, slice_res, cfg.qsm_min_node_size,
                               cfg.qsm_min_octant_taubin, cfg)

    # 2b: Unroll points to (rho, phi)
    rho, phi = _unroll_points(coords, indices, centers, pt_slice_ids, e1, e2)

    # 2b': Drop per-slice rho outliers before they touch any downstream stage
    # (compacts indices / pt_slice_ids / slice_point_indices / rho / phi together).
    n_slices = size(centers, 1)
    rho, phi, pt_slice_ids, slice_point_indices, indices =
        _filter_rho_outliers(rho, phi, pt_slice_ids, slice_point_indices, indices,
                             n_slices, cfg.qsm_rho_percentile)

    # All points in an NBS share the same tree_id (tree_segmentation assigns per-NBS),
    # so a single lookup suffices instead of an argmax over a histogram.
    dominant_tree = isempty(indices) ? Int32(0) : Int32(tree_ids[indices[1]])
    dominant_tree_nbs = nbs_id

    # 2c: 2D spline surface for all slices at once (rho already pre-filtered above)
    spl_results, surface_grid, _ = _method_spline_2d(rho, phi, pt_slice_ids, n_slices, cfg)

    # Create QSMNode per qualifying slice (push directly into `nodes`)
    node_id_per_slice = zeros(Int32, n_slices)
    for s in 1:n_slices
        local_js = slice_point_indices[s]
        n_pts = length(local_js)
        n_pts < cfg.qsm_min_node_size && continue

        # Mean AGH for this slice
        mean_agh = 0.0
        @inbounds for j in local_js
            mean_agh += float(agh_values[indices[j]])
        end
        mean_agh /= n_pts

        # 2D spline (pre-computed); skip slices that fail the completeness gate.
        ca = spl_results[s].cross_area
        circ = spl_results[s].circumference
        completeness = spl_results[s].completeness
        completeness < cfg.qsm_completeness_threshold && continue

        push!(nodes, QSMNode(
            next_node_id, nbs_id, dominant_tree, dominant_tree_nbs,
            mean_agh, slice_res, completeness, n_pts,
            centers[s, 1], centers[s, 2], centers[s, 3],
            info.direction[1], info.direction[2], info.direction[3],
            ca, circ,
            sqrt(max(0.0, ca / π)), circ / (2π),
        ))
        node_id_per_slice[s] = Int32(next_node_id)
        next_node_id += 1
    end

    # Map each NBS point to its slice's QSM node (zero means no node assigned for that slice)
    @inbounds for j in eachindex(indices)
        nid_q = node_id_per_slice[pt_slice_ids[j]]
        nid_q > 0 && (qsm_node_ids[indices[j]] = nid_q)
    end

    # 2d: Generate surface point cloud from the smoothed 2D surface, emitting
    # the per-point surface radius as an attribute alongside coordinates.
    gen_res = cfg.pipeline_subsample_res
    surface_pts, surface_rho = _generate_surface_points(surface_grid, centers, e1, e2, slice_res, gen_res)

    return (next_node_id, surface_pts, surface_rho)
end

# ───────────────────────────────────────────────────────────────────────────────
# 2a. Slicing & centerline
# ───────────────────────────────────────────────────────────────────────────────

"""
    _slice_and_fit_centers(coords, info, slice_res, min_node_size,
                           min_octant_taubin, cfg)
        -> (centers_3d, slice_point_indices, t_vals, t_min,
            point_slice_ids_kept, e1, e2, indices_kept)

Slice an NBS along PC1, run per-slice quality control (largest 3D CC with
continuity tie-break + 3D statistical outlier removal), then fit a circle
center per slice. QC is governed by `cfg.qsm_qc_*` fields; when
`cfg.qsm_qc_enable` is `false` it is a no-op.

Returns:
- `centers_3d` — K×3 fitted centers (NaN-invalid slices interpolated)
- `slice_point_indices` — per-slice positions into `indices_kept`
- `t_vals` — PC1 projection of every *original* NBS point
- `t_min` — minimum t (slice 1 starts here)
- `point_slice_ids_kept` — slice id per *kept* point, parallel to `indices_kept`
- `e1`, `e2` — orthonormal basis perpendicular to PC1
- `indices_kept` — compacted global point indices (subset of `info.point_indices`)
"""
function _slice_and_fit_centers(coords::AbstractMatrix{<:Real}, info::NBSInfo,
                                slice_res::Float64, min_node_size::Int,
                                min_octant_taubin::Int, cfg::FLiPConfig)
    d = info.direction
    cx, cy, cz = info.center
    indices = info.point_indices
    e1, e2 = _build_perpendicular_basis(d)

    n = length(indices)
    t_vals = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        i = indices[j]
        dx = coords[i, 1] - cx
        dy = coords[i, 2] - cy
        dz = coords[i, 3] - cz
        t_vals[j] = dx * d[1] + dy * d[2] + dz * d[3]
    end

    t_min = minimum(t_vals)
    t_max = maximum(t_vals)
    n_slices = max(1, ceil(Int, (t_max - t_min) / slice_res))

    # Assign points to slices
    slice_point_indices = [Int[] for _ in 1:n_slices]
    point_slice_ids = Vector{Int}(undef, n)
    @inbounds for j in 1:n
        s = clamp(floor(Int, (t_vals[j] - t_min) / slice_res) + 1, 1, n_slices)
        push!(slice_point_indices[s], j)  # j is local index into indices
        point_slice_ids[j] = s
    end

    # Per-slice QC tuning (resolved once; reuses existing config knobs)
    qc_enable  = cfg.qsm_qc_enable
    cc_radius  = QC_CC_RADIUS_SCALAR * cfg.pipeline_subsample_res
    cc_min     = cfg.qsm_min_node_size
    cont_ratio = cfg.qsm_qc_continuity_ratio
    sor_k      = cfg.statistical_filter_k_neighbors
    sor_ns     = cfg.statistical_filter_n_sigma

    # Fit circle center per slice
    centers_3d = Matrix{Float64}(undef, n_slices, 3)
    valid_slices = falses(n_slices)
    # Anchor for QC's continuity tie-break: the (cu, cv) of the last slice
    # whose fit succeeded. Nothing until the first valid slice has been fit.
    prev_centroid_uv = nothing

    for s in 1:n_slices
        local_js = slice_point_indices[s]
        if length(local_js) < min_node_size
            # Mark invalid; will be interpolated
            centers_3d[s, :] .= NaN
            continue
        end

        # Project to 2D plane perpendicular to PC1
        ns = length(local_js)
        u_arr = Vector{Float64}(undef, ns)
        v_arr = Vector{Float64}(undef, ns)
        @inbounds for k in 1:ns
            i = indices[local_js[k]]
            dx = coords[i, 1] - cx
            dy = coords[i, 2] - cy
            dz = coords[i, 3] - cz
            u_arr[k] = dx * e1[1] + dy * e1[2] + dz * e1[3]
            v_arr[k] = dx * e2[1] + dy * e2[2] + dz * e2[3]
        end

        # Per-slice QC: keep dominant 3D connected component (continuity tie-break
        # against the previous slice's fitted center) then drop 3D statistical outliers.
        if qc_enable && ns >= cc_min
            local_js, u_arr, v_arr, ns, ok = _qc_clean_slice(
                local_js, ns, u_arr, v_arr, indices, coords,
                prev_centroid_uv, min_node_size,
                cc_radius, cc_min, cont_ratio, sor_k, sor_ns)
            if !ok
                centers_3d[s, :] .= NaN
                slice_point_indices[s] = Int[]
                continue
            end
        end

        # Angular coverage relative to NBS axis: count distinct octants (OCTANT_WIDTH wide)
        covered_octants = falses(8)
        @inbounds for k in 1:ns
            oct = clamp(floor(Int, (atan(v_arr[k], u_arr[k]) + π) / OCTANT_WIDTH) + 1, 1, 8)
            covered_octants[oct] = true
        end
        n_octants = count(covered_octants)

        local cu::Float64, cv::Float64
        if ns >= 10 && n_octants >= min_octant_taubin
            cu, cv, _ = taubin_circle_fit(u_arr, v_arr)

            # Post-hoc octant check around the Taubin-fitted center: a pathologically
            # large radius places the center far outside the slice, collapsing the
            # angular spread seen from there. Fall back to centroid in that case.
            post_covered = falses(8)
            @inbounds for k in 1:ns
                oct = clamp(floor(Int,
                    (atan(v_arr[k] - cv, u_arr[k] - cu) + π) / OCTANT_WIDTH) + 1, 1, 8)
                post_covered[oct] = true
            end
            if count(post_covered) < min_octant_taubin
                cu = mean(u_arr); cv = mean(v_arr)
            end
        else
            cu = mean(u_arr)
            cv = mean(v_arr)
        end
        valid_slices[s] = true

        # Anchor for next slice's continuity tie-break: use the fitted center,
        # not the raw centroid, so partial arcs don't drag the anchor toward
        # the arc side of the stem.
        prev_centroid_uv = (cu, cv)

        # Persist the cleaned slice subset (positions into the original indices).
        slice_point_indices[s] = local_js

        # Convert 2D center back to 3D
        t_center = t_min + (s - 0.5) * slice_res
        centers_3d[s, 1] = cx + t_center * d[1] + cu * e1[1] + cv * e2[1]
        centers_3d[s, 2] = cy + t_center * d[2] + cu * e1[2] + cv * e2[2]
        centers_3d[s, 3] = cz + t_center * d[3] + cu * e1[3] + cv * e2[3]
    end

    _finalize_centerline!(centers_3d, valid_slices, info, t_min, slice_res)

    # Compact indices / pt_slice_ids / slice_point_indices to drop QC-rejected
    # points. Renumber slice_point_indices entries to point into the compacted
    # `indices_kept` so all downstream consumers (_unroll_points,
    # _build_rho_surface, the slice→node mapping)
    # operate on a contiguous survivor set without zero sentinels.
    kept_mask = falses(n)
    @inbounds for s in 1:n_slices
        for j in slice_point_indices[s]
            kept_mask[j] = true
        end
    end
    n_kept = count(kept_mask)
    indices_kept         = Vector{Int}(undef, n_kept)
    point_slice_ids_kept = Vector{Int}(undef, n_kept)
    old_to_new           = zeros(Int, n)
    pos = 0
    @inbounds for j in 1:n
        if kept_mask[j]
            pos += 1
            indices_kept[pos]         = indices[j]
            point_slice_ids_kept[pos] = point_slice_ids[j]
            old_to_new[j] = pos
        end
    end
    @inbounds for s in 1:n_slices
        sj = slice_point_indices[s]
        for k in eachindex(sj)
            sj[k] = old_to_new[sj[k]]
        end
    end

    return (centers_3d, slice_point_indices, t_vals, t_min,
            point_slice_ids_kept, e1, e2, indices_kept)
end

"""
    _qc_clean_slice(local_js, ns, u_arr, v_arr, indices, coords,
                    prev_centroid_uv, min_node_size,
                    cc_radius, cc_min, cont_ratio, sor_k, sor_ns)
        -> (local_js, u_arr, v_arr, ns, ok)

Per-slice quality control. SOR runs first on raw XYZ to drop noise, then a
connected-component pass keeps the dominant component (with a continuity
tie-break against the previous slice's fitted (u,v) center). Running SOR
before CC prevents outlier "bridges" from spuriously merging two real
components. Returns the cleaned `(local_js, u_arr, v_arr, ns)` subset plus
`ok::Bool`, where `ok=false` means the slice has too few survivors to fit.

SOR and CC both operate in 3D on raw xyz so they see the natural Euclidean
geometry. The continuity tie-break uses (u,v) centroids against the previous
slice's fitted center; this anchors the kept CC to the stem axis across
slices, not to whichever CC happens to be largest in this particular slice.
"""
function _qc_clean_slice(local_js::Vector{Int}, ns::Int,
                          u_arr::Vector{Float64}, v_arr::Vector{Float64},
                          indices::Vector{Int}, coords::AbstractMatrix{<:Real},
                          prev_centroid_uv, min_node_size::Int,
                          cc_radius::Float64, cc_min::Int,
                          cont_ratio::Float64, sor_k::Int, sor_ns::Float64)
    xyz_slice = Matrix{Float64}(undef, ns, 3)
    @inbounds for k in 1:ns
        i = indices[local_js[k]]
        xyz_slice[k, 1] = coords[i, 1]
        xyz_slice[k, 2] = coords[i, 2]
        xyz_slice[k, 3] = coords[i, 3]
    end

    # 3D statistical outlier removal on raw coordinates — first, so outlier
    # bridges can't spuriously connect two true components in the CC pass.
    inliers = statistical_filter(xyz_slice, sor_k, sor_ns)
    if length(inliers) < min_node_size
        return (local_js, u_arr, v_arr, ns, false)
    end
    local_js  = local_js[inliers]
    u_arr     = u_arr[inliers]
    v_arr     = v_arr[inliers]
    xyz_slice = xyz_slice[inliers, :]
    ns        = length(local_js)

    # Connected-component labelling on the SOR-cleaned slice
    cc_labels = connected_component_labels(xyz_slice, cc_radius, cc_min)

    chosen_label = 1
    n_comp = maximum(cc_labels; init=0)
    if n_comp >= 2 && prev_centroid_uv !== nothing
        sizes = zeros(Int, n_comp)
        @inbounds for k in 1:ns
            L = cc_labels[k]
            L > 0 && (sizes[L] += 1)
        end
        if sizes[2] >= cont_ratio * sizes[1]
            u1 = 0.0; v1 = 0.0; c1 = 0
            u2 = 0.0; v2 = 0.0; c2 = 0
            @inbounds for k in 1:ns
                L = cc_labels[k]
                if L == 1
                    u1 += u_arr[k]; v1 += v_arr[k]; c1 += 1
                elseif L == 2
                    u2 += u_arr[k]; v2 += v_arr[k]; c2 += 1
                end
            end
            u1 /= c1; v1 /= c1; u2 /= c2; v2 /= c2
            pu, pv = prev_centroid_uv
            if (u2 - pu)^2 + (v2 - pv)^2 < (u1 - pu)^2 + (v1 - pv)^2
                chosen_label = 2
            end
        end
    end

    keep_mask = cc_labels .== chosen_label
    if count(keep_mask) < min_node_size
        return (local_js, u_arr, v_arr, ns, false)
    end
    local_js = local_js[keep_mask]
    u_arr    = u_arr[keep_mask]
    v_arr    = v_arr[keep_mask]
    return (local_js, u_arr, v_arr, length(local_js), true)
end

"""
    _finalize_centerline!(centers, valid_slices, info, t_min, slice_res; window=2)

Produce the final centerline matrix in place in two passes:
1. Fill NaN rows by linear interpolation between nearest valid neighbors;
   fall back to the NBS axis at `t_min + (s - 0.5) * slice_res` if no valid
   neighbor exists on a side.
2. Smooth the resulting centerline with a moving-window average of
   half-width `window`. Pass `window=0` to skip the smoothing pass.

Both passes mutate `centers`.
"""
function _finalize_centerline!(centers::Matrix{Float64},
                                valid_slices::AbstractVector{Bool},
                                info::NBSInfo,
                                t_min::Float64,
                                slice_res::Float64;
                                window::Int=2)
    n_slices = size(centers, 1)

    # Pass 1: interpolate NaN slices from nearest valid neighbors (axis fallback)
    d = info.direction
    cx, cy, cz = info.center
    for s in 1:n_slices
        valid_slices[s] && continue
        lo = 0; hi = 0
        for k in (s-1):-1:1
            if valid_slices[k]; lo = k; break; end
        end
        for k in (s+1):n_slices
            if valid_slices[k]; hi = k; break; end
        end
        if lo > 0 && hi > 0
            w = (s - lo) / (hi - lo)
            centers[s, :] .= (1 - w) .* centers[lo, :] .+ w .* centers[hi, :]
        elseif lo > 0
            centers[s, :] .= centers[lo, :]
        elseif hi > 0
            centers[s, :] .= centers[hi, :]
        else
            t_center = t_min + (s - 0.5) * slice_res
            centers[s, 1] = cx + t_center * d[1]
            centers[s, 2] = cy + t_center * d[2]
            centers[s, 3] = cz + t_center * d[3]
        end
    end

    # Pass 2: moving-window average smoothing (in place via single scratch buf)
    if window > 0 && n_slices > 1
        buf = similar(centers)
        @inbounds for s in 1:n_slices
            lo = max(1, s - window)
            hi = min(n_slices, s + window)
            cnt = hi - lo + 1
            sx = 0.0; sy = 0.0; sz = 0.0
            for k in lo:hi
                sx += centers[k, 1]; sy += centers[k, 2]; sz += centers[k, 3]
            end
            buf[s, 1] = sx / cnt; buf[s, 2] = sy / cnt; buf[s, 3] = sz / cnt
        end
        centers .= buf
    end

    return centers
end

"""
    taubin_circle_fit(u, v) -> (cx, cy, r)

Fit a circle to 2D points using the Taubin algebraic method (1991).
Robust to partial arcs. Returns center (cx, cy) and radius r.
Falls back to centroid + mean distance if SVD fails.
"""
function taubin_circle_fit(u::AbstractVector{<:Real}, v::AbstractVector{<:Real})
    n = length(u)
    @assert n == length(v)

    um = mean(u); vm = mean(v)
    uc = u .- um; vc = v .- vm

    # Build constraint matrix for Taubin method
    # Minimize algebraic distance subject to gradient-weighted normalization
    # Z = [u² + v², u, v, 1]  →  ZᵀZ eigenproblem with constraint matrix M
    z1 = uc .^ 2 .+ vc .^ 2
    Z = hcat(z1, uc, vc, ones(n))
    M = Z' * Z ./ n

    # Constraint matrix N (Taubin normalization)
    mean_z1 = mean(z1)
    N = zeros(4, 4)
    N[1, 1] = 8.0 * mean_z1
    N[1, 2] = N[2, 1] = 4.0 * mean(uc)   # should be ~0 since centered
    N[1, 3] = N[3, 1] = 4.0 * mean(vc)
    N[2, 2] = 1.0
    N[3, 3] = 1.0

    # Solve generalized eigenvalue problem M*a = η*N*a
    # Find smallest positive generalized eigenvalue
    try
        F = eigen(Symmetric(M), Symmetric(N))
        # Find smallest positive eigenvalue
        best_idx = 0
        best_val = Inf
        for j in 1:4
            λ = F.values[j]
            if λ > TAUBIN_EIG_TOL && λ < best_val
                best_val = λ
                best_idx = j
            end
        end
        if best_idx > 0
            a = F.vectors[:, best_idx]
            cx_c = -a[2] / (2.0 * a[1])
            cy_c = -a[3] / (2.0 * a[1])
            r = sqrt(cx_c^2 + cy_c^2 + (a[2]^2 + a[3]^2 - 4.0 * a[1] * a[4]) / (4.0 * a[1]^2))
            return (um + cx_c, vm + cy_c, abs(r))
        end
    catch
        # Fall through to simple method
    end

    # Fallback: simple algebraic fit (Kasa)
    A = hcat(2.0 .* uc, 2.0 .* vc, ones(n))
    b = uc .^ 2 .+ vc .^ 2
    try
        x = A \ b
        cx_c = x[1]; cy_c = x[2]
        r = sqrt(x[3] + cx_c^2 + cy_c^2)
        return (um + cx_c, vm + cy_c, abs(r))
    catch
        # Ultimate fallback: centroid + mean distance
        dists = sqrt.(uc .^ 2 .+ vc .^ 2)
        return (um, vm, mean(dists))
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# 2b. Unroll points & per-slice rho statistics
# ───────────────────────────────────────────────────────────────────────────────

"""
    _unroll_points(coords, indices, centers, point_slice_ids, e1, e2)
        -> (rho, phi)

Convert points to cylindrical coordinates relative to the smoothed centerline.
"""
function _unroll_points(coords::AbstractMatrix{<:Real}, indices::Vector{Int},
                        centers::Matrix{Float64}, point_slice_ids::Vector{Int},
                        e1::NTuple{3,Float64}, e2::NTuple{3,Float64})
    n = length(indices)
    rho = Vector{Float64}(undef, n)
    phi = Vector{Float64}(undef, n)

    @inbounds for j in 1:n
        i = indices[j]
        s = point_slice_ids[j]
        dx = coords[i, 1] - centers[s, 1]
        dy = coords[i, 2] - centers[s, 2]
        dz = coords[i, 3] - centers[s, 3]
        u_val = dx * e1[1] + dy * e1[2] + dz * e1[3]
        v_val = dx * e2[1] + dy * e2[2] + dz * e2[3]
        rho[j] = sqrt(u_val^2 + v_val^2)
        phi[j] = atan(v_val, u_val)
    end

    return (rho, phi)
end

"""
    _filter_rho_outliers(rho, phi, pt_slice_ids, slice_point_indices, indices,
                          n_slices, rho_percentile)
        -> (rho, phi, pt_slice_ids, slice_point_indices, indices)

For each slice, drop points whose `rho` exceeds the per-slice
`quantile(rho_slice, rho_percentile)`. The kept points are returned as a
compacted survivor set so all downstream consumers (`_method_spline_2d`,
`_generate_surface_points`, slice→node mapping) operate on contiguous arrays
without sentinel handling.

When `rho_percentile >= 1.0` returns the inputs unchanged (no-op fast path).
"""
function _filter_rho_outliers(rho::Vector{Float64}, phi::Vector{Float64},
                               pt_slice_ids::Vector{Int},
                               slice_point_indices::Vector{Vector{Int}},
                               indices::Vector{Int},
                               n_slices::Int,
                               rho_percentile::Float64)
    rho_percentile >= 1.0 && return (rho, phi, pt_slice_ids, slice_point_indices, indices)

    # Per-slice rho cutoffs (Inf for slices with no points → keeps no-op for them)
    thresh = fill(Inf, n_slices)
    @inbounds for s in 1:n_slices
        ids = slice_point_indices[s]
        isempty(ids) && continue
        thresh[s] = quantile(@view(rho[ids]), rho_percentile)
    end

    n = length(rho)
    keep_mask = falses(n)
    @inbounds for j in 1:n
        keep_mask[j] = rho[j] <= thresh[pt_slice_ids[j]]
    end

    n_kept = count(keep_mask)
    rho_kept           = Vector{Float64}(undef, n_kept)
    phi_kept           = Vector{Float64}(undef, n_kept)
    pt_slice_ids_kept  = Vector{Int}(undef, n_kept)
    indices_kept       = Vector{Int}(undef, n_kept)
    old_to_new         = zeros(Int, n)
    pos = 0
    @inbounds for j in 1:n
        if keep_mask[j]
            pos += 1
            rho_kept[pos]          = rho[j]
            phi_kept[pos]          = phi[j]
            pt_slice_ids_kept[pos] = pt_slice_ids[j]
            indices_kept[pos]      = indices[j]
            old_to_new[j] = pos
        end
    end

    slice_point_indices_kept = [Int[] for _ in 1:n_slices]
    @inbounds for s in 1:n_slices
        for j in slice_point_indices[s]
            nj = old_to_new[j]
            nj > 0 && push!(slice_point_indices_kept[s], nj)
        end
    end

    return (rho_kept, phi_kept, pt_slice_ids_kept, slice_point_indices_kept, indices_kept)
end

# ───────────────────────────────────────────────────────────────────────────────
# 2c. 2D periodic surface smoothing & integration
# ───────────────────────────────────────────────────────────────────────────────

"""
    _method_spline_2d(rho, phi, pt_slice_ids, n_slices, cfg)
        -> Vector{NamedTuple{(:cross_area, :circumference, :completeness)}}

Compute cross-sectional area and circumference for all slices of an NBS
using 2D surface smoothing (periodic in phi, non-periodic in z).
Returns a vector indexed by slice (1:n_slices); slices with no data
have zeros. The angular bin count adapts to NBS size via the median of `rho`.
"""
function _method_spline_2d(rho::Vector{Float64}, phi::Vector{Float64},
                           pt_slice_ids::Vector{Int}, n_slices::Int,
                           cfg::FLiPConfig)
    T = NamedTuple{(:cross_area, :circumference, :completeness), Tuple{Float64, Float64, Float64}}
    results = Vector{T}(undef, n_slices)
    fill!(results, (cross_area=0.0, circumference=0.0, completeness=0.0))

    rho_median_global = isempty(rho) ? 0.01 : median(rho)
    surface_res = cfg.qsm_surface_res_scalar * cfg.pipeline_subsample_res
    phi_bin_num = clamp(ceil(Int, 2π * rho_median_global / surface_res),
                        cfg.qsm_phi_bin_min, cfg.qsm_phi_bin_max)
    dphi = 2π / phi_bin_num

    # Build 2D surface (rho already pre-filtered upstream via cfg.qsm_rho_percentile)
    surface = _build_rho_surface(rho, phi, pt_slice_ids, n_slices, phi_bin_num)

    # Compute completeness per slice before gap-filling
    completeness = Vector{Float64}(undef, n_slices)
    @inbounds for s in 1:n_slices
        n_filled = count(b -> isfinite(surface[b, s]), 1:phi_bin_num)
        completeness[s] = n_filled / phi_bin_num
    end

    # Fill gaps and smooth
    _fill_gaps_2d!(surface)
    _smooth_surface_2d!(surface, 0.5, cfg.qsm_spl_z_smoothing)

    # Extract per-slice metrics
    @inbounds for s in 1:n_slices
        completeness[s] <= 0 && continue

        # Central differences for derivative (periodic)
        circ = 0.0
        area = 0.0
        for b in 1:phi_bin_num
            r = surface[b, s]
            isfinite(r) || continue
            bm = mod1(b - 1, phi_bin_num)
            bp = mod1(b + 1, phi_bin_num)
            dr = (surface[bp, s] - surface[bm, s]) / (2.0 * dphi)
            circ += sqrt(r^2 + dr^2) * dphi
            area += 0.5 * r^2 * dphi
        end
        results[s] = (cross_area=area, circumference=circ, completeness=completeness[s])
    end

    return (results, surface, phi_bin_num)
end

"""
    _build_rho_surface(rho, phi, pt_slice_ids, n_slices, phi_bin_num)
        -> Matrix{Float64}  # (phi_bin_num, n_slices)

Bin all NBS points into a 2D grid of rho values; each cell is the arithmetic
mean of all rho values falling into it, empty cells are NaN. Rho-outlier
filtering happens upstream in `_filter_rho_outliers` (driven by
`cfg.qsm_rho_percentile`), so this routine has no percentile knob of its own.
"""
function _build_rho_surface(rho::Vector{Float64}, phi::Vector{Float64},
                            pt_slice_ids::Vector{Int}, n_slices::Int,
                            phi_bin_num::Int)
    dphi = 2π / phi_bin_num
    bin_sum = zeros(phi_bin_num, n_slices)
    bin_count = zeros(Int, phi_bin_num, n_slices)

    @inbounds for j in eachindex(rho)
        b = clamp(floor(Int, (phi[j] + π) / dphi) + 1, 1, phi_bin_num)
        s = pt_slice_ids[j]
        bin_sum[b, s] += rho[j]
        bin_count[b, s] += 1
    end

    surface = fill(NaN, phi_bin_num, n_slices)
    @inbounds for s in 1:n_slices, b in 1:phi_bin_num
        if bin_count[b, s] > 0
            surface[b, s] = bin_sum[b, s] / bin_count[b, s]
        end
    end
    return surface
end

"""
    _fill_gaps_2d!(surface)

Fill NaN cells in the 2D surface via nearest-neighbor interpolation.
Phi direction (axis 1) wraps periodically; z direction (axis 2) clamps.
"""
function _fill_gaps_2d!(surface::Matrix{Float64})
    nphi, nz = size(surface)
    max_search = max(nphi ÷ 2, nz)

    @inbounds for s in 1:nz, b in 1:nphi
        isnan(surface[b, s]) || continue

        # Search 4 cardinal directions for nearest non-NaN neighbor
        wsum = 0.0
        rsum = 0.0

        # phi+ direction (periodic)
        for offset in 1:max_search
            bp = mod1(b + offset, nphi)
            if isfinite(surface[bp, s])
                w = 1.0 / offset
                wsum += w
                rsum += w * surface[bp, s]
                break
            end
        end
        # phi- direction (periodic)
        for offset in 1:max_search
            bm = mod1(b - offset, nphi)
            if isfinite(surface[bm, s])
                w = 1.0 / offset
                wsum += w
                rsum += w * surface[bm, s]
                break
            end
        end
        # z+ direction (clamped)
        for offset in 1:(nz - s)
            if isfinite(surface[b, s + offset])
                w = 1.0 / offset
                wsum += w
                rsum += w * surface[b, s + offset]
                break
            end
        end
        # z- direction (clamped)
        for offset in 1:(s - 1)
            if isfinite(surface[b, s - offset])
                w = 1.0 / offset
                wsum += w
                rsum += w * surface[b, s - offset]
                break
            end
        end

        if wsum > 0
            surface[b, s] = rsum / wsum
        end
    end
end

"""
    _smooth_surface_2d!(surface, s_phi, s_z, n_passes=1)

Separable 3-point stencil smoothing of the rho surface.
Periodic in phi (axis 1), Neumann boundary in z (axis 2).
"""
function _smooth_surface_2d!(surface::Matrix{Float64},
                              s_phi::Float64, s_z::Float64,
                              n_passes::Int=1)
    nphi, nz = size(surface)
    buf = similar(surface)

    for _ in 1:n_passes
        # Phi pass (periodic)
        w_center_phi = 1.0 - s_phi
        w_side_phi = s_phi / 2.0
        @inbounds for s in 1:nz, b in 1:nphi
            bm = mod1(b - 1, nphi)
            bp = mod1(b + 1, nphi)
            buf[b, s] = w_center_phi * surface[b, s] +
                         w_side_phi * (surface[bm, s] + surface[bp, s])
        end

        # Z pass (Neumann boundary)
        if nz == 1
            copyto!(surface, buf)
        else
            w_center_z = 1.0 - s_z
            w_side_z = s_z / 2.0
            @inbounds for s in 1:nz, b in 1:nphi
                if s == 1
                    surface[b, s] = w_center_z * buf[b, s] + s_z * buf[b, 2]
                elseif s == nz
                    surface[b, s] = w_center_z * buf[b, s] + s_z * buf[b, nz - 1]
                else
                    surface[b, s] = w_center_z * buf[b, s] +
                                     w_side_z * (buf[b, s - 1] + buf[b, s + 1])
                end
            end
        end
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# 2d. Surface point cloud generation
# ───────────────────────────────────────────────────────────────────────────────

"""
    _generate_surface_points(surface, centers, e1, e2, slice_res, gen_res)
        -> (Matrix{Float64}, Vector{Float64})  # (M×3 coords, M surface rho values)

Convert a smoothed 2D rho surface (phi × z-slice) to 3D xyz points.
Linearly interpolates between z-slices to achieve approximately `gen_res`
axial spacing. Returns coordinates and the per-point surface radius (rho),
which equals the smoothed surface value at each generated point (slice points
take `surface[b, s]`; inter-slice points take the linear blend between
slices s and s+1).
"""
function _generate_surface_points(surface::Matrix{Float64},
                                   centers::Matrix{Float64},
                                   e1::NTuple{3,Float64},
                                   e2::NTuple{3,Float64},
                                   slice_res::Float64,
                                   gen_res::Float64)
    nphi, nslices = size(surface)
    dphi = 2π / nphi
    n_zsub = max(1, ceil(Int, slice_res / gen_res))

    # Upper bound for pre-allocation
    n_est = nphi * (nslices + max(0, nslices - 1) * (n_zsub - 1))
    pts = Matrix{Float64}(undef, n_est, 3)
    rho_vals = Vector{Float64}(undef, n_est)
    idx = 0

    @inbounds for s in 1:nslices
        # Points at slice center
        for b in 1:nphi
            rho = surface[b, s]
            (isfinite(rho) && rho > 0) || continue
            phi_val = -π + (b - 0.5) * dphi
            u = rho * cos(phi_val)
            v = rho * sin(phi_val)
            idx += 1
            pts[idx, 1] = centers[s, 1] + u * e1[1] + v * e2[1]
            pts[idx, 2] = centers[s, 2] + u * e1[2] + v * e2[2]
            pts[idx, 3] = centers[s, 3] + u * e1[3] + v * e2[3]
            rho_vals[idx] = rho
        end

        # Interpolated points between slice s and s+1
        if s < nslices && n_zsub > 1
            for k in 1:(n_zsub - 1)
                frac = k / n_zsub
                icx = centers[s, 1] + frac * (centers[s+1, 1] - centers[s, 1])
                icy = centers[s, 2] + frac * (centers[s+1, 2] - centers[s, 2])
                icz = centers[s, 3] + frac * (centers[s+1, 3] - centers[s, 3])
                for b in 1:nphi
                    rho1 = surface[b, s]
                    rho2 = surface[b, s+1]
                    (isfinite(rho1) && isfinite(rho2) && rho1 > 0 && rho2 > 0) || continue
                    rho = rho1 + frac * (rho2 - rho1)
                    phi_val = -π + (b - 0.5) * dphi
                    u = rho * cos(phi_val)
                    v = rho * sin(phi_val)
                    idx += 1
                    pts[idx, 1] = icx + u * e1[1] + v * e2[1]
                    pts[idx, 2] = icy + u * e1[2] + v * e2[2]
                    pts[idx, 3] = icz + u * e1[3] + v * e2[3]
                    rho_vals[idx] = rho
                end
            end
        end
    end

    if idx > 0
        return (pts[1:idx, :], rho_vals[1:idx])
    else
        return (Matrix{Float64}(undef, 0, 3), Float64[])
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# 2e. Frustum geometry
# ───────────────────────────────────────────────────────────────────────────────

"""Compute frustum volume and surface area between consecutive nodes."""
function _frustum_metrics(r1::Float64, r2::Float64, h::Float64)
    vol = (π / 3.0) * h * (r1^2 + r2^2 + r1 * r2)
    sa = π * (r1 + r2) * sqrt(h^2 + (r1 - r2)^2)
    return (vol, sa)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Output aggregation & I/O
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _build_node_table(nodes) -> (columns, headers, vol, sa)

Build the node-level results as a vector of typed per-column vectors (each
element of `columns` is one column, in `headers` order). Also returns the
per-node frustum `vol` and `sa` vectors directly so `_build_tree_table` can
aggregate them without round-tripping through a stringly-typed lookup.
"""
function _build_node_table(nodes::Vector{QSMNode})
    isempty(nodes) && return (AbstractVector[], String[], Float64[], Float64[])

    # Group by NBS for frustum computation (NBS ids are dense from tree_segmentation)
    nbs_ids_per_node = Int32[nd.nbs_id for nd in nodes]
    nbs_groups = group_indices_by_label(1:length(nodes), nbs_ids_per_node)

    n = length(nodes)
    vol = zeros(n)
    sa = zeros(n)

    for idxs in nbs_groups
        sort!(idxs; by=i -> nodes[i].agh)
        nn = length(idxs)
        for k in 1:nn
            nd = nodes[idxs[k]]
            h = nd.height
            if nn == 1 || k == 1 || k == nn
                r1 = nd.radius_area
                if k == 1 && nn > 1
                    r2 = nodes[idxs[k+1]].radius_area
                    vol[idxs[k]], sa[idxs[k]] = _frustum_metrics(r1, r2, h)
                elseif k == nn && nn > 1
                    r0 = nodes[idxs[k-1]].radius_area
                    vol[idxs[k]], sa[idxs[k]] = _frustum_metrics(r0, r1, h)
                else
                    vol[idxs[k]], sa[idxs[k]] = _frustum_metrics(r1, r1, h)
                end
            else
                r0 = nodes[idxs[k-1]].radius_area
                r1 = nd.radius_area
                r2 = nodes[idxs[k+1]].radius_area
                ra = (r0 + r1) / 2
                rb = (r1 + r2) / 2
                vol[idxs[k]], sa[idxs[k]] = _frustum_metrics(ra, rb, h)
            end
        end
    end

    headers = [
        "qsm_node_id", "tree_nbs_id", "tree_id", "agh",
        "cross_area", "circumference",
        "radius_area", "radius_circ",
        "height", "volume", "surface_area",
        "completeness", "n_points",
        "center_x", "center_y", "center_z",
        "direction_x", "direction_y", "direction_z",
    ]

    columns = AbstractVector[
        Int[nd.qsm_node_id for nd in nodes],
        Int32[nd.nbs_id for nd in nodes],
        Int32[nd.tree_id for nd in nodes],
        Float64[nd.agh for nd in nodes],
        Float64[nd.cross_area for nd in nodes],
        Float64[nd.circumference for nd in nodes],
        Float64[nd.radius_area for nd in nodes],
        Float64[nd.radius_circ for nd in nodes],
        Float64[nd.height for nd in nodes],
        vol,
        sa,
        Float64[nd.completeness for nd in nodes],
        Int[nd.n_points for nd in nodes],
        Float64[nd.center_x for nd in nodes],
        Float64[nd.center_y for nd in nodes],
        Float64[nd.center_z for nd in nodes],
        Float64[nd.direction_x for nd in nodes],
        Float64[nd.direction_y for nd in nodes],
        Float64[nd.direction_z for nd in nodes],
    ]

    return (columns, headers, vol, sa)
end

"""
    _build_tree_table(nodes, vol, sa, cfg) -> (columns, headers)

Aggregate node-level results to tree-level biometrics. `vol` and `sa` are the
per-node frustum volume / surface area vectors returned by `_build_node_table`
(parallel to `nodes`).
"""
function _build_tree_table(nodes::Vector{QSMNode},
                           vol::Vector{Float64}, sa::Vector{Float64},
                           cfg::FLiPConfig)
    isempty(nodes) && return (AbstractVector[], String[])

    # Group by tree_id (dense ids from tree_segmentation); output ordered by tree_id ascending
    tree_ids_per_node = Int32[nd.tree_id for nd in nodes]
    tree_groups = group_indices_by_label(1:length(nodes), tree_ids_per_node)
    n_trees = length(tree_groups)
    bh = cfg.qsm_breast_height

    tree_headers = [
        "tree_id",
        "volume", "surface_area", "height",
        "dbh_area", "dbh_circ",
        "n_points", "n_nodes",
        "x", "y",
    ]

    col_tree_id  = Vector{Int32}(undef, n_trees)
    col_volume   = Vector{Float64}(undef, n_trees)
    col_surface  = Vector{Float64}(undef, n_trees)
    col_height   = Vector{Float64}(undef, n_trees)
    col_dbh_a    = Vector{Float64}(undef, n_trees)
    col_dbh_c    = Vector{Float64}(undef, n_trees)
    col_n_points = Vector{Int}(undef, n_trees)
    col_n_nodes  = Vector{Int}(undef, n_trees)
    col_x        = Vector{Float64}(undef, n_trees)
    col_y        = Vector{Float64}(undef, n_trees)

    for (ti, idxs) in enumerate(tree_groups)
        total_vol = sum(i -> vol[i], idxs)
        total_sa  = sum(i -> sa[i],  idxs)
        total_pts = sum(i -> nodes[i].n_points, idxs)
        max_agh   = maximum(i -> nodes[i].agh, idxs)

        # DBH: find node closest to breast height
        best_bh_idx = idxs[1]
        best_bh_dist = abs(nodes[idxs[1]].agh - bh)
        for i in idxs
            d = abs(nodes[i].agh - bh)
            if d < best_bh_dist
                best_bh_dist = d
                best_bh_idx = i
            end
        end

        col_tree_id[ti]  = nodes[idxs[1]].tree_id
        col_volume[ti]   = total_vol
        col_surface[ti]  = total_sa
        col_height[ti]   = max_agh
        col_dbh_a[ti]    = 2.0 * nodes[best_bh_idx].radius_area
        col_dbh_c[ti]    = 2.0 * nodes[best_bh_idx].radius_circ
        col_n_points[ti] = total_pts
        col_n_nodes[ti]  = length(idxs)
        col_x[ti]        = nodes[best_bh_idx].center_x
        col_y[ti]        = nodes[best_bh_idx].center_y
    end

    columns = AbstractVector[
        col_tree_id, col_volume, col_surface, col_height,
        col_dbh_a, col_dbh_c, col_n_points, col_n_nodes,
        col_x, col_y,
    ]
    return (columns, tree_headers)
end

