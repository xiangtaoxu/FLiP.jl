"""
NBS geometric modeling: fit a per-node cylinder model for each NBS of a labeled point cloud
(the cfg-driven fitting FLOW: linearity filter, per-NBS driver, slicing + per-slice QC +
centerline). Pure kernels live in `util/nbs_utils.jl`; aggregation/IO in `report/`.
(Public entry currently `qsm`; renamed to `model_nbs` later.)
"""

# 8-way angular partition used in slice angular-coverage checks
const OCTANT_WIDTH = π / 4

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

"""
    model_nbs(; pc, group_attr=:tree_nbs_id, node_id_attr=:node_id, emit_surface=false, cfg=_CFG)
        -> (status, nodes, surface_parts)

Fit a per-node cylinder model (`QSMNode`) for each linear NBS of a labeled point cloud,
grouping points by `group_attr`, and stamp the per-point `node_id_attr` (z-slice id). Returns
the fitted `nodes` and, when `emit_surface`, the per-NBS surface parts for `report/` to
assemble into a surface cloud.

This function only MODELS — it does no biometric aggregation, CSV writing, or surface-cloud
assembly (those are reporting; see `report/biometrics.jl` / `report/surface_cloud.jl`). It is
run twice: a fit-only trial inside `tree_segmentation` (`group_attr=:nbs_id`,
`node_id_attr=:trial_node_id`, `emit_surface=false`) and the final pass on the assembled cloud
(`group_attr=:tree_nbs_id`, `node_id_attr=:node_id`, `emit_surface=true`).

`status` is `:success`, `:no_linear_nbs`, or `:no_data`. `surface_parts` is a NamedTuple
`(coords, nbs, rho, T)` of the per-NBS surface arrays (empty unless `emit_surface`).
"""
function model_nbs(; pc::Union{Nothing,PointCloud}=nothing,
                     group_attr::Symbol=:tree_nbs_id, node_id_attr::Symbol=:node_id,
                     emit_surface::Bool=false, cfg::FLiPConfig=_CFG)
    Tc = pc === nothing ? Float64 : eltype(coordinates(pc))
    empty_parts = (coords=Matrix{Tc}[], nbs=Vector{Int32}[], rho=Vector{Tc}[], T=Tc)
    if pc === nothing || npoints(pc) == 0
        @warn "$_LOG_PREFIX model_nbs: no point cloud data available"
        return (status=:no_data, nodes=QSMNode[], surface_parts=empty_parts)
    end

    coords = coordinates(pc); N = size(coords, 1)
    # `group_attr` is `:tree_nbs_id` (final, post-assembly) or `:nbs_id` (pre-assembly trial).
    tree_ids = hasattribute(pc, :tree_id) ? getattribute(pc, :tree_id) : zeros(Int32, N)
    group_ids = hasattribute(pc, group_attr) ? getattribute(pc, group_attr) : zeros(Int32, N)
    agh_values = hasattribute(pc, :AGH) ? getattribute(pc, :AGH) : zeros(Float64, N)

    linear_nbs = _filter_linear_nbs(coords, group_ids, cfg)
    n_linear = count(!isnothing, linear_nbs)
    @info "$_LOG_PREFIX   modeling $N points, $n_linear linear NBS (linearity_threshold=$(cfg.tree.model.nbs_linearity_threshold))"
    if n_linear == 0
        setattribute!(pc, node_id_attr, zeros(Int32, N))
        return (status=:no_linear_nbs, nodes=QSMNode[], surface_parts=empty_parts)
    end

    nodes = QSMNode[]; sizehint!(nodes, n_linear * 10); next_node_id = 1
    node_ids = zeros(Int32, N)
    # Surface parts are kept at the cloud's own precision (`Tc`); QSM math stays in Float64.
    surface_parts = Matrix{Tc}[]; surface_nbs_parts = Vector{Int32}[]; surface_rho_parts = Vector{Tc}[]
    progress = ProgressReporter("modeling NBS", n_linear); n_done = 0

    for nid in 1:length(linear_nbs)
        info = linear_nbs[nid]
        info === nothing && continue
        nid32 = Int32(nid)
        # `lean = !emit_surface`: the trial pass skips surface-point generation.
        next_node_id, surf_pts, surf_rho = _process_single_nbs!(nodes, node_ids, coords, info, nid32,
                                            tree_ids, agh_values, cfg, next_node_id; lean=!emit_surface)
        if emit_surface && size(surf_pts, 1) > 0
            push!(surface_parts, surf_pts)
            push!(surface_nbs_parts, fill(nid32, size(surf_pts, 1)))
            push!(surface_rho_parts, surf_rho)
        end
        n_done += 1; report!(progress, n_done)
    end
    @info "$_LOG_PREFIX   $(length(nodes)) nodes modeled"
    setattribute!(pc, node_id_attr, node_ids)

    parts = emit_surface ? (coords=surface_parts, nbs=surface_nbs_parts, rho=surface_rho_parts, T=Tc) : empty_parts
    return (status=:success, nodes=nodes, surface_parts=parts)
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
util/pca.jl) and adds the QSM-specific tree-stem orientation: PC1 is
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
    # Embarrassingly parallel by NBS id: writes to distinct `result[nid]`.
    # `pca_linearity` allocates a local 3×3 covariance per call; no shared
    # mutable state. `nbs_groups` is fully populated above and read-only here.
    _parallel_for(K, effective_nthreads(cfg)) do nid
        indices = nbs_groups[nid]
        isempty(indices) && return
        pca = pca_linearity(coords, indices, cfg.tree.model.nbs_linearity_threshold)
        pca === nothing && return

        # Orient PC1 by z (tree-stem convention: PC1 from low-z to high-z)
        d = pca.direction
        c = pca.center
        z_min_i = indices[1]; z_max_i = indices[1]
        @inbounds for i in indices
            if coords[i, 3] < coords[z_min_i, 3]; z_min_i = i; end
            if coords[i, 3] > coords[z_max_i, 3]; z_max_i = i; end
        end
        @inbounds proj_low  = (coords[z_min_i, 1] - c[1]) * d[1] +
                              (coords[z_min_i, 2] - c[2]) * d[2] +
                              (coords[z_min_i, 3] - c[3]) * d[3]
        @inbounds proj_high = (coords[z_max_i, 1] - c[1]) * d[1] +
                              (coords[z_max_i, 2] - c[2]) * d[2] +
                              (coords[z_max_i, 3] - c[3]) * d[3]
        dvec = proj_high < proj_low ? (-d[1], -d[2], -d[3]) : d

        @inbounds result[nid] = NBSInfo(dvec, pca.center, pca.eigenvalues, pca.linearity, indices)
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
                              next_node_id::Int;
                              lean::Bool=false)
    qsm = cfg.tree.model                                          # local alias for terseness
    slice_res = qsm.slice_height_scalar * cfg.pipeline.subsample_res

    # 2a: Slice, fit centers, interpolate, smooth. Per-slice QC inside
    # _slice_and_fit_centers may drop points; `indices` is rebound here to the
    # compacted survivor list (positions into `coords`), and pt_slice_ids /
    # slice_point_indices are renumbered to match. The returned `centers`
    # already has NaN slices interpolated and moving-window smoothing applied
    # (both done inside `_finalize_centerline!`). Dropped points keep
    # qsm_node_id = 0 by default.
    centers, slice_point_indices, _, _, pt_slice_ids, e1, e2, indices =
        _slice_and_fit_centers(coords, info, slice_res, qsm.min_node_size,
                               qsm.min_octant_taubin, cfg)

    # 2b: Unroll points to (rho, phi)
    rho, phi = _unroll_points(coords, indices, centers, pt_slice_ids, e1, e2)

    # 2b': Drop per-slice rho outliers before they touch any downstream stage
    # (compacts indices / pt_slice_ids / slice_point_indices / rho / phi together).
    n_slices = size(centers, 1)
    rho, phi, pt_slice_ids, slice_point_indices, indices =
        _filter_rho_outliers(rho, phi, pt_slice_ids, slice_point_indices, indices,
                             n_slices, qsm.rho_percentile)

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
        n_pts < qsm.min_node_size && continue

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
        completeness < qsm.completeness_threshold && continue

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
    # the per-point surface radius as an attribute alongside coordinates. The lean
    # trial pass only needs the fitted node cylinders, so it skips this entirely.
    if lean
        surface_pts = Matrix{eltype(coords)}(undef, 0, 3)
        surface_rho = eltype(coords)[]
    else
        gen_res = cfg.pipeline.subsample_res
        surface_pts, surface_rho = _generate_surface_points(surface_grid, centers, e1, e2, slice_res, gen_res, eltype(coords))
    end

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
center per slice. QC is governed by `cfg.tree.model.qc_*` fields; when
`cfg.tree.model.qc_enable` is `false` it is a no-op.

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
    qc_enable  = cfg.tree.model.qc_enable
    cc_radius  = QC_CC_RADIUS_SCALAR * cfg.pipeline.subsample_res
    cc_min     = cfg.tree.model.min_node_size
    cont_ratio = cfg.tree.model.qc_continuity_ratio
    sor_k      = cfg.statistical_filter.k_neighbors
    sor_ns     = cfg.statistical_filter.n_sigma

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

