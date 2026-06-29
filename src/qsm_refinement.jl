"""
Post-QSM refinement for FLiP.jl.

Umbrella for methods that improve QSM results after a first QSM pass. The current
(and only) method is [`nbs_merge_by_volume_overlap`](@ref): it detects
Non-Branching Segments (NBS, keyed by `tree_nbs_id`) whose fitted QSM cylinders
overlap in 3-D and merges them by rewriting `:tree_nbs_id` / `:tree_id` on the
cloud. The pipeline then re-runs QSM on the relabeled cloud so each merged
segment is re-fit (re-PCA, re-sliced) as one continuous NBS.

Determinism: overlap volume is estimated on a FIXED global voxel lattice (see
`voxelized_cylinder_volume`), candidate pairs are deduplicated in a sorted set,
and merge edges are applied in sorted order — so a single in-session run is
reproducible and independent of thread scheduling, matching FLiP's no-random-seed
guarantee. (Resuming refinement purely from disk is best-effort: the tree cloud
and node CSV are quantized on write — see `_read_qsm_nodes_csv` — so a borderline
pair near a gate threshold could resolve differently than the in-session run. Run
refinement in the same session as QSM, the default, for exact reproducibility.)

Known limitation: pure volume overlap cannot bridge *gap-separated collinear*
segments (no shared volume ⇒ no merge). A future endpoint-proximity /
axis-collinearity method could live here under the same umbrella.
"""

using NearestNeighbors: KDTree, inrange

# A finite cylinder from one QSM node: midpoint, unit axis, radius, half-height.
const Cyl = NamedTuple{(:center, :axis, :radius, :half_height),
                       Tuple{NTuple{3,Float64}, NTuple{3,Float64}, Float64, Float64}}

"""Per-segment (tree_nbs) cylinder model assembled from its QSM nodes."""
struct SegModel
    seg_id::Int32                 # tree_nbs_id (== QSMNode.nbs_id in QSM output)
    tree_id::Int32
    cyls::Vector{Cyl}
    aabb::NTuple{6,Float64}       # union of cylinder AABBs
    vol_vox::Float64             # deterministic self-volume on the global lattice
    n_points::Int                # Σ node n_points
    completeness::Float64        # mean node completeness
    min_agh::Float64             # min node AGH (for grounded-trunk detection)
end

@inline function _unit3(x::Float64, y::Float64, z::Float64)
    n = sqrt(x * x + y * y + z * z)
    return n > 0 ? (x / n, y / n, z / n) : (0.0, 0.0, 1.0)
end

@inline _aabb_intersection(a::NTuple{6,Float64}, b::NTuple{6,Float64}) =
    (max(a[1], b[1]), min(a[2], b[2]),
     max(a[3], b[3]), min(a[4], b[4]),
     max(a[5], b[5]), min(a[6], b[6]))

@inline function _point_in_any(p::NTuple{3,Float64}, cyls::Vector{Cyl})
    @inbounds for c in cyls
        point_in_cylinder(p, c.center, c.axis, c.radius, c.half_height) && return true
    end
    return false
end

"""
    _voxel_intersection_volume(cyls_a, cyls_b, box, voxel_res) -> Float64

Deterministic volume of the 3-D intersection of two cylinder unions, restricted
to `box`, on the same global lattice as `voxelized_cylinder_volume` (a voxel
counts iff its center is inside some cylinder of `cyls_a` AND some cylinder of
`cyls_b`). Because the lattice is shared, the result is ≤ each segment's
self-volume, so the overlap ratio is ≤ 1.
"""
function _voxel_intersection_volume(cyls_a::Vector{Cyl}, cyls_b::Vector{Cyl},
                                    box::NTuple{6,Float64}, voxel_res::Float64)
    (!(voxel_res > 0) || isempty(cyls_a) || isempty(cyls_b)) && return 0.0
    kx0 = floor(Int, box[1] / voxel_res); kx1 = ceil(Int, box[2] / voxel_res) - 1
    ky0 = floor(Int, box[3] / voxel_res); ky1 = ceil(Int, box[4] / voxel_res) - 1
    kz0 = floor(Int, box[5] / voxel_res); kz1 = ceil(Int, box[6] / voxel_res) - 1
    cnt = 0
    @inbounds for kx in kx0:kx1
        cx = (kx + 0.5) * voxel_res
        for ky in ky0:ky1
            cy = (ky + 0.5) * voxel_res
            for kz in kz0:kz1
                cz = (kz + 0.5) * voxel_res
                p = (cx, cy, cz)
                (_point_in_any(p, cyls_a) && _point_in_any(p, cyls_b)) && (cnt += 1)
            end
        end
    end
    return cnt * voxel_res^3
end

"""
    _build_seg_models(nodes, voxel_res, nthread) -> Vector{SegModel}

Group QSM nodes by `tree_nbs_id` and build one `SegModel` per segment: a finite
cylinder per node (`radius_area`, axis = normalized PC1 direction, half-height =
node height / 2), the union AABB, the deterministic self-volume, total points,
mean completeness, and min AGH. Nodes with non-finite/≤0 radius are skipped;
segments with no valid cylinders or zero volume are dropped.
"""
function _build_seg_models(nodes::Vector{QSMNode}, voxel_res::Float64, nthread::Integer)
    isempty(nodes) && return SegModel[]
    groups = Dict{Int32,Vector{Int}}()
    @inbounds for (i, nd) in enumerate(nodes)
        nd.nbs_id > 0 || continue
        push!(get!(groups, nd.nbs_id, Int[]), i)
    end
    seg_ids = sort!(collect(keys(groups)))

    # Assemble cylinders + aggregates serially (cheap), defer volume to a parallel pass.
    raw = Vector{Tuple{Int32,Int32,Vector{Cyl},NTuple{6,Float64},Int,Float64,Float64}}()
    for sid in seg_ids
        idxs = groups[sid]
        cyls = Cyl[]
        tot_pts = 0
        sum_comp = 0.0
        min_agh = Inf
        tid = nodes[idxs[1]].tree_id
        bb = (Inf, -Inf, Inf, -Inf, Inf, -Inf)
        for ix in idxs
            nd = nodes[ix]
            tot_pts += nd.n_points
            sum_comp += nd.completeness
            min_agh = min(min_agh, nd.agh)
            r = nd.radius_area
            (isfinite(r) && r > 0) || continue
            ax = _unit3(nd.direction_x, nd.direction_y, nd.direction_z)
            c = (center = (nd.center_x, nd.center_y, nd.center_z),
                 axis = ax, radius = r, half_height = nd.height / 2)::Cyl
            push!(cyls, c)
            cb = cylinder_aabb(c.center, c.axis, c.radius, c.half_height)
            bb = (min(bb[1], cb[1]), max(bb[2], cb[2]),
                  min(bb[3], cb[3]), max(bb[4], cb[4]),
                  min(bb[5], cb[5]), max(bb[6], cb[6]))
        end
        isempty(cyls) && continue
        push!(raw, (sid, tid, cyls, bb, tot_pts, sum_comp / length(idxs), min_agh))
    end
    isempty(raw) && return SegModel[]

    vols = zeros(Float64, length(raw))
    _parallel_for(length(raw), nthread) do i
        r = raw[i]
        vols[i] = voxelized_cylinder_volume(r[3], r[4], voxel_res)
    end

    models = SegModel[]
    for (i, r) in enumerate(raw)
        vols[i] > 0 || continue
        push!(models, SegModel(r[1], r[2], r[3], r[4], vols[i], r[5], r[6], r[7]))
    end
    return models
end

"""
    _candidate_pairs(models, cfg) -> Vector{Tuple{Int32,Int32}}

Generate cross-segment candidate pairs whose cylinders could overlap, via a
KDTree over all node centers. For node `i` (reach `rᵢ + hᵢ/2`), any node `j` of a
different segment that could belong to an overlapping cylinder satisfies
`‖cᵢ−cⱼ‖ ≤ reachᵢ + Rmax`, where `Rmax = max reach`. Returns sorted unique
unordered pairs. O(M·k), avoiding the O(K²) all-pairs scan.
"""
function _candidate_pairs(models::Vector{SegModel}, cfg::FLiPConfig)
    M = sum(m -> length(m.cyls), models; init=0)
    M == 0 && return Tuple{Int32,Int32}[]
    centers = Matrix{Float64}(undef, 3, M)
    owner = Vector{Int32}(undef, M)
    reach = Vector{Float64}(undef, M)
    k = 0
    for m in models
        for c in m.cyls
            k += 1
            centers[1, k] = c.center[1]; centers[2, k] = c.center[2]; centers[3, k] = c.center[3]
            owner[k] = m.seg_id
            reach[k] = c.radius + c.half_height
        end
    end
    Rmax = maximum(reach)
    margin = cfg.qsm_refinement.candidate_radius_scalar * cfg.pipeline.subsample_res
    tree = KDTree(centers)
    pairset = Set{Tuple{Int32,Int32}}()
    q = Vector{Float64}(undef, 3)
    @inbounds for i in 1:M
        q[1] = centers[1, i]; q[2] = centers[2, i]; q[3] = centers[3, i]
        idxs = inrange(tree, q, reach[i] + Rmax + margin)
        oi = owner[i]
        for j in idxs
            oj = owner[j]
            oj == oi && continue
            lo = oi < oj ? oi : oj
            hi = oi < oj ? oj : oi
            push!(pairset, (lo, hi))
        end
    end
    pairs = collect(pairset)
    sort!(pairs)
    return pairs
end

"""
    _merge_reason(A, B, ratio, cfg) -> Symbol

Decide whether segments `A` and `B` are merge-eligible. Returns `:ok`, or the
first failing gate: `:below_overlap_threshold`, `:completeness_gate`,
`:min_points_gate`, `:absorb_guard` (the larger-volume "absorber" is too sparse
relative to the segment it would swallow), `:cross_tree_disabled`, or
`:grounded_trunk_guard` (both segments reach the ground).
"""
function _merge_reason(A::SegModel, B::SegModel, ratio::Float64, cfg::FLiPConfig)
    rc = cfg.qsm_refinement
    ratio > rc.overlap_threshold || return :below_overlap_threshold
    (A.completeness >= rc.completeness_gate && B.completeness >= rc.completeness_gate) ||
        return :completeness_gate
    (A.n_points >= rc.min_points_gate && B.n_points >= rc.min_points_gate) ||
        return :min_points_gate
    absorber, absorbed = A.vol_vox >= B.vol_vox ? (A, B) : (B, A)
    absorber.n_points >= rc.absorb_min_point_ratio * absorbed.n_points || return :absorb_guard
    if A.tree_id != B.tree_id
        rc.cross_tree || return :cross_tree_disabled
        if rc.protect_grounded_trunks
            ground = cfg.tree_segmentation.nearground_agh_threshold
            (A.min_agh <= ground && B.min_agh <= ground) && return :grounded_trunk_guard
        end
    end
    return :ok
end

function _write_merge_report(path::String, pairs::Vector{Tuple{Int32,Int32}},
                             models::Vector{SegModel}, segidx::Dict{Int32,Int},
                             inter::Vector{Float64}, ratio::Vector{Float64},
                             reasons::Vector{Symbol}, merged_flag::BitVector)
    np = length(pairs)
    tnbs_a = Vector{Int32}(undef, np); tnbs_b = Vector{Int32}(undef, np)
    tid_a  = Vector{Int32}(undef, np); tid_b  = Vector{Int32}(undef, np)
    vol_a  = Vector{Float64}(undef, np); vol_b = Vector{Float64}(undef, np)
    npa    = Vector{Int}(undef, np); npb = Vector{Int}(undef, np)
    cmpa   = Vector{Float64}(undef, np); cmpb = Vector{Float64}(undef, np)
    cross  = Vector{Int}(undef, np)
    decision = Vector{String}(undef, np); reason = Vector{String}(undef, np)
    for pi in 1:np
        (a, b) = pairs[pi]
        A = models[segidx[a]]; B = models[segidx[b]]
        tnbs_a[pi] = A.seg_id; tnbs_b[pi] = B.seg_id
        tid_a[pi] = A.tree_id; tid_b[pi] = B.tree_id
        vol_a[pi] = A.vol_vox; vol_b[pi] = B.vol_vox
        npa[pi] = A.n_points; npb[pi] = B.n_points
        cmpa[pi] = A.completeness; cmpb[pi] = B.completeness
        cross[pi] = A.tree_id != B.tree_id ? 1 : 0
        decision[pi] = merged_flag[pi] ? "merged" : "rejected"
        reason[pi] = String(reasons[pi])
    end
    headers = ["tree_nbs_a", "tree_nbs_b", "tree_id_a", "tree_id_b",
               "vol_a", "vol_b", "inter_vol", "overlap_ratio",
               "n_points_a", "n_points_b", "completeness_a", "completeness_b",
               "cross_tree", "decision", "reason"]
    columns = AbstractVector[tnbs_a, tnbs_b, tid_a, tid_b, vol_a, vol_b, inter, ratio,
                             npa, npb, cmpa, cmpb, cross, decision, reason]
    _write_csv(path, columns, headers)
end

"""
    nbs_merge_by_volume_overlap(; pc, nodes, cfg=_CFG, output_dir="", output_prefix="output")
        -> NamedTuple

Detect tree_nbs segments whose fitted QSM cylinders overlap in 3-D and merge them
via union-find. In `mode == "apply"` the merge is applied to `pc` IN PLACE by
rewriting `:tree_nbs_id` / `:tree_id` (then re-densifying with
`relabel_by_occurrence` + `_relabel_tree_nbs_within_trees!`); in `"flag_only"`
the cloud is left untouched. A per-candidate-pair report CSV is always written
(when `output_dir` is set). Does NOT re-run QSM — the pipeline stage does that.

Returns `(status, n_segments_in, n_segments_out, n_groups_merged,
n_cross_tree_merges, pc_output, report_csv_path, merged)`, where `merged` is
`true` only when the cloud was actually relabeled (drives the second QSM pass).
"""
function nbs_merge_by_volume_overlap(; pc::PointCloud, nodes::Vector{QSMNode},
                                     cfg::FLiPConfig = _CFG,
                                     output_dir::AbstractString = "",
                                     output_prefix::AbstractString = "output")
    rc = cfg.qsm_refinement
    voxel_res = rc.voxel_res_scalar * cfg.pipeline.subsample_res
    nthread = effective_nthreads(cfg)
    report_path = isempty(output_dir) ? "" : joinpath(output_dir, "$(output_prefix)nbs_merge_report.csv")

    nores = (status=:no_nodes, n_segments_in=0, n_segments_out=0, n_groups_merged=0,
             n_cross_tree_merges=0, pc_output=pc, report_csv_path="", merged=false)

    (hasattribute(pc, :tree_nbs_id) && hasattribute(pc, :tree_id)) || begin
        @warn "$_LOG_PREFIX qsm_refinement: cloud lacks :tree_nbs_id/:tree_id; skipping"
        return nores
    end
    voxel_res > 0 || begin
        @warn "$_LOG_PREFIX qsm_refinement: non-positive voxel resolution; skipping"
        return nores
    end

    models = _build_seg_models(nodes, voxel_res, nthread)
    if isempty(models)
        @info "$_LOG_PREFIX   qsm_refinement: no QSM segments to evaluate"
        return nores
    end
    segidx = Dict{Int32,Int}()
    for (i, m) in enumerate(models)
        segidx[m.seg_id] = i
    end

    pairs = _candidate_pairs(models, cfg)
    npairs = length(pairs)

    # Overlap volume + ratio per candidate pair (parallel, disjoint slots).
    inter = zeros(Float64, npairs)
    ratio = zeros(Float64, npairs)
    _parallel_for(npairs, nthread) do pi
        (a, b) = pairs[pi]
        A = models[segidx[a]]; B = models[segidx[b]]
        if aabbs_overlap(A.aabb, B.aabb)
            box = _aabb_intersection(A.aabb, B.aabb)
            iv = _voxel_intersection_volume(A.cyls, B.cyls, box, voxel_res)
            mn = min(A.vol_vox, B.vol_vox)
            inter[pi] = iv
            ratio[pi] = mn > 0 ? iv / mn : 0.0
        end
    end

    reasons = Vector{Symbol}(undef, npairs)
    for pi in 1:npairs
        (a, b) = pairs[pi]
        reasons[pi] = _merge_reason(models[segidx[a]], models[segidx[b]], ratio[pi], cfg)
    end

    # Union-find over the tree_nbs id space (covers cloud ids and model ids).
    maxseg = 0
    tnbs_attr = getattribute(pc, :tree_nbs_id)
    @inbounds for v in tnbs_attr
        maxseg = max(maxseg, Int(v))
    end
    for m in models
        maxseg = max(maxseg, Int(m.seg_id))
    end
    parent = collect(1:maxseg)
    ranks = zeros(Int, maxseg)
    compsize = ones(Int, maxseg)
    merged_flag = falses(npairs)
    for pi in 1:npairs
        reasons[pi] === :ok || continue
        (a, b) = pairs[pi]
        ai = Int(a); bi = Int(b)
        ra = _uf_find!(parent, ai); rb = _uf_find!(parent, bi)
        if ra == rb
            merged_flag[pi] = true
            continue
        end
        if rc.max_group_size > 0 && compsize[ra] + compsize[rb] > rc.max_group_size
            reasons[pi] = :group_size_cap
            continue
        end
        _uf_union!(parent, ranks, ai, bi)
        nr = _uf_find!(parent, ai)
        compsize[nr] = compsize[ra] + compsize[rb]
        merged_flag[pi] = true
    end

    # Representative tree_id per group (densest segment wins) + stats.
    rep_tree = Dict{Int,Int32}()
    rep_pts = Dict{Int,Int}()
    rep_seg = Dict{Int,Int32}()
    root_members = Dict{Int,Vector{Int}}()
    for (i, m) in enumerate(models)
        r = _uf_find!(parent, Int(m.seg_id))
        push!(get!(root_members, r, Int[]), i)
        if !haskey(rep_pts, r) || m.n_points > rep_pts[r] ||
           (m.n_points == rep_pts[r] && m.seg_id < rep_seg[r])
            rep_pts[r] = m.n_points
            rep_seg[r] = m.seg_id
            rep_tree[r] = m.tree_id
        end
    end
    n_groups_merged = 0
    n_cross = 0
    for (_, mem) in root_members
        length(mem) >= 2 || continue
        n_groups_merged += 1
        length(unique(models[i].tree_id for i in mem)) > 1 && (n_cross += 1)
    end
    n_segments_out = length(root_members)

    do_apply = (rc.mode == "apply") && n_groups_merged > 0
    if do_apply
        tid_attr = getattribute(pc, :tree_id)
        new_tnbs = Vector{Int32}(undef, length(tnbs_attr))
        new_tid = Vector{Int32}(undef, length(tid_attr))
        @inbounds for i in eachindex(tnbs_attr)
            t = Int(tnbs_attr[i])
            if t > 0
                r = _uf_find!(parent, t)
                new_tnbs[i] = Int32(r)
                new_tid[i] = haskey(rep_tree, r) ? rep_tree[r] : Int32(tid_attr[i])
            else
                new_tnbs[i] = Int32(0)
                new_tid[i] = Int32(tid_attr[i])
            end
        end
        new_tid = relabel_by_occurrence(new_tid; positive_only=true, T_out=Int32)
        _relabel_tree_nbs_within_trees!(new_tid, new_tnbs)
        setattribute!(pc, :tree_id, new_tid)
        setattribute!(pc, :tree_nbs_id, new_tnbs)
    end

    isempty(report_path) || _write_merge_report(report_path, pairs, models, segidx,
                                                inter, ratio, reasons, merged_flag)

    @info "$_LOG_PREFIX   qsm_refinement: $(length(models)) segments, $npairs candidate pair(s), " *
          "$n_groups_merged merge group(s) ($n_cross cross-tree)" *
          (do_apply ? "" : rc.mode == "flag_only" ? " [flag_only: cloud unchanged]" : " [no merges]")

    return (status=:success, n_segments_in=length(models), n_segments_out=n_segments_out,
            n_groups_merged=n_groups_merged, n_cross_tree_merges=n_cross,
            pc_output=pc, report_csv_path=report_path, merged=do_apply)
end

"""
    _read_qsm_nodes_csv(path) -> Vector{QSMNode}

Resume fallback: reconstruct the `QSMNode` fields needed for refinement from a
`{prefix}qsm_nodes.csv` written by a prior QSM pass. Columns are matched by
header name; unused fields are left at sensible defaults. Returns an empty vector
if the file is missing or lacks required columns.

Note: the CSV stores coordinates at the writer's precision (`_write_csv` rounds to
8 significant digits), so for large global-CRS coordinates the reconstructed
centers are coarser than the in-memory `qsm()` output. The pipeline therefore
prefers `qsm_res.nodes` (full precision) and only falls back here when refinement
is resumed without an in-session QSM result; see the determinism note above.
"""
function _read_qsm_nodes_csv(path::AbstractString)
    isfile(path) || return QSMNode[]
    lines = readlines(path)
    length(lines) >= 2 || return QSMNode[]
    headers = String.(strip.(split(lines[1], ",")))
    col = Dict(h => i for (i, h) in enumerate(headers))
    needed = ["qsm_node_id", "tree_nbs_id", "tree_id", "agh", "cross_area", "circumference",
              "radius_area", "radius_circ", "height", "completeness", "n_points",
              "center_x", "center_y", "center_z", "direction_x", "direction_y", "direction_z"]
    all(h -> haskey(col, h), needed) || return QSMNode[]
    nodes = QSMNode[]
    for li in 2:length(lines)
        isempty(strip(lines[li])) && continue
        parts = split(lines[li], ",")
        gf(f) = parse(Float64, parts[col[f]])
        gi(f) = round(Int, parse(Float64, parts[col[f]]))
        push!(nodes, QSMNode(
            gi("qsm_node_id"),
            Int32(gi("tree_nbs_id")),
            Int32(gi("tree_id")),
            Int32(gi("tree_nbs_id")),
            gf("agh"), gf("height"), gf("completeness"), gi("n_points"),
            gf("center_x"), gf("center_y"), gf("center_z"),
            gf("direction_x"), gf("direction_y"), gf("direction_z"),
            gf("cross_area"), gf("circumference"), gf("radius_area"), gf("radius_circ"),
        ))
    end
    return nodes
end
