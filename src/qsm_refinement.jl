"""
Post-QSM refinement for FLiP.jl.

Umbrella for methods that improve QSM results after a first QSM pass. The current
(and only) method is [`nbs_merge_by_volume_overlap`](@ref): it works at the
**node (QSM-slice) level**. Each Non-Branching Segment (NBS, keyed by
`tree_nbs_id`) is processed as a "focal" in descending-size order (largest first);
a focal **claims the individual nodes** of smaller NBS whose fitted cylinder
overlaps the focal's cylinder-union. Only the points of claimed nodes are
relabeled (via the per-point `:qsm_node_id` attribute) — so a partial overlap
reassigns just the overlapping nodes, never the whole NBS. The pipeline then
re-runs QSM on the relabeled cloud so each segment is re-fit.

Each node ends up in the largest NBS that overlaps it. Note this does NOT collapse
transitive chains: if A–B overlap and B–C overlap but A–C are tangent, C's node
joins B (not A) — by design, since a node joins the largest NBS overlapping *it*.

Determinism: per-node overlap is estimated on a FIXED global voxel lattice (see
`voxelized_cylinder_volume`); NBS are processed in a fixed `(-n_points, seg_id)`
order, candidate nodes are gathered deterministically, the parallel ratio pass
writes disjoint slots, and claims are applied in ascending node-index order — so a
single in-session run is reproducible and thread-order independent, matching
FLiP's no-random-seed guarantee. (Resuming refinement purely from disk is
best-effort: the tree cloud and node CSV are quantized on write — see
`_read_qsm_nodes_csv` — so a borderline node near a gate threshold could resolve
differently than the in-session run. Run refinement in the same session as QSM,
the default, for exact reproducibility.)

Known limitation (out of scope): pure volume overlap cannot bridge gap-separated
collinear segments (no shared volume ⇒ no claim).
"""

using NearestNeighbors: KDTree, inrange

# A finite cylinder from one QSM node: midpoint, unit axis, radius, half-height.
const Cyl = NamedTuple{(:center, :axis, :radius, :half_height),
                       Tuple{NTuple{3,Float64}, NTuple{3,Float64}, Float64, Float64}}

"""Per-node cylinder model (one QSM slice). `qsm_node_id` links to per-point `:qsm_node_id`."""
struct NodeModel
    qsm_node_id::Int
    seg_id::Int32                 # owning tree_nbs_id (original)
    tree_id::Int32
    cyl::Cyl
    aabb::NTuple{6,Float64}
    vol_vox::Float64             # deterministic per-node self-volume (global lattice)
    n_points::Int
    completeness::Float64
    agh::Float64
end

"""Per-NBS focal anchor: union of the NBS's original node cylinders + aggregates."""
struct SegModel
    seg_id::Int32
    tree_id::Int32
    cyls::Vector{Cyl}
    aabb::NTuple{6,Float64}       # union of node AABBs
    n_points::Int                # Σ node n_points (size for ordering)
    completeness::Float64        # mean node completeness
    min_agh::Float64             # min node AGH (grounded-trunk detection)
end

"""One reassignment for the merged-only report (a node moved into a focal NBS)."""
struct NodeMove
    qsm_node_id::Int
    from_tree_nbs::Int32
    to_tree_nbs::Int32
    from_tree_id::Int32
    to_tree_id::Int32
    overlap_ratio::Float64
    n_points::Int
    completeness::Float64
    agh::Float64
    cross_tree::Bool
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

Deterministic volume of the 3-D intersection of two cylinder unions, restricted to
`box`, on the same global lattice as `voxelized_cylinder_volume` (a voxel counts
iff its center is inside some cylinder of `cyls_a` AND some cylinder of `cyls_b`).
Because the lattice is shared, the result is ≤ each side's self-volume, so the
per-node overlap ratio is ≤ 1. Used here with `cyls_a = [node.cyl]` and
`cyls_b = focal.cyls`.
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
    _build_node_models(nodes, voxel_res, nthread) -> Vector{NodeModel}

One `NodeModel` per QSM node with finite `radius_area > 0` (cylinder: axis =
normalized PC1 direction, radius = `radius_area`, half-height = node height / 2).
Per-node self-volume is computed on the global lattice (parallel, disjoint slots);
zero-volume nodes are dropped. Returns nodes in input order (ascending `qsm_node_id`).
"""
function _build_node_models(nodes::Vector{QSMNode}, voxel_res::Float64, nthread::Integer)
    isempty(nodes) && return NodeModel[]
    raw = Vector{Tuple{Int,Int32,Int32,Cyl,NTuple{6,Float64},Int,Float64,Float64}}()
    for nd in nodes
        nd.nbs_id > 0 || continue
        r = nd.radius_area
        (isfinite(r) && r > 0) || continue
        ax = _unit3(nd.direction_x, nd.direction_y, nd.direction_z)
        cyl = (center = (nd.center_x, nd.center_y, nd.center_z),
               axis = ax, radius = r, half_height = nd.height / 2)::Cyl
        bb = cylinder_aabb(cyl.center, cyl.axis, cyl.radius, cyl.half_height)
        push!(raw, (nd.qsm_node_id, nd.nbs_id, nd.tree_id, cyl, bb, nd.n_points, nd.completeness, nd.agh))
    end
    isempty(raw) && return NodeModel[]

    vols = zeros(Float64, length(raw))
    _parallel_for(length(raw), nthread) do i
        r = raw[i]
        vols[i] = voxelized_cylinder_volume([r[4]], r[5], voxel_res)
    end

    models = NodeModel[]
    for (i, r) in enumerate(raw)
        vols[i] > 0 || continue
        push!(models, NodeModel(r[1], r[2], r[3], r[4], r[5], vols[i], r[6], r[7], r[8]))
    end
    return models
end

"""
    _build_focal_models(nodemodels) -> (Vector{SegModel}, Dict{Int32,Vector{Int}})

Group node-model indices by `seg_id` (tree_nbs_id) and build a focal `SegModel` per
NBS (union cylinders + AABB, Σ points, mean completeness, min AGH). Returns the
seg models (ordered by `seg_id`) and the `seg_id → node-index` map.
"""
function _build_focal_models(nodemodels::Vector{NodeModel})
    seg_nodes = Dict{Int32,Vector{Int}}()
    for (i, nm) in enumerate(nodemodels)
        push!(get!(seg_nodes, nm.seg_id, Int[]), i)
    end
    seg_ids = sort!(collect(keys(seg_nodes)))
    segs = SegModel[]
    for sid in seg_ids
        idxs = seg_nodes[sid]
        cyls = Cyl[nodemodels[i].cyl for i in idxs]
        bb = nodemodels[idxs[1]].aabb
        tot = 0; sc = 0.0; magh = Inf
        for i in idxs
            nm = nodemodels[i]
            a = nm.aabb
            bb = (min(bb[1], a[1]), max(bb[2], a[2]),
                  min(bb[3], a[3]), max(bb[4], a[4]),
                  min(bb[5], a[5]), max(bb[6], a[6]))
            tot += nm.n_points; sc += nm.completeness; magh = min(magh, nm.agh)
        end
        push!(segs, SegModel(sid, nodemodels[idxs[1]].tree_id, cyls, bb, tot, sc / length(idxs), magh))
    end
    return segs, seg_nodes
end

"""
    _node_merge_reason(focal, node, ratio, cfg) -> Symbol

Whether `node` (from a smaller NBS) may be claimed by `focal`. Returns `:ok` or the
first failing gate: `:below_overlap_threshold`, `:node_completeness_gate`,
`:focal_completeness_gate`, `:min_points_gate` (per-focal), `:cross_tree_disabled`,
`:grounded_trunk_guard`.
"""
function _node_merge_reason(F::SegModel, n::NodeModel, ratio::Float64, cfg::FLiPConfig)
    rc = cfg.qsm_refinement
    ratio > rc.overlap_threshold || return :below_overlap_threshold
    n.completeness >= rc.completeness_gate || return :node_completeness_gate
    F.completeness >= rc.completeness_gate || return :focal_completeness_gate
    F.n_points >= rc.min_points_gate || return :min_points_gate
    if n.tree_id != F.tree_id
        rc.cross_tree || return :cross_tree_disabled
        if rc.protect_grounded_trunks
            ground = cfg.tree_segmentation.nearground_agh_threshold
            (n.agh <= ground && F.min_agh <= ground) && return :grounded_trunk_guard
        end
    end
    return :ok
end

function _write_node_merge_report(path::String, moves::Vector{NodeMove})
    n = length(moves)
    qid = Vector{Int}(undef, n)
    fnb = Vector{Int32}(undef, n); tnb = Vector{Int32}(undef, n)
    ftd = Vector{Int32}(undef, n); ttd = Vector{Int32}(undef, n)
    rat = Vector{Float64}(undef, n); npt = Vector{Int}(undef, n)
    cmp = Vector{Float64}(undef, n); agv = Vector{Float64}(undef, n)
    crs = Vector{Int}(undef, n)
    ord = sortperm(moves; by = mv -> mv.qsm_node_id)   # deterministic: ascending node id
    for (k, j) in enumerate(ord)
        mv = moves[j]
        qid[k] = mv.qsm_node_id; fnb[k] = mv.from_tree_nbs; tnb[k] = mv.to_tree_nbs
        ftd[k] = mv.from_tree_id; ttd[k] = mv.to_tree_id; rat[k] = mv.overlap_ratio
        npt[k] = mv.n_points; cmp[k] = mv.completeness; agv[k] = mv.agh
        crs[k] = mv.cross_tree ? 1 : 0
    end
    headers = ["qsm_node_id", "from_tree_nbs", "to_tree_nbs", "from_tree_id", "to_tree_id",
               "node_overlap_ratio", "node_n_points", "node_completeness", "node_agh", "cross_tree"]
    columns = AbstractVector[qid, fnb, tnb, ftd, ttd, rat, npt, cmp, agv, crs]
    _write_csv(path, columns, headers)
end

"""
    nbs_merge_by_volume_overlap(; pc, nodes, cfg=_CFG, output_dir="", output_prefix="output")
        -> NamedTuple

Node-level refinement: process NBS as focals in descending-size order; each focal
claims the individual nodes of smaller NBS whose cylinder overlaps the focal's
cylinder-union (per-node ratio = `inter_vol(node, focal_union) / vol(node)` >
`overlap_threshold`, gates pass). In `mode == "apply"` the claimed nodes' points
(matched by `:qsm_node_id`) are relabeled IN PLACE — `:tree_nbs_id` ← focal,
`:tree_id` ← focal's tree — then re-densified with `relabel_by_occurrence` +
`_relabel_tree_nbs_within_trees!`; in `"flag_only"` the cloud is untouched. A
report listing ONLY the reassignments is written when `output_dir` is set. Does NOT
re-run QSM — the pipeline stage does that.

Returns `(status, n_segments_in, n_segments_out, n_nodes_moved, n_focal_nbs,
n_cross_tree_moves, pc_output, report_csv_path, merged)`; `merged` is `true` only
when the cloud was actually relabeled (drives the second QSM pass).
"""
function nbs_merge_by_volume_overlap(; pc::PointCloud, nodes::Vector{QSMNode},
                                     cfg::FLiPConfig = _CFG,
                                     output_dir::AbstractString = "",
                                     output_prefix::AbstractString = "output")
    rc = cfg.qsm_refinement
    voxel_res = rc.voxel_res_scalar * cfg.pipeline.subsample_res
    nthread = effective_nthreads(cfg)
    report_path = isempty(output_dir) ? "" : joinpath(output_dir, "$(output_prefix)nbs_merge_report.csv")

    nores = (status=:no_nodes, n_segments_in=0, n_segments_out=0, n_nodes_moved=0,
             n_focal_nbs=0, n_cross_tree_moves=0, pc_output=pc, report_csv_path="", merged=false)

    (hasattribute(pc, :tree_nbs_id) && hasattribute(pc, :tree_id) && hasattribute(pc, :qsm_node_id)) || begin
        @warn "$_LOG_PREFIX qsm_refinement: cloud lacks :tree_nbs_id/:tree_id/:qsm_node_id; skipping"
        return nores
    end
    voxel_res > 0 || begin
        @warn "$_LOG_PREFIX qsm_refinement: non-positive voxel resolution; skipping"
        return nores
    end

    nodemodels = _build_node_models(nodes, voxel_res, nthread)
    if isempty(nodemodels)
        @info "$_LOG_PREFIX   qsm_refinement: no QSM node cylinders to evaluate"
        return nores
    end
    segmodels, seg_nodes = _build_focal_models(nodemodels)

    # KDTree over node centers (column k ↔ nodemodels[k]); adaptive search radius.
    Nn = length(nodemodels)
    centers = Matrix{Float64}(undef, 3, Nn)
    reach = Vector{Float64}(undef, Nn)
    @inbounds for k in 1:Nn
        c = nodemodels[k].cyl
        centers[1, k] = c.center[1]; centers[2, k] = c.center[2]; centers[3, k] = c.center[3]
        reach[k] = c.radius + c.half_height
    end
    Rmax = maximum(reach)
    # `margin ≥ 0` keeps the search radius (reach + Rmax + margin) a provable upper
    # bound on any overlapping-node distance, so a negative config scalar can't
    # introduce false-negative candidates.
    margin = max(0.0, rc.candidate_radius_scalar * cfg.pipeline.subsample_res)
    tree = KDTree(centers)

    # Focal processing order: largest first (by point count), ties by seg_id.
    order = sortperm(segmodels; by = m -> (-m.n_points, m.seg_id))
    rank_of = Dict{Int32,Int}()
    for (r, oi) in enumerate(order)
        rank_of[segmodels[oi].seg_id] = r
    end

    claimed_by = Int32[nm.seg_id for nm in nodemodels]
    last_focal = zeros(Int, Nn)
    moves = NodeMove[]
    q = Vector{Float64}(undef, 3)

    for (r, oi) in enumerate(order)
        F = segmodels[oi]
        cand = Int[]
        for ni in seg_nodes[F.seg_id]
            c = nodemodels[ni].cyl.center
            q[1] = c[1]; q[2] = c[2]; q[3] = c[3]
            for nj in inrange(tree, q, reach[ni] + Rmax + margin)
                nm = nodemodels[nj]
                nm.seg_id == F.seg_id && continue       # same NBS
                last_focal[nj] == r && continue         # dedupe within this focal
                last_focal[nj] = r
                rank_of[nm.seg_id] > r || continue      # candidate must be smaller/later
                claimed_by[nj] == nm.seg_id || continue # not already claimed by a larger focal
                aabbs_overlap(nm.aabb, F.aabb) || continue
                push!(cand, nj)
            end
        end
        isempty(cand) && continue

        ratios = zeros(Float64, length(cand))
        _parallel_for(length(cand), nthread) do t
            nm = nodemodels[cand[t]]
            box = _aabb_intersection(nm.aabb, F.aabb)
            iv = _voxel_intersection_volume([nm.cyl], F.cyls, box, voxel_res)
            ratios[t] = nm.vol_vox > 0 ? iv / nm.vol_vox : 0.0
        end

        for t in sortperm(cand)                          # claim in ascending node-index order
            nj = cand[t]
            nm = nodemodels[nj]
            _node_merge_reason(F, nm, ratios[t], cfg) === :ok || continue
            claimed_by[nj] = F.seg_id
            push!(moves, NodeMove(nm.qsm_node_id, nm.seg_id, F.seg_id, nm.tree_id, F.tree_id,
                                  ratios[t], nm.n_points, nm.completeness, nm.agh, nm.tree_id != F.tree_id))
        end
    end

    n_segments_out = length(unique(claimed_by))
    do_apply = (rc.mode == "apply") && !isempty(moves)
    if do_apply
        move_map = Dict{Int,Tuple{Int32,Int32}}()
        for mv in moves
            move_map[mv.qsm_node_id] = (mv.to_tree_nbs, mv.to_tree_id)
        end
        qn = getattribute(pc, :qsm_node_id)
        tnbs = getattribute(pc, :tree_nbs_id)
        tid = getattribute(pc, :tree_id)
        new_tnbs = Vector{Int32}(undef, length(tnbs))
        new_tid = Vector{Int32}(undef, length(tid))
        @inbounds for i in eachindex(qn)
            qv = Int(qn[i])
            if qv > 0 && haskey(move_map, qv)
                mt = move_map[qv]
                new_tnbs[i] = mt[1]; new_tid[i] = mt[2]
            else
                new_tnbs[i] = Int32(tnbs[i]); new_tid[i] = Int32(tid[i])
            end
        end
        new_tid = relabel_by_occurrence(new_tid; positive_only=true, T_out=Int32)
        _relabel_tree_nbs_within_trees!(new_tid, new_tnbs)
        setattribute!(pc, :tree_id, new_tid)
        setattribute!(pc, :tree_nbs_id, new_tnbs)
    end

    isempty(report_path) || _write_node_merge_report(report_path, moves)

    n_focal = length(unique(mv.to_tree_nbs for mv in moves))
    n_cross = count(mv -> mv.cross_tree, moves)
    @info "$_LOG_PREFIX   qsm_refinement: $(length(segmodels)) NBS / $Nn nodes; " *
          "$(length(moves)) node(s) moved into $n_focal focal NBS ($n_cross cross-tree)" *
          (do_apply ? "" : rc.mode == "flag_only" ? " [flag_only: cloud unchanged]" : " [no moves]")

    return (status=:success, n_segments_in=length(segmodels), n_segments_out=n_segments_out,
            n_nodes_moved=length(moves), n_focal_nbs=n_focal, n_cross_tree_moves=n_cross,
            pc_output=pc, report_csv_path=report_path, merged=do_apply)
end

"""
    _read_qsm_nodes_csv(path) -> Vector{QSMNode}

Resume fallback: reconstruct the `QSMNode` fields needed for refinement from a
`{prefix}qsm_nodes.csv` written by a prior QSM pass. Columns are matched by header
name; unused fields are left at sensible defaults. Returns an empty vector if the
file is missing or lacks required columns.

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
