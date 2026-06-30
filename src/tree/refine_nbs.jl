"""
Pre-assembly NBS refinement (per connected component), two strategies sharing the orchestrator,
node-models, and skeleton-node snap: NBS-level Rule B (connectivity) and node-level volume
overlap. The larger-volume NBS absorbs in both; deterministic (fixed global voxel lattice).
"""

"""
    _build_node_models(nodes, voxel_res, nthread) -> Vector{NodeModel}

One `NodeModel` per trial-QSM node with finite `radius_area > 0` (cylinder: axis =
normalized PC1 direction, radius = `radius_area`, half-height = node height / 2). Per-node
self-volume is computed on the global lattice (parallel, disjoint slots); zero-volume nodes
are dropped. Returns nodes in input order (ascending `trial_node_id`).
"""
function _build_node_models(nodes::Vector{QSMNode}, voxel_res::Float64, nthread::Integer)
    isempty(nodes) && return NodeModel[]
    raw = Vector{Tuple{Int,Int32,Cyl,NTuple{6,Float64},Int,Float64,Float64}}()
    for nd in nodes
        nd.nbs_id > 0 || continue
        r = nd.radius_area
        (isfinite(r) && r > 0) || continue
        ax = _unit3(nd.direction_x, nd.direction_y, nd.direction_z)
        cyl = (center = (nd.center_x, nd.center_y, nd.center_z),
               axis = ax, radius = r, half_height = nd.height / 2)::Cyl
        bb = cylinder_aabb(cyl.center, cyl.axis, cyl.radius, cyl.half_height)
        push!(raw, (nd.qsm_node_id, nd.nbs_id, cyl, bb, nd.n_points, nd.completeness, nd.agh))
    end
    isempty(raw) && return NodeModel[]

    vols = zeros(Float64, length(raw))
    _parallel_for(length(raw), nthread) do i
        r = raw[i]
        vols[i] = voxelized_cylinder_volume([r[3]], r[4], voxel_res)
    end

    models = NodeModel[]
    for (i, r) in enumerate(raw)
        vols[i] > 0 || continue
        push!(models, NodeModel(r[1], r[2], r[3], r[4], vols[i], r[5], r[6], r[7]))
    end
    return models
end

"""
    _nbs_skel_nodes(skel_to_nbs) -> Dict{Int32, Vector{Int}}

Invert the skeleton-vertex → NBS map into NBS → its skeleton vertices (sorted), for the
Rule-B connectivity scan.
"""
function _nbs_skel_nodes(skel_to_nbs::AbstractVector{<:Integer})
    out = Dict{Int32, Vector{Int}}()
    @inbounds for sv in eachindex(skel_to_nbs)
        nb = Int32(skel_to_nbs[sv])
        nb > 0 && push!(get!(out, nb, Int[]), sv)
    end
    return out
end

"""
    _nbs_volumes(node_seg, nodemodels) -> Dict{Int32, Float64}

Total cylinder volume per NBS, keyed by the *current* segment label in `node_seg`
(`node_seg[i]` is node `i`'s NBS, updated after Rule B). NBS with no nodes are absent (treated
as volume 0 by callers).
"""
function _nbs_volumes(node_seg::AbstractVector{Int32}, nodemodels::Vector{NodeModel})
    vol = Dict{Int32, Float64}()
    @inbounds for i in eachindex(nodemodels)
        vol[node_seg[i]] = get(vol, node_seg[i], 0.0) + nodemodels[i].vol_vox
    end
    return vol
end

# Volume-ordered donor/receiver comparison: `a` outranks `b` as a receiver iff it has the
# larger cylinder volume, ties broken so the SMALLER nbs id wins (is the receiver).
@inline _bigger(a::Int32, b::Int32, vol::Dict{Int32,Float64}) =
    (get(vol, a, 0.0), -a) > (get(vol, b, 0.0), -b)

"""
    _rule_b_merges(skel_to_nbs, graph_skeleton, nbs_vol, merge_threshold)
        -> (parent::Dict{Int32,Int32}, moves::Vector{RuleBMove})

Standalone, deterministic Rule B. For each NBS `k`, the receiver candidate `R` is the
neighbor NBS touching the most of `k`'s skeleton nodes (ties → smaller id); `k` is merged
into `R` iff the fraction of `k`'s skeleton nodes adjacent to `R` is ≥ `merge_threshold`
AND `R` has the larger cylinder volume. All edges are recorded against the original
labeling (snapshot), then resolved into a `parent` map (each `k` points to its final
volume-maximal root). Returns the resolved `parent` map and one move row per merged NBS.
"""
function _rule_b_merges(skel_to_nbs::AbstractVector{<:Integer},
                        graph_skeleton::SimpleGraph{Int},
                        nbs_vol::Dict{Int32,Float64},
                        merge_threshold::Float64)
    nbs_nodes = _nbs_skel_nodes(skel_to_nbs)
    donors = sort!(collect(keys(nbs_nodes)))         # fixed iteration order
    edge = Dict{Int32,Int32}()                       # donor → immediate receiver
    raw_moves = Vector{Tuple{Int32,Int32,Float64,Int}}()  # (donor, receiver, frac, n_donor_nodes)

    for k in donors
        verts = nbs_nodes[k]
        # vert_count[R] = number of k's skeleton nodes with ≥1 edge to a node labeled R.
        vert_count = Dict{Int32,Int}()
        for v in verts
            touched = Set{Int32}()
            for sn in neighbors(graph_skeleton, v)
                r = Int32(skel_to_nbs[sn])
                (r > 0 && r != k) || continue
                push!(touched, r)
            end
            for r in touched
                vert_count[r] = get(vert_count, r, 0) + 1
            end
        end
        # Receiver = the best-connected neighbor that is BIGGER-volume and meets the
        # connectivity threshold; rank by (count, volume, -id) so ties are broken toward
        # the larger NBS, then the smaller id (deterministic, independent of Dict order).
        best = Int32(0); best_key = (-1, -Inf, typemin(Int32)); best_frac = 0.0
        for (R, c) in vert_count
            frac = c / length(verts)
            (frac >= merge_threshold && _bigger(R, k, nbs_vol)) || continue
            key = (c, get(nbs_vol, R, 0.0), -R)
            if key > best_key
                best_key = key; best = R; best_frac = frac
            end
        end
        best == 0 && continue
        edge[k] = best
        push!(raw_moves, (k, best, best_frac, length(verts)))
    end

    # Resolve to roots. Edges strictly increase in (volume, -id) ⇒ acyclic, so following
    # parents terminates at a unique volume-maximal root.
    function find_root(x::Int32)
        while haskey(edge, x)
            x = edge[x]
        end
        return x
    end
    parent = Dict{Int32,Int32}()
    for k in keys(edge)
        parent[k] = find_root(k)
    end

    moves = RuleBMove[]
    for (k, _, frac, ndon) in raw_moves
        rcv = parent[k]
        push!(moves, RuleBMove(k, rcv, get(nbs_vol, k, 0.0), get(nbs_vol, rcv, 0.0), frac, ndon))
    end
    sort!(moves; by = m -> m.donor_nbs)
    return parent, moves
end

"""
    _build_focal_models(node_seg, nodemodels, nbs_vol) -> (Vector{SegModel}, Dict{Int32,Vector{Int}})

Group node indices by their current segment label (`node_seg`) and build a focal `SegModel`
per NBS (union cylinders + AABB, total volume, mean completeness). Returns the seg models
(ordered by `seg_id`) and the `seg_id → node-index` map.
"""
function _build_focal_models(node_seg::AbstractVector{Int32}, nodemodels::Vector{NodeModel},
                             nbs_vol::Dict{Int32,Float64})
    seg_nodes = Dict{Int32,Vector{Int}}()
    for i in eachindex(nodemodels)
        push!(get!(seg_nodes, node_seg[i], Int[]), i)
    end
    seg_ids = sort!(collect(keys(seg_nodes)))
    segs = SegModel[]
    for sid in seg_ids
        idxs = seg_nodes[sid]
        cyls = Cyl[nodemodels[i].cyl for i in idxs]
        bb = nodemodels[idxs[1]].aabb
        sc = 0.0; tot = 0
        for i in idxs
            nm = nodemodels[i]
            a = nm.aabb
            bb = (min(bb[1], a[1]), max(bb[2], a[2]),
                  min(bb[3], a[3]), max(bb[4], a[4]),
                  min(bb[5], a[5]), max(bb[6], a[6]))
            sc += nm.completeness; tot += nm.n_points
        end
        push!(segs, SegModel(sid, cyls, bb, get(nbs_vol, sid, 0.0), tot, sc / length(idxs)))
    end
    return segs, seg_nodes
end

"""
    _node_merge_reason(focal, node, ratio, cfg) -> Symbol

Whether `node` (from a smaller-volume NBS) may be claimed by `focal`. Returns `:ok` or the
first failing gate: `:below_overlap_threshold`, `:node_completeness_gate`,
`:focal_completeness_gate`, `:min_points_gate` (per-focal). (No tree/cross-tree gates:
refinement is pre-assembly, per-CC.)
"""
function _node_merge_reason(F::SegModel, n::NodeModel, ratio::Float64, cfg::FLiPConfig)
    rc = cfg.tree.refine
    ratio > rc.overlap_threshold || return :below_overlap_threshold
    n.completeness >= rc.completeness_gate || return :node_completeness_gate
    F.completeness >= rc.completeness_gate || return :focal_completeness_gate
    F.n_points >= rc.min_points_gate || return :min_points_gate
    return :ok
end

"""
    _volume_node_moves(node_seg, nodemodels, nbs_vol, voxel_res, margin, cfg)
        -> (node_claimed_by::Vector{Int32}, moves::Vector{NodeMove})

Node-level volume merge. Process NBS focals largest-volume first; each focal claims the
individual nodes of smaller-volume NBS whose cylinder overlaps the focal's cylinder union
(per-node ratio = inter_vol(node, focal) / vol(node) > `overlap_threshold`, gates pass).
A node already claimed by a larger focal is skipped. `node_seg` is the post-Rule-B labeling.
"""
function _volume_node_moves(node_seg::Vector{Int32}, nodemodels::Vector{NodeModel},
                            nbs_vol::Dict{Int32,Float64}, voxel_res::Float64,
                            margin::Float64, cfg::FLiPConfig)
    Nn = length(nodemodels)
    nthread = effective_nthreads(cfg)
    segmodels, seg_nodes = _build_focal_models(node_seg, nodemodels, nbs_vol)

    centers = Matrix{Float64}(undef, 3, Nn)
    reach = Vector{Float64}(undef, Nn)
    @inbounds for k in 1:Nn
        c = nodemodels[k].cyl
        centers[1, k] = c.center[1]; centers[2, k] = c.center[2]; centers[3, k] = c.center[3]
        reach[k] = c.radius + c.half_height
    end
    Rmax = Nn == 0 ? 0.0 : maximum(reach)
    tree = KDTree(centers)

    # Focal order: largest volume first, ties by seg_id; receiver must out-rank candidate.
    order = sortperm(segmodels; by = m -> (-m.vol_vox, m.seg_id))
    rank_of = Dict{Int32,Int}()
    for (r, oi) in enumerate(order)
        rank_of[segmodels[oi].seg_id] = r
    end

    node_claimed_by = copy(node_seg)
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
                node_seg[nj] == F.seg_id && continue        # same NBS
                last_focal[nj] == r && continue           # dedupe within this focal
                last_focal[nj] = r
                rank_of[node_seg[nj]] > r || continue        # candidate must be smaller/later
                node_claimed_by[nj] == node_seg[nj] || continue   # not already claimed by a larger focal
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

        for t in sortperm(cand)                            # claim in ascending node-index order
            nj = cand[t]
            nm = nodemodels[nj]
            _node_merge_reason(F, nm, ratios[t], cfg) === :ok || continue
            node_claimed_by[nj] = F.seg_id
            push!(moves, NodeMove(nm.trial_node_id, node_seg[nj], F.seg_id,
                                  ratios[t], nm.n_points, nm.completeness, nm.agh))
        end
    end
    return node_claimed_by, moves
end

"""
    refine_nbs(; nbs_id, node_id, trial_node_id, nodes, graph_skeleton, skel_to_nbs, cfg)
        -> NamedTuple

Per-CC, pre-assembly NBS refinement (see module docstring). Operates on per-point label
vectors: `nbs_id` (rewritten), `node_id` (skeleton node, for the snap), `trial_node_id`
(links points to trial-QSM nodes). `nodes` are this CC's trial-QSM nodes; `graph_skeleton`
and `skel_to_nbs` (skeleton vertex → nbs) drive Rule B.

Returns `(nbs_id, n_rule_b_merges, n_nodes_moved, rule_b_moves, node_moves)`. In
`cfg.tree.refine.mode == "flag_only"` the returned `nbs_id` is unchanged (moves are still
reported); in `"apply"` it is the rewritten, densely-relabeled labeling.
"""
function refine_nbs(; nbs_id::AbstractVector{<:Integer},
                      node_id::AbstractVector{<:Integer},
                      trial_node_id::AbstractVector{<:Integer},
                      nodes::Vector{QSMNode},
                      graph_skeleton::SimpleGraph{Int},
                      skel_to_nbs::AbstractVector{<:Integer},
                      cfg::FLiPConfig)
    rc = cfg.tree.refine
    voxel_res = rc.voxel_res_scalar * cfg.pipeline.subsample_res
    nthread = effective_nthreads(cfg)
    new_nbs = Int32.(nbs_id)
    nores = (nbs_id=new_nbs, n_rule_b_merges=0, n_nodes_moved=0,
             rule_b_moves=RuleBMove[], node_moves=NodeMove[])

    voxel_res > 0 || (@warn "$_LOG_PREFIX refine_nbs: non-positive voxel resolution; skipping"; return nores)

    nodemodels = _build_node_models(nodes, voxel_res, nthread)
    node_seg = Int32[nm.seg_id for nm in nodemodels]

    # ── Step 1: NBS-level Rule B merge ───────────────────────────────────────
    merge_threshold = cfg.tree.assembly.merge_threshold
    nbs_vol0 = _nbs_volumes(node_seg, nodemodels)
    parent, rule_b_moves = _rule_b_merges(skel_to_nbs, graph_skeleton, nbs_vol0, merge_threshold)
    relabel(x::Int32) = get(parent, x, x)
    @inbounds for i in eachindex(node_seg)
        node_seg[i] = relabel(node_seg[i])
    end

    # ── Step 2: node-level volume merge (on the Rule-B-merged labeling) ──────
    nbs_vol1 = _nbs_volumes(node_seg, nodemodels)
    margin = max(0.0, rc.candidate_radius_scalar * cfg.pipeline.subsample_res)
    node_claimed_by, node_moves = isempty(nodemodels) ? (node_seg, NodeMove[]) :
        _volume_node_moves(node_seg, nodemodels, nbs_vol1, voxel_res, margin, cfg)

    do_apply = rc.mode == "apply"
    if do_apply && (!isempty(rule_b_moves) || !isempty(node_moves))
        # Rule B: rewrite every point's nbs by the resolved parent map.
        @inbounds for i in eachindex(new_nbs)
            new_nbs[i] = relabel(new_nbs[i])
        end
        # Volume: move the points of each claimed node to its new NBS.
        if !isempty(node_moves)
            move_map = Dict{Int,Int32}()
            for k in eachindex(nodemodels)
                node_claimed_by[k] != node_seg[k] && (move_map[nodemodels[k].trial_node_id] = node_claimed_by[k])
            end
            if !isempty(move_map)
                @inbounds for i in eachindex(new_nbs)
                    tv = Int(trial_node_id[i])
                    tv > 0 && haskey(move_map, tv) && (new_nbs[i] = move_map[tv])
                end
            end
        end
        # Fix #1 — snap to skeleton-node granularity: each skeleton node's points take the
        # plurality nbs (tie → smaller nbs), so no skeleton node is split across NBS.
        _snap_to_skeleton_nodes!(new_nbs, node_id)
        # NOTE: labels are merged into existing (receiver) ids — a subset of the input — so
        # they stay globally unique across CCs. The caller does the final dense relabel.
    end

    n_moved = length(unique(mv.trial_node_id for mv in node_moves))
    return (nbs_id=new_nbs, n_rule_b_merges=length(rule_b_moves), n_nodes_moved=n_moved,
            rule_b_moves=rule_b_moves, node_moves=node_moves)
end

"""
    _snap_to_skeleton_nodes!(nbs_id, node_id) -> nbs_id

For each skeleton node (`node_id` value > 0), set ALL its points to the plurality `nbs_id`
among them (ties → smaller `nbs_id`). Keeps `nbs_id` consistent at skeleton-node
granularity so the downstream `skel_to_nbs` map is single-valued and order-independent.
Points with `node_id == 0` are left untouched.
"""
function _snap_to_skeleton_nodes!(nbs_id::AbstractVector{Int32}, node_id::AbstractVector{<:Integer})
    groups = Dict{Int, Vector{Int}}()
    @inbounds for i in eachindex(node_id)
        nd = Int(node_id[i])
        nd > 0 && push!(get!(groups, nd, Int[]), i)
    end
    for (_, idxs) in groups
        length(idxs) <= 1 && continue
        counts = Dict{Int32,Int}()
        @inbounds for i in idxs
            counts[nbs_id[i]] = get(counts, nbs_id[i], 0) + 1
        end
        length(counts) == 1 && continue                    # already consistent
        best = first(sort!(collect(keys(counts)); by = v -> (-counts[v], v)))
        @inbounds for i in idxs
            nbs_id[i] = best
        end
    end
    return nbs_id
end

# ── Report writers (called by the tree-segmentation stage under enable_debug_info) ──

