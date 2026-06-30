"""
Tree assembly: grow trees from near-ground seed NBS through the skeleton graph (Rule A/B),
rescue orphan branches across occlusion gaps, relabel `tree_nbs_id` within trees.
"""

"""
    assemble_segments(graph, coords, nbs_id, node_id, agh_values,
                      graph_skeleton, skeleton_cloud; cfg) -> NamedTuple

Assemble non-branching segments into individual trees by iteratively growing
from near-ground seed NBS outward through the skeleton graph.

# Arguments
- `graph::SimpleGraph{Int}`: point-level radius graph
- `coords::AbstractMatrix{<:Real}`: N×3 point coordinates
- `nbs_id::AbstractVector{<:Integer}`: per-point NBS label (0=discarded, 1..k)
- `node_id::AbstractVector{<:Integer}`: per-point node label (globally unique)
- `agh_values::AbstractVector{<:Real}`: per-point above-ground height
- `graph_skeleton::SimpleGraph{Int}`: skeleton graph (one vertex per node)
- `skeleton_cloud::PointCloud`: skeleton point cloud with `:node_id` attribute
- `cfg::FLiPConfig`: uses `tree_nearground_agh_threshold`, `tree_assembly_merge_threshold`
- `enable_rule_b::Bool`: if `false`, skip connectivity-based NBS merging (Rule B) so each
  NBS stays its own branch — used when NBS merging is done upstream (`refine_nbs`)

# Returns
`(tree_nbs_id::Vector{Int32}, tree_id::Vector{Int32})` where `tree_id` is the
per-point tree assignment (0=unassigned) and `tree_nbs_id` is the per-point NBS
label re-ordered continuously by descending size.
"""
function assemble_segments(
    graph::SimpleGraph{Int},
    coords::AbstractMatrix{<:Real},
    nbs_id::AbstractVector{<:Integer},
    node_id::AbstractVector{<:Integer},
    agh_values::AbstractVector{<:Real},
    graph_skeleton::SimpleGraph{Int},
    skeleton_cloud::PointCloud;
    cfg::FLiPConfig = _CFG,
    enable_rule_b::Bool = true,
)
    N = size(coords, 1)
    length(nbs_id) == N      || throw(ArgumentError("nbs_id length must match number of points"))
    length(node_id) == N     || throw(ArgumentError("node_id length must match number of points"))
    length(agh_values) == N  || throw(ArgumentError("agh_values length must match number of points"))

    # Use the same near-ground ceiling as label_non_branching_segments:
    # threshold + 2× subsample resolution to account for discretisation
    nearground_ceiling = cfg.tree.extraction.nearground_agh_threshold + 2.0 * cfg.pipeline.subsample_res
    merge_threshold    = cfg.tree.assembly.merge_threshold

    tree_id     = zeros(Int32, N)
    tree_nbs_id = Int32.(copy(nbs_id))

    K_nbs    = Int(maximum(nbs_id;  init=0))
    max_node = Int(maximum(node_id; init=0))
    K_nbs == 0 && return (tree_nbs_id = tree_nbs_id, tree_id = tree_id)

    # ── Step 4.0: precomputations ────────────────────────────────
    info = _init_assembly_info(graph, nbs_id, node_id, graph_skeleton, skeleton_cloud,
                               K_nbs, max_node)
    nbs_points     = info.nbs_points
    skel_to_nbs    = info.skel_to_nbs
    nbs_skel_nodes = info.nbs_skel_nodes
    nbs_adj        = info.nbs_adj

    # ── Step 4.1: seed trees from near-ground NBS ────────────────
    seed_res = _seed_trees_from_nearground!(tree_id, nbs_points, agh_values, nearground_ceiling)
    nbs_tree     = seed_res.nbs_tree       # Vector{Int32}  (0 = unassigned)
    assigned_nbs = seed_res.assigned_nbs   # BitVector
    next_tree_id = seed_res.next_tree_id

    cfg.pipeline.enable_debug_info && @info "Assembly: seeded $(next_tree_id - 1) trees from near-ground NBS"

    # A component is "grounded" iff it had at least one near-ground seed. An ungrounded
    # component is still grown (so its branches get `tree_nbs_id` labels) but its
    # `tree_id` is zeroed below, so its points become orphans for the occlusion rescue.
    had_nearground = next_tree_id > Int32(1)

    # ── Step 4.1b: fallback seed for ground-disconnected CCs ────
    # If this component has no near-ground NBS, iterative growth would leave every
    # NBS unassigned. We always seed the largest NBS as a fresh tree so step 4.2 can
    # grow through it and assign branch labels; the temporary `tree_id` is zeroed below.
    if next_tree_id == Int32(1)
        assigned_tid = _seed_largest_nbs!(tree_id, nbs_tree, assigned_nbs,
                                          nbs_points, next_tree_id; cfg=cfg)
        if assigned_tid > 0
            next_tree_id += Int32(1)
        end
    end

    # ── Step 4.2: iterative growth via skeleton graph ────────────
    iteration = _iterative_tree_growth!(tree_id, tree_nbs_id, nbs_tree, assigned_nbs,
                                        K_nbs, nbs_points, nbs_skel_nodes, skel_to_nbs,
                                        nbs_adj, graph_skeleton, Float64(merge_threshold);
                                        cfg=cfg, enable_rule_b=enable_rule_b)

    # ── Step 4.3: re-order tree_nbs_id by descending group size ──────
    # Within this assemble_segments call, every non-zero tree_nbs_id value belongs
    # to exactly one tree: original NBS labels are unique 1..K, Rule A leaves them
    # untouched, and Rule B adopts the target NBS's (already-unique) label. So
    # keying by tree_nbs_id alone is equivalent to keying by (tree_id, tree_nbs_id)
    # and lets us delegate to `relabel_by_occurrence`.
    @inbounds for i in 1:N
        tree_id[i] == 0 && (tree_nbs_id[i] = Int32(0))
    end
    tree_nbs_id = relabel_by_occurrence(tree_nbs_id; positive_only=true, T_out=Int32)

    # Ground-disconnected component: keep the branch labels (`tree_nbs_id`) but drop the
    # temporary `tree_id` so every point becomes an orphan (`tree_id==0 && tree_nbs_id>0`)
    # for `assemble_occluded_segments` to rescue.
    had_nearground || fill!(tree_id, Int32(0))

    if cfg.pipeline.enable_debug_info
        n_trees = length(unique(tid for tid in tree_id if tid > 0))
        n_assigned_pts = count(>(Int32(0)), tree_id)
        @info "Assembly complete" n_trees n_assigned_points=n_assigned_pts total_points=N iterations=iteration
    end

    return (tree_nbs_id = tree_nbs_id, tree_id = tree_id)
end

"""
    _init_assembly_info(graph, nbs_id, node_id, graph_skeleton, skeleton_cloud,
                        K_nbs, max_node) -> NamedTuple

Build the index structures consumed by `assemble_segments`:

- `nbs_points::Vector{Vector{Int}}` — NBS id → point indices
- `node_to_skel::Vector{Int}` — node id → skeleton vertex (0 = not in skeleton)
- `skel_to_nbs::Vector{Int}` — skeleton vertex → NBS label
- `nbs_skel_nodes::Vector{Vector{Int}}` — NBS id → skeleton vertex set
- `nbs_adj::SparseMatrixCSC{Int,Int}` — symmetric K_nbs×K_nbs adjacency-count matrix

NBS labels and node ids are both dense within a component (1..K_nbs and 1..max_node
respectively), so each container is Vector-indexed for O(1) hash-free lookup. The
adjacency uses `SparseMatrixCSC` so missing-pair lookups return `0` without any
explicit `get(.., 0)` wrapper.
"""
function _init_assembly_info(graph::SimpleGraph{Int},
                             nbs_id::AbstractVector{<:Integer},
                             node_id::AbstractVector{<:Integer},
                             graph_skeleton::SimpleGraph{Int},
                             skeleton_cloud::PointCloud,
                             K_nbs::Int, max_node::Int)
    N      = length(nbs_id)
    n_skel = nv(graph_skeleton)

    # (a) nbs_id → point indices
    nbs_points = [Int[] for _ in 1:K_nbs]
    @inbounds for i in 1:N
        nid = Int(nbs_id[i])
        nid > 0 && push!(nbs_points[nid], i)
    end

    # (b) node_id → skeleton vertex index
    skel_node_ids = getattribute(skeleton_cloud, :node_id)
    node_to_skel  = zeros(Int, max_node)
    @inbounds for si in 1:n_skel
        nid = Int(skel_node_ids[si])
        (1 <= nid <= max_node) && (node_to_skel[nid] = si)
    end

    # (c) skeleton vertex → NBS label
    skel_to_nbs = zeros(Int, n_skel)
    @inbounds for i in 1:N
        nid = Int(node_id[i]); sid = Int(nbs_id[i])
        (nid > 0 && sid > 0) || continue
        sv = (1 <= nid <= max_node) ? node_to_skel[nid] : 0
        sv > 0 && (skel_to_nbs[sv] = sid)
    end

    # (d) NBS → skeleton vertex set
    nbs_skel_nodes = [Int[] for _ in 1:K_nbs]
    for sv in 1:n_skel
        nlab = skel_to_nbs[sv]
        nlab > 0 && push!(nbs_skel_nodes[nlab], sv)
    end

    # (e) NBS↔NBS adjacency as a symmetric SparseMatrixCSC.
    adj_rows = Int[]; adj_cols = Int[]; adj_vals = Int[]
    @inbounds for e in Graphs.edges(graph)
        a = Int(nbs_id[src(e)]); b = Int(nbs_id[dst(e)])
        (a > 0 && b > 0 && a != b) || continue
        push!(adj_rows, a); push!(adj_cols, b); push!(adj_vals, 1)
        push!(adj_rows, b); push!(adj_cols, a); push!(adj_vals, 1)
    end
    nbs_adj = sparse(adj_rows, adj_cols, adj_vals, K_nbs, K_nbs, +)

    return (nbs_points    = nbs_points,
            node_to_skel  = node_to_skel,
            skel_to_nbs   = skel_to_nbs,
            nbs_skel_nodes = nbs_skel_nodes,
            nbs_adj       = nbs_adj)
end

"""
    _seed_trees_from_nearground!(tree_id, nbs_points, agh_values, nearground_ceiling)
        -> NamedTuple

For each NBS whose minimum AGH is at or below `nearground_ceiling`, assign a fresh
tree id (1, 2, …) and mutate `tree_id` to that value for every point in the NBS.

`nbs_points::Vector{Vector{Int}}` is indexed by dense NBS id 1..K_nbs (empty entries
are allowed for ids not produced by `label_non_branching_segments`). Returns
`(nbs_tree::Vector{Int32}, assigned_nbs::BitVector, next_tree_id::Int32)`, both
indexed by NBS id (`nbs_tree[k] == 0` means unassigned).
"""
function _seed_trees_from_nearground!(tree_id::AbstractVector{Int32},
                                      nbs_points::Vector{Vector{Int}},
                                      agh_values::AbstractVector{<:Real},
                                      nearground_ceiling::Real)
    K_nbs        = length(nbs_points)
    next_tree_id = Int32(1)
    nbs_tree     = zeros(Int32, K_nbs)
    assigned_nbs = falses(K_nbs)
    ceiling_f64  = Float64(nearground_ceiling)

    @inbounds for k in 1:K_nbs
        pts = nbs_points[k]
        isempty(pts) && continue
        min_agh = Inf
        for i in pts
            v = Float64(agh_values[i])
            v < min_agh && (min_agh = v)
        end
        if min_agh <= ceiling_f64
            nbs_tree[k]     = next_tree_id
            assigned_nbs[k] = true
            for i in pts
                tree_id[i] = next_tree_id
            end
            next_tree_id += Int32(1)
        end
    end

    return (nbs_tree=nbs_tree, assigned_nbs=assigned_nbs, next_tree_id=next_tree_id)
end

"""
    _seed_largest_nbs!(tree_id, nbs_tree, assigned_nbs, nbs_points,
                       next_tree_id; cfg=_CFG) -> Int32

Find the single largest NBS by `length(nbs_points[k])` and seed it with
`next_tree_id`. Mutates `tree_id`, `nbs_tree`, `assigned_nbs` in place.
Returns the assigned tree id, or `Int32(0)` if every NBS slot is empty.

Intended as a fallback for `assemble_segments` when a connected component
has no near-ground NBS — gives a ground-disconnected CC its own
deterministic seed so step 4.2 can grow a tree through it instead of
leaving everything for the cross-component orphan rescue to clean up.
The caller is responsible for the "no near-ground seeds exist" gate; this
helper does not consult `assigned_nbs` to skip already-seeded NBS because
in its only intended call site nothing is seeded yet.
"""
function _seed_largest_nbs!(tree_id::AbstractVector{Int32},
                            nbs_tree::Vector{Int32},
                            assigned_nbs::BitVector,
                            nbs_points::Vector{Vector{Int}},
                            next_tree_id::Int32;
                            cfg::FLiPConfig=_CFG)
    K_nbs  = length(nbs_points)
    best_k = 0
    best_n = 0
    @inbounds for k in 1:K_nbs
        n = length(nbs_points[k])
        if n > best_n
            best_n = n
            best_k = k
        end
    end
    best_k == 0 && return Int32(0)

    nbs_tree[best_k]     = next_tree_id
    assigned_nbs[best_k] = true
    @inbounds for i in nbs_points[best_k]
        tree_id[i] = next_tree_id
    end
    cfg.pipeline.enable_debug_info && @info "Assembly: seeded largest NBS $best_k ($best_n points) as ungrounded tree $next_tree_id (no near-ground seed in this CC)"
    return next_tree_id
end

"""
    _check_merge_and_update_nbs!(point_idxs, tree_id, tree_nbs_id,
                                 frac, merge_threshold,
                                 tid_if_branch, tnid_if_branch,
                                 tid_if_merge,  tnid_if_merge) -> Symbol

Apply the Rule A / Rule B merge decision to every index in `point_idxs`.

- **Rule B** fires when `frac >= merge_threshold` AND `tnid_if_merge > 0`:
  `tree_id[i] = tid_if_merge`, `tree_nbs_id[i] = tnid_if_merge` for each point.
- **Rule A** otherwise (frac below threshold OR no valid merge target OR
  `enable_rule_b == false`): `tree_id[i] = tid_if_branch`,
  `tree_nbs_id[i] = tnid_if_branch` — the NBS is preserved as its own branch.

`enable_rule_b = false` disables connectivity merging entirely (every NBS stays its
own branch); used when NBS merging has already been done upstream (e.g. by the
in-stage `refine_nbs`), so assembly only groups NBS into trees.

Returns `:rule_a` or `:rule_b`. Used by `_iterative_tree_growth!` so the merge rule
has a single implementation.
"""
# Pure Rule A/B decision (no array mutation, table-testable): `:rule_b` (merge into an existing
# NBS) fires only when connectivity merging is enabled, the connected fraction meets or exceeds
# the threshold, and a valid merge target exists; otherwise `:rule_a` (NBS stays its own branch).
# The `>=` boundary matches `refine_nbs`'s Rule B so both stages decide identically at frac == threshold.
@inline _assembly_rule(frac::Float64, merge_threshold::Float64, tnid_if_merge::Integer,
                       enable_rule_b::Bool) =
    (enable_rule_b && frac >= merge_threshold && tnid_if_merge > 0) ? :rule_b : :rule_a

@inline function _check_merge_and_update_nbs!(
    point_idxs::AbstractVector{Int},
    tree_id::AbstractVector{Int32},
    tree_nbs_id::AbstractVector{Int32},
    frac::Float64,
    merge_threshold::Float64,
    tid_if_branch::Int32, tnid_if_branch::Int32,
    tid_if_merge::Int32,  tnid_if_merge::Int32,
    enable_rule_b::Bool = true,
)
    is_merge = _assembly_rule(frac, merge_threshold, tnid_if_merge, enable_rule_b) === :rule_b
    out_tid  = is_merge ? tid_if_merge  : tid_if_branch
    out_tnid = is_merge ? tnid_if_merge : tnid_if_branch
    @inbounds for i in point_idxs
        tree_id[i]     = out_tid
        tree_nbs_id[i] = out_tnid
    end
    return is_merge ? :rule_b : :rule_a
end

"""
    _iterative_tree_growth!(tree_id, tree_nbs_id, nbs_tree, assigned_nbs,
                            K_nbs, nbs_points, nbs_skel_nodes, skel_to_nbs,
                            nbs_adj, graph_skeleton, merge_threshold; cfg=_CFG) -> Int

Iteratively grow trees from the currently-seeded NBS outward through the
skeleton graph. `assigned_nbs[k] == true` marks the seed set on entry; the
function consumes it and mutates `assigned_nbs`, `nbs_tree`, per-point
`tree_id`, and per-point `tree_nbs_id` in place. Returns the number of
growth iterations performed.

`frontier_info` (built locally) maps an unassigned NBS to the set of
candidate `tree_id → connection_count` votes from its already-assigned
skeleton neighbors. Within each iteration the frontier is processed largest
NBS first; each NBS picks the highest-voted tree, then dispatches to
[`_check_merge_and_update_nbs!`](@ref) which encodes the shared Rule A /
Rule B decision:

- **Rule A** (`frac_connected < merge_threshold`) — most of the NBS's
  skeleton nodes are internal/unassigned, so it gets attached as a new
  branch of the winning tree; the NBS keeps its own `tree_nbs_id`.
- **Rule B** (`frac_connected >= merge_threshold`) — the NBS straddles an
  already-assigned NBS, so it merges into that target's `tree_nbs_id` and
  inherits its `tree_id`.

Frontier updates from within an iteration are deferred and applied
afterwards, so within one round the decisions all see the same snapshot.
"""
function _iterative_tree_growth!(tree_id::AbstractVector{Int32},
                                 tree_nbs_id::AbstractVector{Int32},
                                 nbs_tree::Vector{Int32},
                                 assigned_nbs::BitVector,
                                 K_nbs::Int,
                                 nbs_points::Vector{Vector{Int}},
                                 nbs_skel_nodes::Vector{Vector{Int}},
                                 skel_to_nbs::Vector{Int},
                                 nbs_adj::SparseMatrixCSC{Int,Int},
                                 graph_skeleton::SimpleGraph{Int},
                                 merge_threshold::Float64;
                                 cfg::FLiPConfig=_CFG,
                                 enable_rule_b::Bool=true)
    # `frontier_info` is built once and maintained INCREMENTALLY: each time an NBS
    # transitions to assigned we add its contributions to its unassigned skeleton
    # neighbors and remove it from the frontier set. Updates from within an iteration
    # are DEFERRED and applied after the iteration to match the original
    # rebuild-each-iteration semantics (later k's in the same round don't see earlier
    # k's contributions, matching the snapshot-and-process model).
    frontier_info = Dict{Int, Dict{Int32, Int}}()   # frontier_nbs → Dict(tree_id → connection_count)
    for k in 1:K_nbs
        assigned_nbs[k] || continue
        _update_frontier_after_assign!(frontier_info, k, nbs_tree[k],
                                       nbs_skel_nodes, skel_to_nbs, assigned_nbs,
                                       nbs_adj, graph_skeleton)
    end

    pending_assignments = Tuple{Int,Int32}[]
    iteration = 0
    while !isempty(frontier_info)
        iteration += 1

        # Sort frontier NBS by number of points (large → small)
        frontier_sorted = sort!(collect(keys(frontier_info));
                                by = k -> -length(nbs_points[k]))
        empty!(pending_assignments)
        n_assigned_this_round = 0

        for k in frontier_sorted
            assigned_nbs[k] && continue   # may have been assigned earlier this round

            skel_nodes_k     = nbs_skel_nodes[k]
            tree_connections = frontier_info[k]
            n_total_nodes    = length(skel_nodes_k)

            # Fused single pass over skel_nodes_k × skeleton neighbors that computes
            # BOTH `n_nodes_with_tree_conn` (for Rule A/B branch test) and
            # `skel_neighbor_counts` (for Rule B target selection). Was two separate
            # passes before — same total cost regardless of which branch fires.
            skel_neighbor_counts = Dict{Int, Int}()
            n_nodes_with_tree_conn = 0
            for sv in skel_nodes_k
                has_tree_conn = false
                for sn in Graphs.neighbors(graph_skeleton, sv)
                    nbr_nbs = skel_to_nbs[sn]
                    (nbr_nbs > 0 && nbr_nbs != k) || continue
                    skel_neighbor_counts[nbr_nbs] = get(skel_neighbor_counts, nbr_nbs, 0) + 1
                    if !has_tree_conn && assigned_nbs[nbr_nbs]
                        n_nodes_with_tree_conn += 1
                        has_tree_conn = true
                    end
                end
            end

            # Best tree = most total point-level connections
            best_tree  = Int32(0)
            best_count = 0
            for (tid, cnt) in tree_connections
                if cnt > best_count
                    best_count = cnt
                    best_tree  = tid
                end
            end
            best_tree == 0 && continue   # shouldn't happen for a valid frontier

            frac_connected = n_total_nodes > 0 ? n_nodes_with_tree_conn / n_total_nodes : 0.0

            # Rule B target: NBS with the highest skeleton-edge count (reuses the
            # `skel_neighbor_counts` from the fused loop above). `target_tnid == 0`
            # means no candidate exists, and the helper falls back to Rule A.
            target_nbs   = 0
            target_count = 0
            for (nbr_nbs, cnt) in skel_neighbor_counts
                if cnt > target_count
                    target_count = cnt
                    target_nbs   = nbr_nbs
                end
            end
            target_tid_raw = target_nbs > 0 ? nbs_tree[target_nbs] : Int32(0)
            # Split-NBS edge case: target marked -1 → treat as unassigned tree
            target_tid     = target_tid_raw == Int32(-1) ? Int32(0) : target_tid_raw
            target_tnid    = Int32(target_nbs)

            rule = _check_merge_and_update_nbs!(
                nbs_points[k], tree_id, tree_nbs_id,
                frac_connected, merge_threshold,
                best_tree,  Int32(k),       # Rule A: new branch under best_tree, keep own NBS id
                target_tid, target_tnid,    # Rule B: merge into target NBS within its tree
                enable_rule_b,
            )

            if rule === :rule_a
                nbs_tree[k]     = best_tree
                assigned_nbs[k] = true
                push!(pending_assignments, (k, best_tree))
                n_assigned_this_round += 1
            elseif target_tnid > 0
                nbs_tree[k]     = target_tid
                assigned_nbs[k] = true
                push!(pending_assignments, (k, target_tid))
                n_assigned_this_round += 1
            end
        end

        # Defer frontier updates until iteration end so within-iteration assignments
        # don't influence later k's in the same round (matches original rebuild-fresh
        # behavior; only `assigned_nbs` and per-point `tree_id` update live).
        for (k, tid) in pending_assignments
            _update_frontier_after_assign!(frontier_info, k, tid,
                                           nbs_skel_nodes, skel_to_nbs, assigned_nbs,
                                           nbs_adj, graph_skeleton)
        end

        n_assigned_this_round == 0 && break

        cfg.pipeline.enable_debug_info && @info "Assembly iteration $iteration: assigned $n_assigned_this_round NBS" total_assigned=count(assigned_nbs)
    end

    return iteration
end

"""
    _update_frontier_after_assign!(frontier_info, k, tid, nbs_skel_nodes,
                                   skel_to_nbs, assigned_nbs, nbs_adj,
                                   graph_skeleton) -> nothing

After NBS `k` transitions to assigned (tree id `tid`), remove `k` from `frontier_info`
and add its contribution to every currently-unassigned skeleton neighbor's frontier
entry. When `tid == 0` (Rule B merge into a split target with no valid tree), only
the removal happens — no tree votes are recorded.
"""
function _update_frontier_after_assign!(frontier_info::Dict{Int,Dict{Int32,Int}},
                                        k::Int, tid::Int32,
                                        nbs_skel_nodes::Vector{Vector{Int}},
                                        skel_to_nbs::Vector{Int},
                                        assigned_nbs::BitVector,
                                        nbs_adj::SparseMatrixCSC,
                                        graph_skeleton::SimpleGraph{Int})
    delete!(frontier_info, k)
    tid > 0 || return nothing

    @inbounds for sv in nbs_skel_nodes[k]
        for sn in Graphs.neighbors(graph_skeleton, sv)
            nbr_nbs = skel_to_nbs[sn]
            (nbr_nbs > 0 && !assigned_nbs[nbr_nbs]) || continue
            conn_count = nbs_adj[nbr_nbs, k]
            info = get!(frontier_info, nbr_nbs, Dict{Int32,Int}())
            info[tid] = get(info, tid, 0) + conn_count
        end
    end
    return nothing
end

# ── Step 5: occlusion-gap rescue of orphan branches ──────────────

"""
    _occluded_round_votes(frontier, coords, tree_id, tree_nbs_id, orphan_voxel_index,
                          assigned, inv_vs, link_tol2, n_thread)
        -> Dict{Tuple{Int32,Int32}, Int}

Count, for one wavefront round, the point-level links between frontier points and
still-unassigned orphan points: `(orphan tree_nbs_id, frontier tree_id) → link count`.
Each frontier point is matched against orphan points in its own voxel ±1 (the voxel size
guarantees a `≤ √link_tol2` link spans at most one voxel) under an exact squared-distance
test. Frontier points are scanned in contiguous chunks; per-chunk dicts are summed
(order-independent), so the result is deterministic.
"""
function _occluded_round_votes(frontier::Vector{Int}, coords::AbstractMatrix{<:Real},
                               tree_id::AbstractVector{Int32}, tree_nbs_id::AbstractVector{Int32},
                               orphan_voxel_index::Dict{NTuple{3,Int}, Vector{Int}},
                               assigned::Set{Int32}, inv_vs::Float64, link_tol2::Float64,
                               n_thread::Integer)
    M = length(frontier)
    M == 0 && return Dict{Tuple{Int32,Int32}, Int}()

    function count_chunk(lo, hi)
        d = Dict{Tuple{Int32,Int32}, Int}()
        @inbounds for fi in lo:hi
            f    = frontier[fi]
            ftid = tree_id[f]
            fx = float(coords[f, 1]); fy = float(coords[f, 2]); fz = float(coords[f, 3])
            vx = floor(Int, fx * inv_vs); vy = floor(Int, fy * inv_vs); vz = floor(Int, fz * inv_vs)
            for dz in -1:1, dy in -1:1, dx in -1:1
                bucket = get(orphan_voxel_index, (vx + dx, vy + dy, vz + dz), nothing)
                bucket === nothing && continue
                for c in bucket
                    branch = tree_nbs_id[c]
                    branch in assigned && continue
                    ddx = float(coords[c, 1]) - fx
                    ddy = float(coords[c, 2]) - fy
                    ddz = float(coords[c, 3]) - fz
                    (ddx * ddx + ddy * ddy + ddz * ddz) <= link_tol2 || continue
                    key = (branch, ftid)
                    d[key] = get(d, key, 0) + 1
                end
            end
        end
        return d
    end

    nt = min(Int(n_thread), M)
    (nt <= 1 || Threads.nthreads() == 1) && return count_chunk(1, M)

    chunk   = cld(M, nt)
    nchunks = cld(M, chunk)
    parts   = Vector{Dict{Tuple{Int32,Int32}, Int}}(undef, nchunks)
    _parallel_for(nchunks, nt) do c
        lo = (c - 1) * chunk + 1
        hi = min(c * chunk, M)
        parts[c] = count_chunk(lo, hi)   # distinct slot per chunk → race-free
    end
    votes = Dict{Tuple{Int32,Int32}, Int}()
    for d in parts, (k, v) in d
        votes[k] = get(votes, k, 0) + v
    end
    return votes
end

"""
    assemble_occluded_segments(coords, tree_id, tree_nbs_id; cfg) -> nothing

Rescue *orphan branches* (points with `tree_id == 0 && tree_nbs_id > 0`) into neighboring
grounded trees across occlusion gaps, via a multi-source wavefront that grows outward from
the grounded points (`tree_id > 0`).

A compact voxel index is built over the orphan points only and reused across rounds. Each
round, every current frontier point votes for the orphan branches it links to (exact point
distance `≤ occlusion_tol + sub_res`); each still-unassigned branch is then assigned, as a
whole, to the grounded `tree_id` with the most point-level links (ties broken by smaller
id), keeping its already globally-unique `tree_nbs_id`. The newly assigned points form the
next frontier, so chains of occluded segments are bridged progressively. Branches never
reached stay orphans (`tree_id == 0`). No-op when `occlusion_tol ≤ 0`.

Mutates `tree_id` in place.
"""
function assemble_occluded_segments(
    coords::AbstractMatrix{<:Real},
    tree_id::AbstractVector{Int32},
    tree_nbs_id::AbstractVector{Int32};
    cfg::FLiPConfig = _CFG,
)
    occlusion_tol = cfg.tree.assembly.occlusion_tolerance
    occlusion_tol > 0 || return nothing
    N         = size(coords, 1)
    sub_res   = cfg.pipeline.subsample_res
    link_tol2 = (occlusion_tol + sub_res)^2
    inv_vs    = 1.0 / (occlusion_tol + 2.0 * sub_res)   # voxel ≥ link distance ⇒ ±1 covers a link
    nt        = effective_nthreads(cfg)
    VK        = NTuple{3, Int}

    voxel_of(i) = (floor(Int, float(coords[i, 1]) * inv_vs),
                   floor(Int, float(coords[i, 2]) * inv_vs),
                   floor(Int, float(coords[i, 3]) * inv_vs))

    # ── orphan points (parallel scan) → voxel index + branch→points map ──
    orphan_idx = _parallel_findall(N, nt) do i
        @inbounds tree_id[i] == 0 && tree_nbs_id[i] > 0
    end
    isempty(orphan_idx) && return nothing

    orphan_voxel_index = Dict{VK, Vector{Int}}()
    orphan_pts         = Dict{Int32, Vector{Int}}()   # tree_nbs_id → point idxs (atomic unit)
    @inbounds for i in orphan_idx
        push!(get!(orphan_voxel_index, voxel_of(i), Int[]), i)
        push!(get!(orphan_pts, tree_nbs_id[i], Int[]), i)
    end
    @info "$_LOG_PREFIX   assemble_occluded_segments: $(length(orphan_pts)) orphan branches (occlusion_tol=$occlusion_tol)"

    # ── contact voxels: orphan voxels expanded by ±1 (where a grounded point may link) ──
    contact_voxels = Set{VK}()
    for vk in keys(orphan_voxel_index)
        for dz in -1:1, dy in -1:1, dx in -1:1
            push!(contact_voxels, (vk[1] + dx, vk[2] + dy, vk[3] + dz))
        end
    end

    # ── seed frontier: grounded points sitting in a contact voxel (parallel scan) ──
    frontier = _parallel_findall(N, nt) do i
        @inbounds tree_id[i] > 0 && (voxel_of(i) in contact_voxels)
    end

    assigned = Set{Int32}()   # orphan branch ids already rescued
    while !isempty(frontier)
        votes = _occluded_round_votes(frontier, coords, tree_id, tree_nbs_id,
                                      orphan_voxel_index, assigned, inv_vs, link_tol2, nt)
        isempty(votes) && break

        # per-branch argmax frontier tree_id (tie → smaller id)
        best  = Dict{Int32, Int32}()
        bestc = Dict{Int32, Int}()
        for ((branch, tid), c) in votes
            cur    = get(bestc, branch, 0)
            curtid = get(best,  branch, typemax(Int32))
            if c > cur || (c == cur && tid < curtid)
                best[branch]  = tid
                bestc[branch] = c
            end
        end
        isempty(best) && break

        newly = Int[]
        for branch in sort!(collect(keys(best)))
            tid = best[branch]
            pts = orphan_pts[branch]
            @inbounds for i in pts
                tree_id[i] = tid          # keep tree_nbs_id (already globally unique)
            end
            push!(assigned, branch)
            append!(newly, pts)
        end
        frontier = newly
    end
    return nothing
end

"""
    _relabel_tree_nbs_within_trees!(tree_id, tree_nbs_id) -> nothing

Relabel `tree_nbs_id` so that within each `tree_id` group the non-zero values form a
contiguous block ranked by descending occurrence, with the blocks laid out sequentially by
ascending `tree_id`. The `tree_id == 0` (orphan) group sorts **last**, so valid-tree
`tree_nbs_id` occupy the low labels and orphan labels are the largest. The result is
per-tree contiguous AND globally unique (each new `tree_nbs_id` maps to exactly one
`tree_id`), keeping QSM's group-by-`tree_nbs_id` correct. `tree_nbs_id == 0` stays 0.
"""
function _relabel_tree_nbs_within_trees!(tree_id::AbstractVector{Int32},
                                         tree_nbs_id::AbstractVector{Int32})
    N = length(tree_id)
    pair_count = Dict{Tuple{Int32, Int32}, Int}()
    @inbounds for i in 1:N
        tn = tree_nbs_id[i]
        tn > 0 || continue
        key = (tree_id[i], tn)
        pair_count[key] = get(pair_count, key, 0) + 1
    end
    isempty(pair_count) && return nothing

    pairs = collect(keys(pair_count))
    # tree_id asc but the orphan group (tree_id==0) sorts LAST, then count desc, tnbs asc.
    sort!(pairs; by = p -> (p[1] == 0 ? typemax(Int32) : p[1], -pair_count[p], p[2]))

    new_label = Dict{Tuple{Int32, Int32}, Int32}()
    sizehint!(new_label, length(pairs))
    next = Int32(1)
    for p in pairs
        new_label[p] = next
        next += Int32(1)
    end
    @inbounds for i in 1:N
        tn = tree_nbs_id[i]
        tn > 0 || continue
        tree_nbs_id[i] = new_label[(tree_id[i], tn)]
    end
    return nothing
end

# ── Step 7: skeleton OBJ writer ───────────────────────────────────

