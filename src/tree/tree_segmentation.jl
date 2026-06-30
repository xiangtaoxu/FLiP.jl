"""
Tree-segmentation orchestrator: per-connected-component pipeline that labels NBS, runs a trial
model + NBS refinement, assembles trees, then rescues orphans and relabels. Entry `tree_segmentation`.
"""

"""
    tree_segmentation(pc::PointCloud; cfg::FLiPConfig=_CFG) -> NamedTuple

Run the individual-tree segmentation workflow on a point cloud already
carrying `:AGH` (above-ground height).

Per-component processing keeps peak memory at O(N_largest_component × avg_degree)
rather than O(N_total × avg_degree). Components are discovered by a lightweight
coordinate-only union-find before any graph is built.

# Returns

`NamedTuple` with fields:
- `filtered_cloud::PointCloud` — the above-ground points with `:nbs_id`,
  `:tree_id`, `:tree_nbs_id` attached (the persisted `:node_id` is added later by
  the final QSM stage; under `enable_debug_info` the transient skeleton `:node_id`
  and `:trial_node_id` are also attached)
- `pc_output::PointCloud` — alias of `filtered_cloud` (back-compat)
- `n_components::Int` — number of valid connected components processed
- `neighbor_radius::Float64` — the radius used for component discovery /
  per-component graph build

Internally: Pass 1 labels NBS per component; a global lean trial QSM fits per-NBS
cylinders; Pass 2 rebuilds each component's skeleton, runs `refine_nbs` (when
`tree.refine.enable`), then assembles trees (Rule B off); finally
the orphan rescue + dense relabels run.
"""
# Reserve a contiguous global id block of size `local_max` (returns the pre-increment offset).
@inline _reserve_block!(c::Threads.Atomic{Int}, local_max::Int) = Int32(Threads.atomic_add!(c, local_max))

# Scatter a CC-local label vector into a global array at this CC's point slots, offset by `off`
# (0 stays 0). The reservation order is thread-completion order, but every consumer groups by
# label value and a final `relabel_by_occurrence` canonicalizes, so the result is deterministic.
@inline function _scatter_offset!(g::Vector{Int32}, cc_idx::Vector{Int},
                                  local_v::AbstractVector{<:Integer}, off::Int32)
    @inbounds for (k, gi) in enumerate(cc_idx)
        v = local_v[k]
        g[gi] = v > 0 ? Int32(v) + off : Int32(0)
    end
    return g
end

# Build the skeleton-vertex → NBS map for one CC: each skeleton vertex (carrying a CC-local
# `node_id`) takes the NBS of the points in that node.
function _skel_to_nbs_map(local_node::AbstractVector{Int32}, cc_nbs::AbstractVector{Int32},
                          skel_node_ids::AbstractVector)
    max_ln = Int(maximum(local_node; init=Int32(0)))
    node_to_nbs = zeros(Int32, max_ln)
    @inbounds for li in eachindex(local_node)
        ln = Int(local_node[li]); nb = cc_nbs[li]
        (ln > 0 && nb > 0) && (node_to_nbs[ln] = nb)
    end
    return Int32[(1 <= Int(ln) <= max_ln) ? node_to_nbs[Int(ln)] : Int32(0) for ln in skel_node_ids]
end

function tree_segmentation(pc::PointCloud; cfg::FLiPConfig=_CFG)
    hasattribute(pc, :AGH) || throw(ArgumentError(
        "tree_segmentation requires AGH attribute on input point cloud"))

    agh = getattribute(pc, :AGH)

    # ── Step 1: filter to above-ground points ─────────────────────
    threshold = cfg.tree.extraction.nearground_agh_threshold
    nearground_idx = findall(i -> isfinite(float(agh[i])) && float(agh[i]) > threshold,
                             eachindex(agh))
    empty_result = (
        filtered_cloud  = pc[1:0],
        pc_output       = pc[1:0],
        n_components    = 0,
        neighbor_radius = 0.0,
    )
    isempty(nearground_idx) && return empty_result

    pc_filtered     = pc[nearground_idx]
    coords_filtered = coordinates(pc_filtered)
    agh_filtered    = float.(getattribute(pc_filtered, :AGH))
    N = size(coords_filtered, 1)

    neighbor_radius = cfg.tree.extraction.neighbor_radius > 0 ?
                      cfg.tree.extraction.neighbor_radius :
                      2.0 * cfg.pipeline.subsample_res
    neighbor_radius > 0 || throw(ArgumentError("tree neighbor radius must be > 0"))

    # ── Step 2: discover components via coordinate-only union-find ──
    cc_labels = connected_component_labels(coords_filtered, neighbor_radius,
                                           cfg.tree.extraction.min_nbs_size)
    K_cc = Int(maximum(cc_labels; init=0))
    cc_indices_by_id = [Int[] for _ in 1:K_cc]
    @inbounds for i in eachindex(cc_labels)
        lab = cc_labels[i]
        lab > 0 && push!(cc_indices_by_id[lab], i)
    end
    cc_labels = Int[]   # free per-point label vector (~ 8N bytes)

    n_components = K_cc
    @info "$_LOG_PREFIX   $N filtered points → $n_components connected components (min_cc=$(cfg.tree.extraction.min_nbs_size))"
    n_components == 0 && return empty_result

    nthread = effective_nthreads(cfg)
    debug   = cfg.pipeline.enable_debug_info

    # ── Pass 1 (parallel per CC): NBS labeling only ───────────────
    # Produces global `nbs_id` (atomic-offset, used to group the trial QSM) plus a
    # CC-local skeleton `node_id` per component (transient — the skeleton + assembly
    # need it, but it is not the persisted node id; the final QSM emits that).
    global_nbs_id = zeros(Int32, N)
    cc_node_ids   = Vector{Vector{Int32}}(undef, n_components)
    nbs_counter   = Threads.Atomic{Int}(0)
    _parallel_for(n_components, nthread) do ci
        @inbounds begin
            cc_indices = cc_indices_by_id[ci]
            cc_coords  = coords_filtered[cc_indices, :]
            cc_agh     = agh_filtered[cc_indices]
            g_res   = build_radius_graph(cc_coords, neighbor_radius; weights=false)
            nbs_res = label_non_branching_segments(g_res.graph, cc_coords, cc_agh; cfg=cfg)
            local_nbs_max = Int(maximum(nbs_res.nbs_id; init=Int32(0)))
            _scatter_offset!(global_nbs_id, cc_indices, nbs_res.nbs_id,
                             _reserve_block!(nbs_counter, local_nbs_max))
            cc_node_ids[ci] = Int32.(nbs_res.node_id)
        end
    end

    # ── Global lean trial QSM (grouped by nbs_id) → cylinders for refinement ──
    do_refine   = cfg.tree.refine.enable
    trial_nodes = QSMNode[]
    if do_refine
        setattribute!(pc_filtered, :nbs_id, global_nbs_id)
        trial = model_nbs(pc=pc_filtered, cfg=cfg, group_attr=:nbs_id,
                          node_id_attr=:trial_node_id, emit_surface=false)
        trial_nodes = trial.nodes
    end
    # Partition trial nodes by CC (each nbs belongs to exactly one CC).
    nbs_to_cc = zeros(Int32, Int(nbs_counter[]))
    @inbounds for ci in 1:n_components, gi in cc_indices_by_id[ci]
        nb = global_nbs_id[gi]
        nb > 0 && (nbs_to_cc[nb] = Int32(ci))
    end
    cc_trial_nodes = [QSMNode[] for _ in 1:n_components]
    for nd in trial_nodes
        (nd.nbs_id > 0 && nd.nbs_id <= length(nbs_to_cc)) || continue
        ci = nbs_to_cc[nd.nbs_id]
        ci > 0 && push!(cc_trial_nodes[ci], nd)
    end
    trial_node_id_global = (do_refine && hasattribute(pc_filtered, :trial_node_id)) ?
        Int32.(getattribute(pc_filtered, :trial_node_id)) : zeros(Int32, N)

    # ── Pass 2 (parallel per CC): refine NBS (rebuilt skeleton) then assemble ──
    # The skeleton is rebuilt here (transient, never tracked across passes). Trees
    # use their own atomic id blocks; assembly runs with Rule B disabled because NBS
    # merging is done by `refine_nbs`.
    global_node_id     = zeros(Int32, N)   # CC-local node id, globally offset (debug only)
    global_tree_id     = zeros(Int32, N)
    global_tree_nbs_id = zeros(Int32, N)
    node_counter     = Threads.Atomic{Int}(0)
    tree_counter     = Threads.Atomic{Int}(0)
    tree_nbs_counter = Threads.Atomic{Int}(0)
    sink             = RefineReportSink()

    _parallel_for(n_components, nthread) do ci
        @inbounds begin
            cc_indices = cc_indices_by_id[ci]
            cc_coords  = coords_filtered[cc_indices, :]
            cc_agh     = agh_filtered[cc_indices]
            local_node = cc_node_ids[ci]
            cc_nbs     = Int32[global_nbs_id[gi] for gi in cc_indices]

            g_res    = build_radius_graph(cc_coords, neighbor_radius; weights=false)
            skel_res = create_skeleton_cloud(g_res.graph, cc_coords, local_node)

            if do_refine && !isempty(cc_trial_nodes[ci])
                skel_to_nbs = _skel_to_nbs_map(local_node, cc_nbs,
                                               getattribute(skel_res.skeleton_cloud, :node_id))
                cc_trial_pp = Int32[trial_node_id_global[gi] for gi in cc_indices]
                refine_res = refine_nbs(nbs_id=cc_nbs, node_id=local_node, trial_node_id=cc_trial_pp,
                                  nodes=cc_trial_nodes[ci], graph_skeleton=skel_res.graph_skeleton,
                                  skel_to_nbs=skel_to_nbs, cfg=cfg)
                cc_nbs = refine_res.nbs_id
                _scatter_offset!(global_nbs_id, cc_indices, cc_nbs, Int32(0))  # already global ids
                record!(sink, refine_res; debug=debug)
            end

            # Assembly array-indexes by max(nbs_id); the refined `cc_nbs` carries (sparse,
            # large) GLOBAL ids, so relabel to a dense CC-local block just for this call.
            # The global nbs labeling is stored separately above.
            cc_nbs_local = relabel_by_occurrence(cc_nbs; positive_only=true, T_out=Int32)
            assembled = assemble_segments(g_res.graph, cc_coords, cc_nbs_local, local_node, cc_agh,
                                    skel_res.graph_skeleton, skel_res.skeleton_cloud;
                                    cfg=cfg, enable_rule_b=false)

            _scatter_offset!(global_node_id, cc_indices, local_node,
                             _reserve_block!(node_counter,     Int(maximum(local_node;      init=Int32(0)))))
            _scatter_offset!(global_tree_id, cc_indices, assembled.tree_id,
                             _reserve_block!(tree_counter,     Int(maximum(assembled.tree_id;     init=Int32(0)))))
            _scatter_offset!(global_tree_nbs_id, cc_indices, assembled.tree_nbs_id,
                             _reserve_block!(tree_nbs_counter, Int(maximum(assembled.tree_nbs_id; init=Int32(0)))))
        end
    end

    empty!(cc_indices_by_id)

    # ── Step 5: rescue orphan branches into neighboring grounded trees across
    # occlusion gaps (orphan ⟺ tree_id==0 && tree_nbs_id>0) ──
    assemble_occluded_segments(coords_filtered, global_tree_id, global_tree_nbs_id; cfg=cfg)

    # ── Step 6: dense relabels — tree_id by point count, tree_nbs_id contiguous
    # within trees, and nbs_id densified (refinement leaves gaps where NBS merged) ──
    global_tree_id = relabel_by_occurrence(global_tree_id; positive_only=true, T_out=Int32)
    tree_offset    = Int32(maximum(global_tree_id; init=0))
    _relabel_tree_nbs_within_trees!(global_tree_id, global_tree_nbs_id)
    global_nbs_id  = relabel_by_occurrence(global_nbs_id; positive_only=true, T_out=Int32)

    # ── Step 7: attach attributes ─────────────────────────────────
    # The persisted node id comes from the FINAL QSM stage, so `:node_id` is NOT set
    # here. Transient ids (skeleton node_id, trial_node_id) are kept only under debug.
    setattribute!(pc_filtered, :nbs_id,      global_nbs_id)
    setattribute!(pc_filtered, :tree_id,     global_tree_id)
    setattribute!(pc_filtered, :tree_nbs_id, global_tree_nbs_id)
    if debug
        setattribute!(pc_filtered, :node_id, global_node_id)
        write_refine_reports(sink, cfg.pipeline.output_dir, cfg.pipeline.output_prefix)
    else
        haskey(pc_filtered.attrs, :trial_node_id) && delete!(pc_filtered.attrs, :trial_node_id)
    end

    @info "$_LOG_PREFIX   n_components=$n_components, n_trees=$tree_offset, " *
          "n_nbs=$(Int(maximum(global_nbs_id; init=Int32(0))))" *
          (do_refine ? " (refined: $(sink.n_moves[]) node moves, $(sink.n_rule_b[]) NBS merges)" : "")

    return (
        filtered_cloud  = pc_filtered,
        pc_output       = pc_filtered,
        n_components    = Int(n_components),
        neighbor_radius = neighbor_radius,
    )
end

# ── Step 3b: NBS labeling ─────────────────────────────────────────

