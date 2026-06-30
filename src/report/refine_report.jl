"""
NBS-refinement reports: CSV writers for whole-NBS Rule-B merges and node-level volume moves
(emitted under `enable_debug_info`).
"""

function _write_node_merge_report(path::String, moves::Vector{NodeMove})
    ord = sortperm(moves; by = mv -> mv.trial_node_id)
    n = length(ord)
    tid = Vector{Int}(undef, n); fnb = Vector{Int32}(undef, n); tnb = Vector{Int32}(undef, n)
    rat = Vector{Float64}(undef, n); npt = Vector{Int}(undef, n)
    cmp = Vector{Float64}(undef, n); agv = Vector{Float64}(undef, n)
    for (k, j) in enumerate(ord)
        mv = moves[j]
        tid[k] = mv.trial_node_id; fnb[k] = mv.from_nbs; tnb[k] = mv.to_nbs
        rat[k] = mv.overlap_ratio; npt[k] = mv.n_points; cmp[k] = mv.completeness; agv[k] = mv.agh
    end
    headers = ["trial_node_id", "from_nbs", "to_nbs", "node_overlap_ratio",
               "node_n_points", "node_completeness", "node_agh"]
    _write_csv(path, AbstractVector[tid, fnb, tnb, rat, npt, cmp, agv], headers)
end

function _write_rule_b_report(path::String, moves::Vector{RuleBMove})
    n = length(moves)
    dn = Vector{Int32}(undef, n); rn = Vector{Int32}(undef, n)
    dv = Vector{Float64}(undef, n); rv = Vector{Float64}(undef, n)
    fr = Vector{Float64}(undef, n); ns = Vector{Int}(undef, n)
    for (k, mv) in enumerate(moves)
        dn[k] = mv.donor_nbs; rn[k] = mv.receiver_nbs; dv[k] = mv.donor_vol
        rv[k] = mv.receiver_vol; fr[k] = mv.frac_connected; ns[k] = mv.n_donor_skel_nodes
    end
    headers = ["donor_nbs", "receiver_nbs", "donor_vol", "receiver_vol",
               "frac_connected", "n_donor_skel_nodes"]
    _write_csv(path, AbstractVector[dn, rn, dv, rv, fr, ns], headers)
end


"""
    RefineReportSink()

Thread-safe accumulator for per-CC `refine_nbs` outcomes, so the tree-segmentation
orchestrator never touches `RuleBMove`/`NodeMove` or the CSV writers directly. `record!`
tallies move counts always and (under `debug`) appends the move rows under a lock;
`write_refine_reports` emits the two debug CSVs.
"""
mutable struct RefineReportSink
    lock::ReentrantLock
    rule_b::Vector{RuleBMove}
    node_moves::Vector{NodeMove}
    n_moves::Threads.Atomic{Int}
    n_rule_b::Threads.Atomic{Int}
end
RefineReportSink() = RefineReportSink(ReentrantLock(), RuleBMove[], NodeMove[],
                                      Threads.Atomic{Int}(0), Threads.Atomic{Int}(0))

function record!(sink::RefineReportSink, r; debug::Bool)
    Threads.atomic_add!(sink.n_moves,  r.n_nodes_moved)
    Threads.atomic_add!(sink.n_rule_b, r.n_rule_b_merges)
    if debug && (!isempty(r.rule_b_moves) || !isempty(r.node_moves))
        lock(sink.lock) do
            append!(sink.rule_b, r.rule_b_moves)
            append!(sink.node_moves, r.node_moves)
        end
    end
    return nothing
end

function write_refine_reports(sink::RefineReportSink, dir::AbstractString, prefix::AbstractString)
    isempty(dir) && return nothing
    isempty(sink.rule_b)     || _write_rule_b_report(joinpath(dir, "$(prefix)nbs_ruleB_merge_report.csv"), sink.rule_b)
    isempty(sink.node_moves) || _write_node_merge_report(joinpath(dir, "$(prefix)nbs_merge_report.csv"), sink.node_moves)
    return nothing
end
