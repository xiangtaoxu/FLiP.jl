"""
Non-Branching Segment labeling: greedy linearity-constrained expansion from near-ground seeds.
"""

"""
    label_non_branching_segments(graph, points, agh_values; cfg) -> NamedTuple

Segment every vertex of `graph` into non-branching segments (NBS) by greedy
neighborhood expansion. Connected components smaller than `cfg.tree.extraction.min_nbs_size`
are discarded upfront (label 0). Valid segments are relabeled by descending
size (largest segment → label 1).

Returns `(nbs_id::Vector{Int32}, node_id::Vector{Int32})`.
"""
function label_non_branching_segments(
    graph::SimpleGraph{Int},
    points::AbstractMatrix{<:Real},
    agh_values::AbstractVector{<:Real};
    cfg::FLiPConfig = _CFG,
)
    N = nv(graph)
    size(points, 1) == N     || throw(ArgumentError("graph vertex count must match number of points"))
    length(agh_values) == N  || throw(ArgumentError("agh_values length must match graph vertex count"))

    min_segment_size       = cfg.tree.extraction.min_nbs_size
    neighbor_distance      = cfg.tree.extraction.nbs_neighbor_distance
    nearground_agh_ceiling = cfg.tree.extraction.nearground_agh_threshold + 2.0 * cfg.pipeline.subsample_res

    global_nbs_id  = zeros(Int, N)
    global_node_id = zeros(Int, N)
    gsws           = GreedySearchWorkspace(N)
    unlabeled_mask = trues(N)

    # Pre-pass: discard isolated clusters smaller than min_segment_size (O(V+E)).
    for comp in connected_components(graph)
        length(comp) >= min_segment_size && continue
        @inbounds for v in comp
            global_nbs_id[v]  = -1
            unlabeled_mask[v] = false
        end
    end

    # Pre-allocated buffers for _find_seed_clusters (reused across iterations)
    seed_candidate_buf = sizehint!(Int[], 1024)
    frontier_buf       = sizehint!(Int[], 1024)   # rejected frontier CCs from greedy searches

    # Pre-sort by ascending z for fallback seed selection.
    z_sorted = sortperm(view(points, :, 3); rev=false)
    z_cursor = Ref(1)

    next_id          = 1
    next_global_node = 1
    n_labeled_total  = count(!, unlabeled_mask)   # already labeled (discarded small CCs)
    # Debug-gated progress: only constructed (and only logged) when the flag is on.
    nbs_progress = cfg.pipeline.enable_debug_info ?
                   ProgressReporter("NBS labeling", N) : nothing

    while true
        seed_clusters = _find_seed_clusters(
            points, agh_values, nearground_agh_ceiling,
            unlabeled_mask, graph,
            gsws.cc_ws;
            is_first_iteration = (next_id == 1),
            z_sorted       = z_sorted,
            z_cursor       = z_cursor,
            frontier_buf   = frontier_buf,
            candidate_buf  = seed_candidate_buf,
        )
        isempty(seed_clusters) && break   # no unlabeled points remain

        for start_vertices in seed_clusters
            # Skip if any point was already claimed by an earlier search this round
            any_claimed = false
            @inbounds for v in start_vertices
                if !unlabeled_mask[v]
                    any_claimed = true
                    break
                end
            end
            any_claimed && continue

            result = greedy_neighborhood_search(
                graph, start_vertices, neighbor_distance;
                vertex_mask          = unlabeled_mask,
                workspace            = gsws,
                points               = points,
                linearity_angle_deg  = Float64(cfg.tree.extraction.linearity_angle_deg),
                min_frontier_cc_size = Int(cfg.tree.extraction.frontier_min_cc_size),
            )
            labeled_idx = result.vertices
            node_ids    = result.node_ids
            n_labeled   = length(labeled_idx)

            # Collect rejected frontier CCs as seeds for future iterations
            append!(frontier_buf, result.rejected_frontier)

            if n_labeled < min_segment_size
                @inbounds for v in labeled_idx
                    global_nbs_id[v]  = -1
                    unlabeled_mask[v] = false
                end
            else
                offset = next_global_node - 1
                @inbounds for i in 1:n_labeled
                    v = labeled_idx[i]
                    global_nbs_id[v]  = next_id
                    global_node_id[v] = node_ids[i] + offset
                    unlabeled_mask[v] = false
                end
                next_global_node += result.max_node_id
                next_id          += 1
            end

            n_labeled_total += n_labeled
        end

        nbs_progress !== nothing && report!(nbs_progress, n_labeled_total; extra="nbs=$(next_id-1)")
    end

    # Relabel valid segments by descending size; -1 (discarded) and 0 (unprocessed)
    # both fall through `positive_only=true` to 0.
    nbs_relabeled = relabel_by_occurrence(global_nbs_id; positive_only=true, T_out=Int32)
    return (nbs_id = nbs_relabeled, node_id = Int32.(global_node_id))
end

"""
    _find_seed_clusters(points, agh_values, nearground_agh_ceiling, unlabeled_mask,
                        graph, cc_workspace;
                        is_first_iteration, z_sorted, z_cursor,
                        frontier_buf, candidate_buf) -> Vector{Vector{Int}}

Find seed clusters for `label_non_branching_segments`. Returns a list of vertex-index
vectors sorted by descending cluster size (first iteration) or ascending z (subsequent).

**First iteration**: collects all unlabeled vertices with AGH ≤ `nearground_agh_ceiling`,
splits them into connected components via `connected_component_subset!`, and returns
them ranked largest-first.

**Subsequent iterations**: draws seeds from `frontier_buf` — rejected frontier CCs
accumulated from prior `greedy_neighborhood_search` calls. Filters out already-labeled
vertices, sorts the remainder by ascending z, and returns each as a single-point seed
cluster. When `frontier_buf` is exhausted, falls back to the next lowest-z unlabeled
point via `z_cursor` (O(1) amortised). Returns an empty vector when no unlabeled points
remain (signals termination).
"""
function _find_seed_clusters(
    points::AbstractMatrix{<:Real},
    agh_values::AbstractVector{<:Real},
    nearground_agh_ceiling::Float64,
    unlabeled_mask::BitVector,
    graph::SimpleGraph{Int},
    cc_workspace::ConnectedComponentSubsetWorkspace;
    is_first_iteration::Bool,
    z_sorted::Vector{Int},
    z_cursor::Ref{Int},
    frontier_buf::Vector{Int},
    candidate_buf::Vector{Int},
)
    empty!(candidate_buf)

    if is_first_iteration
        # Collect all unlabeled near-ground vertices
        @inbounds for v in eachindex(unlabeled_mask)
            unlabeled_mask[v] || continue
            float(agh_values[v]) <= nearground_agh_ceiling || continue
            push!(candidate_buf, v)
        end
        if !isempty(candidate_buf)
            labels = connected_component_subset!(cc_workspace, graph, candidate_buf)
            return group_indices_by_label(candidate_buf, labels)
        end
    else
        # Draw seeds from the rejected-frontier buffer.
        # Filter out vertices that have been labeled since they were added.
        @inbounds for v in frontier_buf
            unlabeled_mask[v] && push!(candidate_buf, v)
        end
        empty!(frontier_buf)

        if !isempty(candidate_buf)
            # Sort by ascending z so growth proceeds upward from branch points
            sort!(candidate_buf; by = v -> @inbounds(points[v, 3]))
            return Vector{Int}[Int[v] for v in candidate_buf]
        end
    end

    # Fallback: advance z_cursor to next unlabeled point (O(1) amortised)
    while z_cursor[] <= length(z_sorted)
        v = z_sorted[z_cursor[]]
        z_cursor[] += 1
        if unlabeled_mask[v]
            return Vector{Int}[Int[v]]
        end
    end

    # No unlabeled points remain — signal termination
    return Vector{Vector{Int}}()
end

# ── Step 3c: skeleton construction ────────────────────────────────

