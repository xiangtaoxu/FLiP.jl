"""
Generic array utilities used by point-cloud filtering and segmentation.

Functions:
- `_uf_find!(parent, i)`                        — union-find root lookup with path halving
- `_uf_union!(parent, ranks, i, j)`             — union-find union by rank
- `relabel_by_occurrence(arr, min_count=1; positive_only, T_out)`
                                                — re-label values by frequency rank; drop below `min_count`
- `group_indices_by_label(indices, labels)`     — group integer indices by per-position labels
"""

"""
    _uf_find!(parent::Vector{Int}, i::Int) -> Int

Return the root of element `i` in a union-find / disjoint-set forest
encoded by `parent`. `parent[k] == k` marks a root; otherwise
`parent[k]` points to a parent element.

Applies *path halving* on the way up — every other node along the
search path is re-pointed at its grandparent — which keeps the
amortized cost of `find` near O(1) without a separate full-path
compaction pass.
"""
function _uf_find!(parent::Vector{Int}, i::Int)
    @inbounds while parent[i] != i
        # Path halving: re-point i at its grandparent on the way up.
        parent[i] = parent[parent[i]]
        i = parent[i]
    end
    return i
end

"""
    _uf_union!(parent::Vector{Int}, ranks::Vector{Int}, i::Int, j::Int)

Merge the sets containing `i` and `j` in a union-find forest. No-op if
they're already in the same set.

Uses *union by rank*: the shallower tree is hung beneath the deeper
tree's root, so the merged tree's depth only grows when both inputs
had equal rank. `ranks[r]` is an upper bound on the depth of the tree
rooted at `r` (not the exact height — path halving in `_uf_find!` can
shorten trees without updating `ranks`).
"""
function _uf_union!(parent::Vector{Int}, ranks::Vector{Int}, i::Int, j::Int)
    root_i = _uf_find!(parent, i)
    root_j = _uf_find!(parent, j)
    root_i == root_j && return

    # Union by rank: ensure root_i names the deeper tree, then hang
    # root_j under root_i.
    if ranks[root_i] < ranks[root_j]
        root_i, root_j = root_j, root_i
    end
    parent[root_j] = root_i

    # Rank only grows when both subtrees were equally deep.
    if ranks[root_i] == ranks[root_j]
        ranks[root_i] += 1
    end
end

"""
    relabel_by_occurrence(arr::AbstractVector{T}, min_count::Integer=1;
                          positive_only::Bool=false,
                          T_out::Type{<:Integer}=Int) -> Vector{T_out}

Re-label every element of `arr` by the frequency rank of its value.

Unique values in `arr` are ranked by descending occurrence count (ties broken by
`isless` on the value). Each occurrence is mapped to its rank index (`1` = most
common, `2` = second most common, …). Values whose total count is below
`min_count` are mapped to `0`.

When `positive_only=true`, non-positive values (≤ 0) are treated as the "drop"
sentinel and map to `0` in the output regardless of count. Use this for label
vectors where `0` means "unassigned" and negatives mean "discarded".

`T_out` controls the element type of the output vector.

Returns a fresh `Vector{T_out}` the same length as `arr`; does not mutate input.
"""
function relabel_by_occurrence(arr::AbstractVector{T}, min_count::Integer=1;
                               positive_only::Bool=false,
                               T_out::Type{<:Integer}=Int) where {T}
    min_count >= 1 || throw(ArgumentError("min_count must be >= 1"))

    counts = Dict{T,Int}()
    @inbounds for v in arr
        positive_only && !(Int(v) > 0) && continue
        counts[v] = get(counts, v, 0) + 1
    end

    ranked = sort(collect(keys(counts)); by = v -> (-counts[v], v))

    label_of = Dict{T,T_out}()
    next_label = T_out(1)
    for v in ranked
        counts[v] >= min_count || continue
        label_of[v] = next_label
        next_label += T_out(1)
    end

    out = Vector{T_out}(undef, length(arr))
    @inbounds for (k, v) in enumerate(arr)
        if positive_only && !(Int(v) > 0)
            out[k] = T_out(0)
        else
            out[k] = get(label_of, v, T_out(0))
        end
    end
    return out
end

"""
    group_indices_by_label(indices::AbstractVector{<:Integer},
                           labels::AbstractVector{<:Integer};
                           max_label::Int = maximum(labels; init=0))
        -> Vector{Vector{Int}}

Group `indices` by their per-position integer `labels` (same length). Returns one
index vector per non-empty positive label, ordered by label value. Positions with
label `0` are dropped. When `labels` was produced by a routine that already ranks
labels (e.g. `connected_component_subset!` ranks by component size), the output
preserves that order.
"""
function group_indices_by_label(indices::AbstractVector{<:Integer},
                                labels::AbstractVector{<:Integer};
                                max_label::Int = maximum(labels; init=0))
    length(indices) == length(labels) ||
        throw(ArgumentError("indices and labels must have the same length"))
    max_label == 0 && return Vector{Vector{Int}}()
    clusters = [Int[] for _ in 1:max_label]
    @inbounds for (i, v) in enumerate(indices)
        lab = Int(labels[i])
        lab > 0 && push!(clusters[lab], Int(v))
    end
    return filter!(!isempty, clusters)
end
