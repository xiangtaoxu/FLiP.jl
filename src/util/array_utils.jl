"""
Generic array utilities used by point-cloud filtering.

Functions:
- `_uf_find!(parent, i)`                        — union-find root lookup with path halving
- `_uf_union!(parent, ranks, i, j)`             — union-find union by rank
- `_filter_array_by_occurrence(arr, min_count=1)`
                                                — re-label values by frequency rank; drop values below `min_count`
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
    _filter_array_by_occurrence(arr::AbstractVector{T},
                                min_count::Integer=1) -> Vector{Int}

Re-label every element of `arr` by the frequency rank of its value.

Unique values in `arr` are ranked by descending occurrence count
(ties broken by `isless` on the value). Each occurrence in `arr` is
mapped to its rank index (`1` = most common, `2` = second most common,
…). Values whose total count is below `min_count` are mapped to `0`,
effectively filtering them out of the labelling.

Used after a union-find pass to convert a `parent` vector (with each
element pointing to its component root) into compact component labels
sized by component support.

Returns a fresh `Vector{Int}` the same length as `arr`; does not
mutate the input.
"""
function _filter_array_by_occurrence(arr::AbstractVector{T},
                                     min_count::Integer=1) where {T}
    min_count >= 1 || throw(ArgumentError("min_count must be >= 1"))

    counts = Dict{T,Int}()
    for v in arr
        counts[v] = get(counts, v, 0) + 1
    end

    ranked = sort(collect(keys(counts)); by = v -> (-counts[v], v))

    label_of = Dict{T,Int}()
    next_label = 1
    for v in ranked
        if counts[v] >= min_count
            label_of[v] = next_label
            next_label += 1
        end
    end

    labels = Vector{Int}(undef, length(arr))
    @inbounds for (k, v) in enumerate(arr)
        labels[k] = get(label_of, v, 0)
    end
    return labels
end
