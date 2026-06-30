"""
Per-tree biometric aggregation: node-level and tree-level result tables (DBH, volume,
surface area) from the fitted QSM nodes.
"""

"""
    _build_node_table(nodes) -> (columns, headers, vol, sa)

Build the node-level results as a vector of typed per-column vectors (each
element of `columns` is one column, in `headers` order). Also returns the
per-node frustum `vol` and `sa` vectors directly so `_build_tree_table` can
aggregate them without round-tripping through a stringly-typed lookup.
"""
function _build_node_table(nodes::Vector{QSMNode})
    isempty(nodes) && return (AbstractVector[], String[], Float64[], Float64[])

    # Group by NBS for frustum computation (NBS ids are dense from tree_segmentation)
    nbs_ids_per_node = Int32[nd.nbs_id for nd in nodes]
    nbs_groups = group_indices_by_label(1:length(nodes), nbs_ids_per_node)

    n = length(nodes)
    vol = zeros(n)
    sa = zeros(n)

    for idxs in nbs_groups
        sort!(idxs; by=i -> nodes[i].agh)
        nn = length(idxs)
        for k in 1:nn
            nd = nodes[idxs[k]]
            h = nd.height
            if nn == 1 || k == 1 || k == nn
                r1 = nd.radius_area
                if k == 1 && nn > 1
                    r2 = nodes[idxs[k+1]].radius_area
                    vol[idxs[k]], sa[idxs[k]] = _frustum_metrics(r1, r2, h)
                elseif k == nn && nn > 1
                    r0 = nodes[idxs[k-1]].radius_area
                    vol[idxs[k]], sa[idxs[k]] = _frustum_metrics(r0, r1, h)
                else
                    vol[idxs[k]], sa[idxs[k]] = _frustum_metrics(r1, r1, h)
                end
            else
                r0 = nodes[idxs[k-1]].radius_area
                r1 = nd.radius_area
                r2 = nodes[idxs[k+1]].radius_area
                ra = (r0 + r1) / 2
                rb = (r1 + r2) / 2
                vol[idxs[k]], sa[idxs[k]] = _frustum_metrics(ra, rb, h)
            end
        end
    end

    headers = [
        "qsm_node_id", "tree_nbs_id", "tree_id", "agh",
        "cross_area", "circumference",
        "radius_area", "radius_circ",
        "height", "volume", "surface_area",
        "completeness", "n_points",
        "center_x", "center_y", "center_z",
        "direction_x", "direction_y", "direction_z",
    ]

    columns = AbstractVector[
        Int[nd.qsm_node_id for nd in nodes],
        Int32[nd.nbs_id for nd in nodes],
        Int32[nd.tree_id for nd in nodes],
        Float64[nd.agh for nd in nodes],
        Float64[nd.cross_area for nd in nodes],
        Float64[nd.circumference for nd in nodes],
        Float64[nd.radius_area for nd in nodes],
        Float64[nd.radius_circ for nd in nodes],
        Float64[nd.height for nd in nodes],
        vol,
        sa,
        Float64[nd.completeness for nd in nodes],
        Int[nd.n_points for nd in nodes],
        Float64[nd.center_x for nd in nodes],
        Float64[nd.center_y for nd in nodes],
        Float64[nd.center_z for nd in nodes],
        Float64[nd.direction_x for nd in nodes],
        Float64[nd.direction_y for nd in nodes],
        Float64[nd.direction_z for nd in nodes],
    ]

    return (columns, headers, vol, sa)
end

"""
    _build_tree_table(nodes, vol, sa, cfg) -> (columns, headers)

Aggregate node-level results to tree-level biometrics. `vol` and `sa` are the
per-node frustum volume / surface area vectors returned by `_build_node_table`
(parallel to `nodes`).
"""
function _build_tree_table(nodes::Vector{QSMNode},
                           vol::Vector{Float64}, sa::Vector{Float64},
                           cfg::FLiPConfig)
    isempty(nodes) && return (AbstractVector[], String[])

    # Group by tree_id (dense ids from tree_segmentation); output ordered by tree_id ascending
    tree_ids_per_node = Int32[nd.tree_id for nd in nodes]
    tree_groups = group_indices_by_label(1:length(nodes), tree_ids_per_node)
    n_trees = length(tree_groups)
    bh = cfg.tree.model.breast_height

    tree_headers = [
        "tree_id",
        "volume", "surface_area", "height",
        "dbh_area", "dbh_circ",
        "n_points", "n_nodes",
        "x", "y",
    ]

    col_tree_id  = Vector{Int32}(undef, n_trees)
    col_volume   = Vector{Float64}(undef, n_trees)
    col_surface  = Vector{Float64}(undef, n_trees)
    col_height   = Vector{Float64}(undef, n_trees)
    col_dbh_a    = Vector{Float64}(undef, n_trees)
    col_dbh_c    = Vector{Float64}(undef, n_trees)
    col_n_points = Vector{Int}(undef, n_trees)
    col_n_nodes  = Vector{Int}(undef, n_trees)
    col_x        = Vector{Float64}(undef, n_trees)
    col_y        = Vector{Float64}(undef, n_trees)

    # Embarrassingly parallel: each tree writes a distinct row of every column.
    # `tree_groups`, `nodes`, `vol`, `sa` are read-only here.
    _parallel_for(n_trees, effective_nthreads(cfg)) do ti
        idxs = tree_groups[ti]
        total_vol = sum(i -> vol[i], idxs)
        total_sa  = sum(i -> sa[i],  idxs)
        total_pts = sum(i -> nodes[i].n_points, idxs)
        max_agh   = maximum(i -> nodes[i].agh, idxs)

        # DBH: find node closest to breast height
        best_bh_idx = idxs[1]
        best_bh_dist = abs(nodes[idxs[1]].agh - bh)
        for i in idxs
            d = abs(nodes[i].agh - bh)
            if d < best_bh_dist
                best_bh_dist = d
                best_bh_idx = i
            end
        end

        @inbounds begin
            col_tree_id[ti]  = nodes[idxs[1]].tree_id
            col_volume[ti]   = total_vol
            col_surface[ti]  = total_sa
            col_height[ti]   = max_agh
            col_dbh_a[ti]    = 2.0 * nodes[best_bh_idx].radius_area
            col_dbh_c[ti]    = 2.0 * nodes[best_bh_idx].radius_circ
            col_n_points[ti] = total_pts
            col_n_nodes[ti]  = length(idxs)
            col_x[ti]        = nodes[best_bh_idx].center_x
            col_y[ti]        = nodes[best_bh_idx].center_y
        end
    end

    columns = AbstractVector[
        col_tree_id, col_volume, col_surface, col_height,
        col_dbh_a, col_dbh_c, col_n_points, col_n_nodes,
        col_x, col_y,
    ]
    return (columns, tree_headers)
end


"""
    write_biometrics(nodes, cfg; output_dir, output_prefix) -> (n_trees, node_csv_path, tree_csv_path)

Aggregate fitted QSM `nodes` into node- and tree-level biometric tables and write
`{prefix}qsm_nodes.csv` / `{prefix}qsm_trees.csv` (only when `output_dir` is non-empty).
"""
function write_biometrics(nodes::Vector{QSMNode}, cfg::FLiPConfig;
                          output_dir::AbstractString, output_prefix::AbstractString)
    node_columns, node_headers, vol, sa = _build_node_table(nodes)
    tree_columns, tree_headers = _build_tree_table(nodes, vol, sa, cfg)
    n_nodes_out = isempty(node_columns) ? 0 : length(first(node_columns))
    n_trees     = isempty(tree_columns) ? 0 : length(first(tree_columns))
    node_csv = joinpath(output_dir, "$(output_prefix)qsm_nodes.csv")
    tree_csv = joinpath(output_dir, "$(output_prefix)qsm_trees.csv")
    if !isempty(output_dir)
        mkpath(output_dir)
        n_nodes_out > 0 && _write_csv(node_csv, node_columns, node_headers)
        n_trees > 0     && _write_csv(tree_csv, tree_columns, tree_headers)
        @info "$_LOG_PREFIX   wrote biometrics: $n_nodes_out nodes, $n_trees trees"
    end
    return (n_trees=n_trees, node_csv_path=node_csv, tree_csv_path=tree_csv)
end
