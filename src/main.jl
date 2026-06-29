"""
    run_pipeline(config_path::AbstractString=_DEFAULT_CONFIG_PATH)

Run FLiP main pipeline stages in order:
1. load config
2. preprocess
3. ground_segmentation
4. tree_segmentation
5. qsm
6. generate_report
"""
function run_pipeline(config_path::AbstractString=_DEFAULT_CONFIG_PATH)
    cfg = _stage_initialization(config_path)
    _log_session_info(cfg)

    pp_output = _stage_preprocess(cfg)
    g_output  = _stage_ground(cfg, pp_output.cloud);    pp_output = _drop_preprocess_clouds(pp_output)
    t_output  = _stage_tree(cfg, g_output.agh);         g_output  = _drop_ground_clouds(g_output)
    q_output   = _stage_qsm(cfg, t_output.result, config_path)
    ref_output = _stage_qsm_refinement(cfg, t_output.result, q_output, config_path)
    q2_output  = _stage_qsm(cfg, ref_output.tree_result, config_path;
                            out_prefix=ref_output.out_prefix, tree_stem="tree_refined",
                            label="qsm (refined)", run=ref_output.refined)
    qsm_final  = ref_output.refined ? q2_output : q_output
    r_output   = _stage_report(cfg, ref_output.tree_result, qsm_final, config_path)
    t_output   = _drop_tree_clouds(t_output)
    ref_output = _drop_refinement_clouds(ref_output); GC.gc()

    return _summarize(cfg, pp_output, g_output, t_output, q_output, ref_output, q2_output, r_output)
end

# ── Stage functions ────────────────────────────────────────────────

"""
    _stage_initialization(config_path::AbstractString) -> FLiPConfig

Load the TOML config, normalize paths (expand `~/`, strip whitespace),
validate required pipeline fields, and create the output directory. Mutates
`cfg` so downstream code sees the resolved values. Returns the prepared
`FLiPConfig` (also mutates the global `_CFG` via `load_config!`).
"""
function _stage_initialization(config_path::AbstractString)
    cfg = load_config!(String(config_path))

    cfg.pipeline.input_path    = expanduser(strip(cfg.pipeline.input_path))
    cfg.pipeline.output_dir    = expanduser(strip(cfg.pipeline.output_dir))
    cfg.pipeline.output_prefix = strip(cfg.pipeline.output_prefix)

    isempty(cfg.pipeline.input_path)    && throw(ArgumentError("pipeline.input_path must be set in config"))
    isempty(cfg.pipeline.output_dir)    && throw(ArgumentError("pipeline.output_dir must be set in config"))
    isempty(cfg.pipeline.output_prefix) && throw(ArgumentError("pipeline.output_prefix must be set in config"))
    (isfile(cfg.pipeline.input_path) || isdir(cfg.pipeline.input_path)) ||
        throw(ArgumentError("pipeline input path not found: $(cfg.pipeline.input_path)"))

    cfg.pipeline.subsample_res > 0 || throw(ArgumentError("pipeline.subsample_res must be > 0"))
    cfg.pipeline.xy_resolution > 0 || throw(ArgumentError("pipeline.xy_resolution must be > 0"))
    cfg.pipeline.idw_k >= 1        || throw(ArgumentError("pipeline.idw_k must be >= 1"))
    cfg.pipeline.idw_power > 0     || throw(ArgumentError("pipeline.idw_power must be > 0"))

    mkpath(cfg.pipeline.output_dir)
    return cfg
end

"""
    _stage_preprocess(cfg) -> (cloud, path, written)

Run the preprocess stage (or skip it if disabled). The output cloud and the
per-scan `_S{i}` files are written to disk by `preprocess(; cfg)` itself.

Returns a NamedTuple `(cloud::Union{Nothing,PointCloud}, path::String,
written::Bool)`: the merged cloud (or `nothing` if disabled), the canonical
single-file output path, and whether the stage produced output.
"""
function _stage_preprocess(cfg::FLiPConfig)
    fmt  = lowercase(cfg.pipeline.output_format)
    path = get_output_path(cfg.pipeline.output_dir, cfg.pipeline.output_prefix, "preprocess", fmt)
    if !cfg.pipeline.enable_preprocess
        _log_stage_skipped("preprocess")
        return (cloud=nothing, path=path, written=false)
    end
    return _with_stage_timing("preprocess") do
        (cloud=preprocess(; cfg=cfg), path=path, written=true)
    end
end

"""
    _stage_ground(cfg, pc_preprocess) -> NamedTuple

Run ground segmentation, write the ground and optional AGH outputs, and
return the resulting clouds plus path / written-flag / point-count metadata.
If `pc_preprocess` is `nothing`, the input is loaded from disk via
`_prepare_stage_input`.

Returns `(ground, agh, ground_path, agh_path, ground_written, agh_written,
n_preprocess, n_ground)`.
"""
function _stage_ground(cfg::FLiPConfig, pc_preprocess)
    fmt = lowercase(cfg.pipeline.output_format)
    ground_path = get_output_path(cfg.pipeline.output_dir, cfg.pipeline.output_prefix, "ground", fmt)
    agh_path    = get_output_path(cfg.pipeline.output_dir, cfg.pipeline.output_prefix, "agh",    fmt)

    if !cfg.pipeline.enable_ground_segmentation
        _log_stage_skipped("ground_segmentation")
        return (ground=nothing, agh=nothing,
                ground_path=ground_path, agh_path=agh_path,
                ground_written=false, agh_written=false,
                n_preprocess=0, n_ground=0)
    end

    return _with_stage_timing("ground_segmentation") do
        ground_input = _prepare_stage_input(pc_preprocess, cfg, "preprocess",
                                            "preprocess output", "ground segmentation")
        res = ground_segmentation(ground_input; cfg=cfg)

        write_pc(ground_path, res.ground_points); @info "$_LOG_PREFIX   wrote: $ground_path"
        agh_written = false
        if cfg.pipeline.enable_agh
            write_pc(agh_path, res.agh_cloud); @info "$_LOG_PREFIX   wrote: $agh_path"
            agh_written = true
        end

        (ground=res.ground_points, agh=res.agh_cloud,
         ground_path=ground_path, agh_path=agh_path,
         ground_written=true, agh_written=agh_written,
         n_preprocess=npoints(res.agh_cloud), n_ground=npoints(res.ground_points))
    end
end

"""
    _stage_tree(cfg, pc_agh) -> NamedTuple

Run tree segmentation, write the tree and skeleton outputs, and return the
full `tree_segmentation` result plus path / written-flag / n_components
metadata. If `pc_agh` is `nothing`, the input is loaded from disk via
`_prepare_stage_input`.

Returns `(result, tree_path, skeleton_path, tree_written, skeleton_written,
n_components)`. `result` is `nothing` when the stage is disabled.
"""
function _stage_tree(cfg::FLiPConfig, pc_agh)
    fmt = lowercase(cfg.pipeline.output_format)
    tree_path     = get_output_path(cfg.pipeline.output_dir, cfg.pipeline.output_prefix, "tree",     fmt)
    skeleton_path = get_output_path(cfg.pipeline.output_dir, cfg.pipeline.output_prefix, "skeleton", fmt)

    if !cfg.pipeline.enable_tree_segmentation
        _log_stage_skipped("tree_segmentation")
        return (result=nothing,
                tree_path=tree_path, skeleton_path=skeleton_path,
                tree_written=false, skeleton_written=false,
                n_components=0)
    end

    return _with_stage_timing("tree_segmentation") do
        tree_input = _prepare_stage_input(pc_agh, cfg, "agh", "AGH output", "tree segmentation")
        GC.gc()
        res = tree_segmentation(tree_input; cfg=cfg)
        tree_input = nothing  # release input

        write_pc(tree_path, res.pc_output); @info "$_LOG_PREFIX   wrote: $tree_path"
        skel_written = cfg.pipeline.enable_skeleton_output
        if skel_written
            write_pc(skeleton_path, res.skeleton_cloud); @info "$_LOG_PREFIX   wrote: $skeleton_path"
        end

        (result=res,
         tree_path=tree_path, skeleton_path=skeleton_path,
         tree_written=true, skeleton_written=skel_written,
         n_components=res.n_components)
    end
end

"""
    _stage_qsm(cfg, tree_res, config_path; out_prefix, tree_stem, label, run) -> NamedTuple

Run the QSM stage (or `(status=:skipped,)` if disabled or `run=false`). If
`tree_res` is `nothing`, the tree result is reconstructed from disk via
`_load_tree_result(cfg; stem=tree_stem)`.

The keyword args let the pipeline run QSM twice without duplicated code: the
first (canonical) pass uses the defaults (`tree_stem="tree"`,
`out_prefix=output_prefix`); the post-refinement pass passes
`tree_stem="tree_refined"` and a refined `out_prefix` so its CSV / surface
outputs are written under distinct names, preserving the pre-merge results.
"""
function _stage_qsm(cfg::FLiPConfig, tree_res, config_path::AbstractString;
                    out_prefix::AbstractString=cfg.pipeline.output_prefix,
                    tree_stem::AbstractString="tree",
                    label::AbstractString="qsm",
                    run::Bool=true)
    if !cfg.pipeline.enable_qsm || !run
        _log_stage_skipped(label)
        return (status=:skipped,)
    end
    return _with_stage_timing(label) do
        tr = isnothing(tree_res) ? _load_tree_result(cfg; stem=tree_stem) : tree_res
        fmt = lowercase(cfg.pipeline.output_format)
        tree_path = get_output_path(cfg.pipeline.output_dir, cfg.pipeline.output_prefix, tree_stem, fmt)
        qsm(tree_result=tr,
            config_path=String(config_path),
            output_dir=cfg.pipeline.output_dir,
            output_prefix=out_prefix,
            tree_cloud_path=tree_path)
    end
end

"""
    _stage_qsm_refinement(cfg, tree_res, qsm_res, config_path) -> NamedTuple

Post-QSM refinement stage (or `(status=:skipped, …)` if disabled). Merges
overlapping tree_nbs segments via [`nbs_merge_by_volume_overlap`] and, when a
merge is applied, writes the relabeled `{prefix}tree_refined.{fmt}` cloud. Does
NOT re-run QSM — `run_pipeline` does that explicitly as a second `_stage_qsm`
call. Returns `(status, tree_result, refined, report_path, refined_tree_path,
out_prefix)`; `tree_result` is the (possibly relabeled) tree result to feed the
second QSM, and `out_prefix` names that pass's outputs.
"""
function _stage_qsm_refinement(cfg::FLiPConfig, tree_res, qsm_res, config_path::AbstractString)
    base_prefix = cfg.pipeline.output_prefix
    if !cfg.pipeline.enable_qsm_refinement
        _log_stage_skipped("qsm_refinement")
        return (status=:skipped, tree_result=tree_res, refined=false,
                report_path="", refined_tree_path="", out_prefix=base_prefix)
    end
    return _with_stage_timing("qsm_refinement") do
        nodes = if qsm_res !== nothing && hasproperty(qsm_res, :nodes) && !isempty(qsm_res.nodes)
            qsm_res.nodes
        else
            _read_qsm_nodes_csv(joinpath(cfg.pipeline.output_dir, "$(base_prefix)qsm_nodes.csv"))
        end
        if isempty(nodes)
            @warn "$_LOG_PREFIX qsm_refinement: no QSM nodes available (run QSM first); skipping"
            return (status=:no_nodes, tree_result=tree_res, refined=false,
                    report_path="", refined_tree_path="", out_prefix=base_prefix)
        end
        tr = isnothing(tree_res) ? _load_tree_result(cfg) : tree_res
        pc = tr.pc_output
        mres = nbs_merge_by_volume_overlap(pc=pc, nodes=nodes, cfg=cfg,
                                           output_dir=cfg.pipeline.output_dir,
                                           output_prefix=base_prefix)
        if !mres.merged
            return (status=mres.status, tree_result=tr, refined=false,
                    report_path=mres.report_csv_path, refined_tree_path="", out_prefix=base_prefix)
        end
        fmt = lowercase(cfg.pipeline.output_format)
        refined_path = get_output_path(cfg.pipeline.output_dir, base_prefix, "tree_refined", fmt)
        write_pc(refined_path, pc)
        @info "$_LOG_PREFIX   wrote refined tree cloud: $refined_path"
        return (status=:success, tree_result=merge(tr, (pc_output=pc,)), refined=true,
                report_path=mres.report_csv_path, refined_tree_path=refined_path,
                out_prefix="$(base_prefix)refined_")
    end
end

"""
    _stage_report(cfg, tree_res, qsm_res, config_path) -> NamedTuple

Run the report stage (or `(status=:skipped,)` if disabled). If `tree_res` is
`nothing` and the report stage is enabled, the tree result is reconstructed
from disk via `_load_tree_result`.
"""
function _stage_report(cfg::FLiPConfig, tree_res, qsm_res, config_path::AbstractString)
    if !cfg.pipeline.enable_generate_report
        _log_stage_skipped("generate_report")
        return (status=:skipped,)
    end
    return _with_stage_timing("generate_report") do
        tr = isnothing(tree_res) ? _load_tree_result(cfg) : tree_res
        generate_report(tree_result=tr,
                        qsm_result=qsm_res,
                        config_path=String(config_path),
                        output_dir=cfg.pipeline.output_dir,
                        output_prefix=cfg.pipeline.output_prefix)
    end
end

# ── Resume helpers (used by the stage functions) ──────────────────

"""
    _prepare_stage_input(data, cfg, stem, label, consumer) -> PointCloud

Materialize a stage's PointCloud input. If `data` is non-nothing, return it
unchanged. Otherwise, try the single-file output `{prefix}{stem}.{fmt}` on
disk; if that's missing, try the multi-file `{prefix}{stem}_S{i}.{fmt}`
pattern and merge. Throws if no source is available.
"""
function _prepare_stage_input(data, cfg::FLiPConfig, stem::AbstractString,
                              label::AbstractString, consumer::AbstractString)
    isnothing(data) || return data

    fmt = lowercase(cfg.pipeline.output_format)
    single = get_output_path(cfg.pipeline.output_dir, cfg.pipeline.output_prefix, stem, fmt)
    if isfile(single)
        @info "$_LOG_PREFIX   resume: loading $single"
        return read_pc(single)
    end

    scans = find_scan_outputs(cfg.pipeline.output_dir, cfg.pipeline.output_prefix, stem, fmt)
    if !isempty(scans)
        @info "$_LOG_PREFIX   resume: loading $(length(scans)) $stem files from $(cfg.pipeline.output_dir)"
        T = coord_type(cfg)
        all_coords = Vector{Matrix{T}}(undef, length(scans))
        all_attrs  = Vector{Dict{Symbol,Vector}}(undef, length(scans))
        for (i, fpath) in enumerate(scans)
            cfg.pipeline.enable_debug_info && @info "$_LOG_PREFIX     resume: loading $fpath"
            pc = read_pc(fpath)
            all_coords[i] = coordinates(pc)
            all_attrs[i]  = _all_attributes(pc)
        end
        merged = merge_pointclouds(all_coords, all_attrs; verbose=cfg.pipeline.enable_debug_info)
        length(scans) > 1 && @info "$_LOG_PREFIX   resume: merged $(length(scans)) scans → $(npoints(merged)) points"
        return merged
    end

    throw(ArgumentError(
        "$consumer requires $label in memory, at $single, or as $(cfg.pipeline.output_prefix)$(stem)_S{i}.$fmt scans; no data available"))
end

"""
    _load_tree_result(cfg) -> NamedTuple

Reconstruct a tree-segmentation result NamedTuple from disk (used by stages
4 and 5 when stage 3 was disabled or skipped). Loads the tree output and the
optional skeleton; fills `filtered_cloud`, `n_components`, `neighbor_radius`
with empty / zero defaults. Throws if the tree output is not on disk.
"""
function _load_tree_result(cfg::FLiPConfig; stem::AbstractString="tree")
    pc_tree = _prepare_stage_input(nothing, cfg, stem,
                                   "tree output", "qsm/report stage")
    fmt = lowercase(cfg.pipeline.output_format)
    skeleton_path = get_output_path(cfg.pipeline.output_dir, cfg.pipeline.output_prefix, "skeleton", fmt)
    pc_skeleton = isfile(skeleton_path) ? read_pc(skeleton_path) : pc_tree[1:0]
    return (
        pc_output=pc_tree,
        skeleton_cloud=pc_skeleton,
        filtered_cloud=pc_tree[1:0],
        n_components=0,
        neighbor_radius=0.0,
    )
end

# ── Memory release + summary (used by run_pipeline) ───────────────

# Each _drop_*_clouds nulls the heavy point-cloud fields in the stage's
# NamedTuple so the GC can reclaim memory, while paths/written-flags/counts
# survive for the summary. Using `merge` keeps these in lock-step with any
# new fields added to the stage return shape.
_drop_preprocess_clouds(pp) = merge(pp, (cloud=nothing,))
_drop_ground_clouds(g)      = merge(g,  (ground=nothing, agh=nothing))
_drop_tree_clouds(t)        = merge(t,  (result=nothing,))
_drop_refinement_clouds(r)  = merge(r,  (tree_result=nothing,))

"""
    _summarize(cfg, pp_output, g_output, t_output, q_output, ref_output, q2_output, r_output) -> NamedTuple

Build the stage-grouped summary returned by `run_pipeline`. Each stage's
paths, written-flags, and counts live in their own sub-NamedTuple
(`preprocess`, `ground`, `agh`, `tree`, `refinement`). `qsm` is the **final**
QSM result (the post-refinement pass when a merge was applied, else the
first-pass result); `qsm_prelim` is always the first pass; `report` is forwarded
as-is.
"""
function _summarize(cfg::FLiPConfig, pp_output, g_output, t_output, q_output, ref_output, q2_output, r_output)
    qsm_final = ref_output.refined ? q2_output : q_output
    return (
        config = (
            input_path    = cfg.pipeline.input_path,
            output_dir    = cfg.pipeline.output_dir,
            output_prefix = cfg.pipeline.output_prefix,
        ),
        preprocess = (path=pp_output.path,
                      written=pp_output.written,
                      n_points=g_output.n_preprocess),
        ground     = (path=g_output.ground_path,
                      written=g_output.ground_written,
                      n_points=g_output.n_ground),
        agh        = (path=g_output.agh_path,
                      written=g_output.agh_written),
        tree       = (path=t_output.tree_path,
                      skeleton_path=t_output.skeleton_path,
                      written=t_output.tree_written,
                      skeleton_written=t_output.skeleton_written,
                      n_components=t_output.n_components),
        qsm        = qsm_final,
        qsm_prelim = q_output,
        refinement = (status=ref_output.status,
                      refined=ref_output.refined,
                      report_path=ref_output.report_path,
                      refined_tree_path=ref_output.refined_tree_path),
        report     = r_output,
    )
end
