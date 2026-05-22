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

    fmt = lowercase(cfg.pipeline_output_format)
    single = get_output_path(cfg.pipeline_output_dir, cfg.pipeline_output_prefix, stem, fmt)
    if isfile(single)
        println("[main] resume: loading $single")
        return read_pc(single)
    end

    scans = find_scan_outputs(cfg.pipeline_output_dir, cfg.pipeline_output_prefix, stem, fmt)
    if !isempty(scans)
        println("[main] resume: loading $(length(scans)) $stem files from $(cfg.pipeline_output_dir)")
        all_coords = Vector{Matrix{<:AbstractFloat}}(undef, length(scans))
        all_attrs  = Vector{Dict{Symbol,Vector}}(undef, length(scans))
        for (i, fpath) in enumerate(scans)
            println("[main] resume: loading $fpath")
            pc = read_pc(fpath)
            all_coords[i] = coordinates(pc)
            all_attrs[i]  = _all_attributes(pc)
        end
        merged = merge_pointclouds(all_coords, all_attrs)
        length(scans) > 1 && println("[main] resume: merged $(length(scans)) scans → $(npoints(merged)) points")
        return merged
    end

    throw(ArgumentError(
        "$consumer requires $label in memory, at $single, or as $(cfg.pipeline_output_prefix)$(stem)_S{i}.$fmt scans; no data available"))
end

"""
    _load_tree_result(cfg) -> NamedTuple

Reconstruct a tree-segmentation result NamedTuple from disk (used by stages
4 and 5 when stage 3 was disabled or skipped). Loads the tree output and the
optional skeleton; fills `filtered_cloud`, `n_components`, `neighbor_radius`
with empty / zero defaults. Throws if the tree output is not on disk.
"""
function _load_tree_result(cfg::FLiPConfig)
    pc_tree = _prepare_stage_input(nothing, cfg, "tree",
                                   "tree output", "qsm/report stage")
    fmt = lowercase(cfg.pipeline_output_format)
    skeleton_path = get_output_path(cfg.pipeline_output_dir, cfg.pipeline_output_prefix, "skeleton", fmt)
    pc_skeleton = isfile(skeleton_path) ? read_pc(skeleton_path) : pc_tree[1:0]
    return (
        pc_output=pc_tree,
        skeleton_cloud=pc_skeleton,
        filtered_cloud=pc_tree[1:0],
        n_components=0,
        neighbor_radius=0.0,
    )
end

"""
    _stage_initialization(config_path::AbstractString) -> FLiPConfig

Load the TOML config, normalize paths (expand `~/`, strip whitespace),
validate required pipeline fields, and create the output directory. Mutates
`cfg` so downstream code sees the resolved values. Returns the prepared
`FLiPConfig` (also mutates the global `_CFG` via `load_config!`).
"""
function _stage_initialization(config_path::AbstractString)
    cfg = load_config!(String(config_path))

    _expand_home(p) = startswith(strip(p), "~/") ? joinpath(homedir(), strip(p)[3:end]) : strip(p)
    cfg.pipeline_input_path    = _expand_home(cfg.pipeline_input_path)
    cfg.pipeline_output_dir    = _expand_home(cfg.pipeline_output_dir)
    cfg.pipeline_output_prefix = strip(cfg.pipeline_output_prefix)

    isempty(cfg.pipeline_input_path)    && throw(ArgumentError("pipeline.input_path must be set in config"))
    isempty(cfg.pipeline_output_dir)    && throw(ArgumentError("pipeline.output_dir must be set in config"))
    isempty(cfg.pipeline_output_prefix) && throw(ArgumentError("pipeline.output_prefix must be set in config"))
    (isfile(cfg.pipeline_input_path) || isdir(cfg.pipeline_input_path)) ||
        throw(ArgumentError("pipeline input path not found: $(cfg.pipeline_input_path)"))

    cfg.pipeline_subsample_res > 0 || throw(ArgumentError("pipeline.subsample_res must be > 0"))
    cfg.pipeline_xy_resolution > 0 || throw(ArgumentError("pipeline.xy_resolution must be > 0"))
    cfg.pipeline_idw_k >= 1        || throw(ArgumentError("pipeline.idw_k must be >= 1"))
    cfg.pipeline_idw_power > 0     || throw(ArgumentError("pipeline.idw_power must be > 0"))

    mkpath(cfg.pipeline_output_dir)
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
    fmt  = lowercase(cfg.pipeline_output_format)
    path = get_output_path(cfg.pipeline_output_dir, cfg.pipeline_output_prefix, "preprocess", fmt)
    if cfg.pipeline_enable_preprocess
        return (cloud=preprocess(; cfg=cfg), path=path, written=true)
    end
    println("[main] preprocess disabled by config")
    return (cloud=nothing, path=path, written=false)
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
    fmt = lowercase(cfg.pipeline_output_format)
    ground_path = get_output_path(cfg.pipeline_output_dir, cfg.pipeline_output_prefix, "ground", fmt)
    agh_path    = get_output_path(cfg.pipeline_output_dir, cfg.pipeline_output_prefix, "agh",    fmt)

    if !cfg.pipeline_enable_ground_segmentation
        println("[main] ground segmentation disabled by config")
        return (ground=nothing, agh=nothing,
                ground_path=ground_path, agh_path=agh_path,
                ground_written=false, agh_written=false,
                n_preprocess=0, n_ground=0)
    end

    ground_input = _prepare_stage_input(pc_preprocess, cfg, "preprocess",
                                        "preprocess output", "ground segmentation")
    res = ground_segmentation(ground_input; cfg=cfg)

    write_pc(ground_path, res.ground_points); println("[main] wrote: $ground_path")
    agh_written = false
    if cfg.pipeline_enable_agh
        write_pc(agh_path, res.agh_cloud); println("[main] wrote: $agh_path")
        agh_written = true
    end

    return (ground=res.ground_points, agh=res.agh_cloud,
            ground_path=ground_path, agh_path=agh_path,
            ground_written=true, agh_written=agh_written,
            n_preprocess=npoints(res.agh_cloud), n_ground=npoints(res.ground_points))
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
    fmt = lowercase(cfg.pipeline_output_format)
    tree_path     = get_output_path(cfg.pipeline_output_dir, cfg.pipeline_output_prefix, "tree",     fmt)
    skeleton_path = get_output_path(cfg.pipeline_output_dir, cfg.pipeline_output_prefix, "skeleton", fmt)

    if !cfg.pipeline_enable_tree_segmentation
        println("[main] tree segmentation disabled by config")
        return (result=nothing,
                tree_path=tree_path, skeleton_path=skeleton_path,
                tree_written=false, skeleton_written=false,
                n_components=0)
    end

    tree_input = _prepare_stage_input(pc_agh, cfg, "agh", "AGH output", "tree segmentation")
    GC.gc()
    res = tree_segmentation(tree_input; cfg=cfg)
    tree_input = nothing  # release input

    write_pc(tree_path,     res.pc_output);      println("[main] wrote: $tree_path")
    write_pc(skeleton_path, res.skeleton_cloud); println("[main] wrote: $skeleton_path")

    return (result=res,
            tree_path=tree_path, skeleton_path=skeleton_path,
            tree_written=true, skeleton_written=true,
            n_components=res.n_components)
end

"""
    _stage_qsm(cfg, tree_res, config_path) -> NamedTuple

Run the QSM stage (or `(status=:skipped,)` if disabled). If `tree_res` is
`nothing` and QSM is enabled, the tree result is reconstructed from disk
via `_load_tree_result`.
"""
function _stage_qsm(cfg::FLiPConfig, tree_res, config_path::AbstractString)
    cfg.pipeline_enable_qsm || return (status=:skipped,)
    tr = isnothing(tree_res) ? _load_tree_result(cfg) : tree_res
    fmt = lowercase(cfg.pipeline_output_format)
    tree_path = get_output_path(cfg.pipeline_output_dir, cfg.pipeline_output_prefix, "tree", fmt)
    return qsm(tree_result=tr,
               config_path=String(config_path),
               output_dir=cfg.pipeline_output_dir,
               output_prefix=cfg.pipeline_output_prefix,
               tree_cloud_path=tree_path)
end

"""
    _stage_report(cfg, tree_res, qsm_res, config_path) -> NamedTuple

Run the report stage (or `(status=:skipped,)` if disabled). If `tree_res` is
`nothing` and the report stage is enabled, the tree result is reconstructed
from disk via `_load_tree_result`.
"""
function _stage_report(cfg::FLiPConfig, tree_res, qsm_res, config_path::AbstractString)
    cfg.pipeline_enable_generate_report || return (status=:skipped,)
    tr = isnothing(tree_res) ? _load_tree_result(cfg) : tree_res
    return generate_report(tree_result=tr,
                           qsm_result=qsm_res,
                           config_path=String(config_path),
                           output_dir=cfg.pipeline_output_dir,
                           output_prefix=cfg.pipeline_output_prefix)
end

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
    input_path    = cfg.pipeline_input_path
    output_dir    = cfg.pipeline_output_dir
    output_prefix = cfg.pipeline_output_prefix
    output_fmt    = lowercase(cfg.pipeline_output_format)

    # 1) preprocess
    pp = _stage_preprocess(cfg)

    # 2) ground segmentation
    g = _stage_ground(cfg, pp.cloud)
    pp = (cloud=nothing, path=pp.path, written=pp.written)  # release preprocess cloud
    pc_agh = g.agh

    # Locals retained for use by downstream stages and the summary builder
    preprocess_path    = pp.path
    preprocess_written = pp.written
    ground_path        = g.ground_path
    agh_path           = g.agh_path
    ground_written     = g.ground_written
    agh_written        = g.agh_written
    n_preprocess       = g.n_preprocess
    n_ground           = g.n_ground

    # 3) tree segmentation
    t = _stage_tree(cfg, pc_agh)
    pc_agh = nothing  # released — _stage_tree consumed it

    # 4) qsm
    q = _stage_qsm(cfg, t.result, config_path)

    # 5) report
    r = _stage_report(cfg, t.result, q, config_path)

    # Locals retained by the summary builder
    tree_path             = t.tree_path
    tree_skeleton_path    = t.skeleton_path
    tree_written          = t.tree_written
    tree_skeleton_written = t.skeleton_written
    tree_components       = t.n_components
    qsm_res    = q
    report_res = r

    # Release heavy data before returning lightweight summary
    t = nothing
    GC.gc()

    return (
        input_path=input_path,
        output_dir=output_dir,
        output_prefix=output_prefix,
        n_preprocess_input=n_preprocess,
        n_preprocess=n_preprocess,
        n_ground=n_ground,
        tree_components=tree_components,
        preprocess_path=preprocess_path,
        ground_path=ground_path,
        agh_path=agh_path,
        tree_path=tree_path,
        skeleton_path=tree_skeleton_path,
        preprocess_written=preprocess_written,
        ground_written=ground_written,
        agh_written=agh_written,
        tree_written=tree_written,
        tree_skeleton_written=tree_skeleton_written,
        qsm_result=qsm_res,
        report_result=report_res,
    )
end
