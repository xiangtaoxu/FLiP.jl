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

    # 1) preprocess (reads input files, preprocesses each, writes individual outputs)
    preprocess_path = get_output_path(output_dir, output_prefix, "preprocess", output_fmt)
    preprocess_written = false
    pc_preprocess = nothing
    if cfg.pipeline_enable_preprocess
        pc_preprocess = preprocess(; cfg=cfg)
        preprocess_written = true
    else
        println("[main] preprocess disabled by config")
    end

    # 2) ground segmentation
    ground_points = nothing
    pc_agh = nothing
    ground_path = get_output_path(output_dir, output_prefix, "ground", output_fmt)
    agh_path = get_output_path(output_dir, output_prefix, "agh", output_fmt)
    ground_written = false
    agh_written = false

    n_preprocess = 0
    n_ground = 0

    if cfg.pipeline_enable_ground_segmentation
        ground_input = _prepare_stage_input(pc_preprocess, cfg, "preprocess", "preprocess output", "ground segmentation")
        pc_preprocess = nothing  # release input — no longer needed
        ground_res = ground_segmentation(ground_input; cfg=cfg)
        ground_input = nothing
        ground_points = ground_res.ground_points
        pc_agh = ground_res.agh_cloud
        n_preprocess = npoints(ground_res.agh_cloud)
        n_ground = npoints(ground_points)
        write_pc(ground_path, ground_points); println("[main] wrote: $ground_path"); ground_written = true
        ground_points = nothing  # written to disk, release
        if cfg.pipeline_enable_agh
            write_pc(agh_path, pc_agh); println("[main] wrote: $agh_path"); agh_written = true
        end
    else
        println("[main] ground segmentation disabled by config")
        pc_preprocess = nothing  # no downstream consumer, release
    end

    # 3) tree segmentation
    tree_path = get_output_path(output_dir, output_prefix, "tree", output_fmt)
    tree_skeleton_path = get_output_path(output_dir, output_prefix, "skeleton", output_fmt)
    tree_written = false
    tree_skeleton_written = false
    tree_components = 0

    tree_res = nothing
    if cfg.pipeline_enable_tree_segmentation
        tree_input = _prepare_stage_input(pc_agh, cfg, "agh", "AGH output", "tree segmentation")
        pc_agh = nothing  # release — tree_input holds the reference
        GC.gc()
        tree_res = tree_segmentation(tree_input; cfg=cfg)
        tree_input = nothing  # release input
        tree_components = tree_res.n_components
        write_pc(tree_path, tree_res.pc_output); println("[main] wrote: $tree_path"); tree_written = true
        write_pc(tree_skeleton_path, tree_res.skeleton_cloud); println("[main] wrote: $tree_skeleton_path"); tree_skeleton_written = true
    else
        println("[main] tree segmentation disabled by config")
        pc_agh = nothing  # no downstream consumer, release
    end

    # 4) qsm
    if isnothing(tree_res) && cfg.pipeline_enable_qsm
        tree_res = _load_tree_result(cfg)
    end

    qsm_res = if cfg.pipeline_enable_qsm
        qsm(tree_result=tree_res, config_path=String(config_path), output_dir=output_dir, output_prefix=output_prefix, tree_cloud_path=tree_path)
    else
        (status=:skipped,)
    end

    # 5) report
    if isnothing(tree_res) && cfg.pipeline_enable_generate_report
        tree_res = _load_tree_result(cfg)
    end

    report_res = if cfg.pipeline_enable_generate_report
        generate_report(
            tree_result=tree_res,
            qsm_result=qsm_res,
            config_path=String(config_path),
            output_dir=output_dir,
            output_prefix=output_prefix,
        )
    else
        (status=:skipped,)
    end

    # Release heavy data before returning lightweight summary
    tree_res = nothing
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
