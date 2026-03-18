@inline function _pipeline_output_path(output_dir::AbstractString, output_prefix::AbstractString, stem::AbstractString)
    return joinpath(output_dir, "$(output_prefix)$(stem).las")
end

@inline function _pipeline_write(path::AbstractString, pc::PointCloud)
    write_las(path, pc)
    println("[main] wrote: $path")
    return true
end

@inline function _pipeline_load(path::AbstractString, label::AbstractString)
    isfile(path) || throw(ArgumentError("$label not found: $path"))
    println("[main] resume: loading $(path)")
    return read_las(path)
end

@inline function _pipeline_require(data, path::AbstractString, data_label::AbstractString, consumer_label::AbstractString)
    if !isnothing(data)
        return data
    end
    if isfile(path)
        return _pipeline_load(path, data_label)
    end
    throw(ArgumentError("$consumer_label requires $data_label in memory or at $path; no data available"))
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
    cfg = load_config!(String(config_path))

    _expand(p) = startswith(strip(p), "~/") ? joinpath(homedir(), strip(p)[3:end]) : strip(p)
    input_path = _expand(cfg.pipeline_input_path)
    output_dir = _expand(cfg.pipeline_output_dir)
    output_prefix = strip(cfg.pipeline_output_prefix)

    isempty(input_path) && throw(ArgumentError("pipeline.input_path must be set in config"))
    isempty(output_dir) && throw(ArgumentError("pipeline.output_dir must be set in config"))
    isempty(output_prefix) && throw(ArgumentError("pipeline.output_prefix must be set in config"))
    isfile(input_path) || throw(ArgumentError("pipeline input file not found: $input_path"))

    cfg.pipeline_subsample_res > 0 || throw(ArgumentError("pipeline.subsample_res must be > 0"))
    cfg.pipeline_xy_resolution > 0 || throw(ArgumentError("pipeline.xy_resolution must be > 0"))
    cfg.pipeline_idw_k >= 1 || throw(ArgumentError("pipeline.idw_k must be >= 1"))
    cfg.pipeline_idw_power > 0 || throw(ArgumentError("pipeline.idw_power must be > 0"))

    mkpath(output_dir)

    pc_input = read_las(input_path)

    # 1) preprocess
    preprocess_path = _pipeline_output_path(output_dir, output_prefix, "preprocess")
    preprocess_written = false
    pc_preprocess = nothing
    if cfg.pipeline_enable_preprocess
        pc_preprocess = preprocess(pc_input; cfg=cfg)
        preprocess_written = _pipeline_write(preprocess_path, pc_preprocess)
    else
        println("[main] preprocess disabled by config")
        if isfile(preprocess_path)
            pc_preprocess = _pipeline_load(preprocess_path, "preprocess output")
            preprocess_written = true
        end
    end

    # 2) ground segmentation
    ground_points = nothing
    pc_agh = nothing
    ground_path = _pipeline_output_path(output_dir, output_prefix, "ground")
    agh_path = _pipeline_output_path(output_dir, output_prefix, "agh")
    ground_written = false
    agh_written = false

    if cfg.pipeline_enable_ground_segmentation
        ground_input = _pipeline_require(pc_preprocess, preprocess_path, "preprocess output", "ground segmentation")
        ground_res = ground_segmentation(ground_input; cfg=cfg)
        ground_points = ground_res.ground_points
        pc_agh = ground_res.agh_cloud
        ground_written = _pipeline_write(ground_path, ground_points)
        if cfg.pipeline_enable_agh
            agh_written = _pipeline_write(agh_path, pc_agh)
        end
    else
        println("[main] ground segmentation disabled by config")
        if isfile(ground_path)
            ground_points = _pipeline_load(ground_path, "ground output")
            ground_written = true
        end
        if isfile(agh_path)
            pc_agh = _pipeline_load(agh_path, "AGH output")
            agh_written = true
        end
    end

    # 3) tree segmentation
    tree_path = _pipeline_output_path(output_dir, output_prefix, "tree")
    tree_skeleton_path = _pipeline_output_path(output_dir, output_prefix, "skeleton")
    tree_written = false
    tree_skeleton_written = false
    tree_components = 0

    tree_res = nothing
    if cfg.pipeline_enable_tree_segmentation
        tree_input = _pipeline_require(pc_agh, agh_path, "AGH output", "tree segmentation")
        tree_res = tree_segmentation(tree_input; cfg=cfg)
        tree_components = tree_res.n_components
        tree_written = _pipeline_write(tree_path, tree_res.pc_output)
        tree_skeleton_written = _pipeline_write(tree_skeleton_path, tree_res.skeleton_cloud)
    else
        println("[main] tree segmentation disabled by config")
    end

    # 4) qsm
    if isnothing(tree_res) && cfg.pipeline_enable_qsm
        pc_tree = _pipeline_require(nothing, tree_path, "tree output", "qsm")
        pc_skeleton = isfile(tree_skeleton_path) ? _pipeline_load(tree_skeleton_path, "tree skeleton output") : pc_tree[1:0]
        tree_res = (
            pc_output=pc_tree,
            skeleton_cloud=pc_skeleton,
            filtered_cloud=pc_tree[1:0],
            n_components=0,
            neighbor_radius=0.0,
        )
    end

    qsm_res = if cfg.pipeline_enable_qsm
        qsm(tree_result=tree_res, config_path=String(config_path), output_dir=output_dir, output_prefix=output_prefix)
    else
        (status=:skipped,)
    end

    # 5) report
    if isnothing(tree_res) && cfg.pipeline_enable_generate_report
        pc_tree = _pipeline_require(nothing, tree_path, "tree output", "generate_report")
        pc_skeleton = isfile(tree_skeleton_path) ? _pipeline_load(tree_skeleton_path, "tree skeleton output") : pc_tree[1:0]
        tree_res = (
            pc_output=pc_tree,
            skeleton_cloud=pc_skeleton,
            filtered_cloud=pc_tree[1:0],
            n_components=0,
            neighbor_radius=0.0,
        )
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

    return (
        input_path=input_path,
        output_dir=output_dir,
        output_prefix=output_prefix,
        n_input=npoints(pc_input),
        n_preprocess=isnothing(pc_preprocess) ? 0 : npoints(pc_preprocess),
        n_ground=isnothing(ground_points) ? 0 : npoints(ground_points),
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
