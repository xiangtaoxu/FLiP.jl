@inline function _pipeline_output_path(output_dir::AbstractString, output_prefix::AbstractString, stem::AbstractString; fmt::AbstractString="las")
    return joinpath(output_dir, "$(output_prefix)$(stem).$(fmt)")
end

@inline function _pipeline_write(path::AbstractString, pc::PointCloud)
    write_pc(path, pc)
    println("[main] wrote: $path")
    return true
end

"""Load a single pipeline output file."""
@inline function _pipeline_load(path::AbstractString, label::AbstractString)
    isfile(path) || throw(ArgumentError("$label not found: $path"))
    println("[main] resume: loading $(path)")
    return read_pc(path)
end

"""
Load pipeline output that may be a single file or multiple `_S{i}` files.

Tries `{prefix}{stem}.{fmt}` first; if not found, discovers
`{prefix}{stem}_S{i}.{fmt}` files, loads each, and merges into one cloud.
Returns `nothing` if no matching files exist.
"""
function _pipeline_load(output_dir::AbstractString, output_prefix::AbstractString, stem::AbstractString, fmt::AbstractString)
    # Try single-file first
    single_path = joinpath(output_dir, "$(output_prefix)$(stem).$(fmt)")
    if isfile(single_path)
        println("[main] resume: loading $single_path")
        return read_pc(single_path)
    end

    # Try multi-file _S{i} pattern
    file_prefix = "$(output_prefix)$(stem)_S"
    ext = "." * fmt
    re_idx = Regex("_S(\\d+)\\." * replace(fmt, r"([.+*?^${}()|[\]\\])" => s"\\\1") * "\$", "i")
    candidates = filter(readdir(output_dir; join=true)) do f
        bn = basename(f)
        startswith(bn, file_prefix) && endswith(lowercase(bn), ext)
    end
    isempty(candidates) && return nothing

    # Sort by scan index
    sort!(candidates, by=f -> parse(Int, match(re_idx, basename(f)).captures[1]))

    println("[main] resume: loading $(length(candidates)) $(stem) files from $output_dir")
    all_coords = Vector{Matrix{<:AbstractFloat}}(undef, length(candidates))
    all_attrs  = Vector{Dict{Symbol,Vector}}(undef, length(candidates))
    for (i, fpath) in enumerate(candidates)
        println("[main] resume: loading $fpath")
        pc = read_pc(fpath)
        all_coords[i] = coordinates(pc)
        all_attrs[i]  = _all_attributes(pc)
    end

    if length(candidates) == 1
        return _build_pointcloud_from_coords(all_coords[1], all_attrs[1])
    end

    # Merge
    common_keys = Set(keys(all_attrs[1]))
    for a in all_attrs[2:end]
        intersect!(common_keys, keys(a))
    end
    merged_attrs = Dict{Symbol,Vector}()
    for k in common_keys
        merged_attrs[k] = vcat([a[k] for a in all_attrs]...)
    end
    merged_coords = vcat(all_coords...)
    println("[main] resume: merged $(length(candidates)) scans → $(size(merged_coords, 1)) points")
    return _build_pointcloud_from_coords(merged_coords, merged_attrs)
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
    (isfile(input_path) || isdir(input_path)) || throw(ArgumentError("pipeline input path not found: $input_path"))

    cfg.pipeline_subsample_res > 0 || throw(ArgumentError("pipeline.subsample_res must be > 0"))
    cfg.pipeline_xy_resolution > 0 || throw(ArgumentError("pipeline.xy_resolution must be > 0"))
    cfg.pipeline_idw_k >= 1 || throw(ArgumentError("pipeline.idw_k must be >= 1"))
    cfg.pipeline_idw_power > 0 || throw(ArgumentError("pipeline.idw_power must be > 0"))

    output_fmt = lowercase(cfg.pipeline_output_format)
    mkpath(output_dir)

    # Update expanded paths back into config for preprocess
    cfg.pipeline_input_path = input_path
    cfg.pipeline_output_dir = output_dir

    # 1) preprocess (reads input files, preprocesses each, writes individual outputs)
    preprocess_path = _pipeline_output_path(output_dir, output_prefix, "preprocess"; fmt=output_fmt)
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
    ground_path = _pipeline_output_path(output_dir, output_prefix, "ground"; fmt=output_fmt)
    agh_path = _pipeline_output_path(output_dir, output_prefix, "agh"; fmt=output_fmt)
    ground_written = false
    agh_written = false

    n_preprocess = 0
    n_ground = 0

    if cfg.pipeline_enable_ground_segmentation
        ground_input = _pipeline_require(pc_preprocess, preprocess_path, "preprocess output", "ground segmentation")
        pc_preprocess = nothing  # release input — no longer needed
        ground_res = ground_segmentation(ground_input; cfg=cfg)
        ground_input = nothing
        ground_points = ground_res.ground_points
        pc_agh = ground_res.agh_cloud
        n_preprocess = npoints(ground_res.agh_cloud)
        n_ground = npoints(ground_points)
        ground_written = _pipeline_write(ground_path, ground_points)
        ground_points = nothing  # written to disk, release
        if cfg.pipeline_enable_agh
            agh_written = _pipeline_write(agh_path, pc_agh)
        end
    else
        println("[main] ground segmentation disabled by config")
        pc_preprocess = nothing  # no downstream consumer, release
    end

    # 3) tree segmentation
    tree_path = _pipeline_output_path(output_dir, output_prefix, "tree"; fmt=output_fmt)
    tree_skeleton_path = _pipeline_output_path(output_dir, output_prefix, "skeleton"; fmt=output_fmt)
    tree_written = false
    tree_skeleton_written = false
    tree_components = 0

    tree_res = nothing
    if cfg.pipeline_enable_tree_segmentation
        tree_input = _pipeline_require(pc_agh, agh_path, "AGH output", "tree segmentation")
        pc_agh = nothing  # release — tree_input holds the reference
        GC.gc()
        tree_res = tree_segmentation(tree_input; cfg=cfg)
        tree_input = nothing  # release input
        tree_components = tree_res.n_components
        tree_written = _pipeline_write(tree_path, tree_res.pc_output)
        tree_skeleton_written = _pipeline_write(tree_skeleton_path, tree_res.skeleton_cloud)
    else
        println("[main] tree segmentation disabled by config")
        pc_agh = nothing  # no downstream consumer, release
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
        qsm(tree_result=tree_res, config_path=String(config_path), output_dir=output_dir, output_prefix=output_prefix, tree_cloud_path=tree_path)
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
