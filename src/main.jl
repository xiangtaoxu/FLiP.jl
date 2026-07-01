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
    # NBS refinement (lean trial QSM + refine_nbs) now runs INSIDE tree_segmentation,
    # so the QSM stage is a single final pass over the refined tree cloud.
    t_output  = _stage_tree(cfg, g_output.agh);         g_output  = _drop_ground_clouds(g_output)
    q_output  = _stage_qsm(cfg, t_output.result, config_path)
    r_output  = _stage_report(cfg, t_output.result, q_output, config_path)
    t_output  = _drop_tree_clouds(t_output); GC.gc()

    return _summarize(cfg, pp_output, g_output, t_output, q_output, r_output)
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
    # Ground mesh is a triangulated surface, always written as PLY (CloudCompare-readable)
    ground_mesh_path = get_output_path(cfg.pipeline.output_dir, cfg.pipeline.output_prefix, "ground_mesh", "ply")

    if !cfg.pipeline.enable_ground_segmentation
        _log_stage_skipped("ground_segmentation")
        return (ground=nothing, agh=nothing,
                ground_path=ground_path, agh_path=agh_path,
                ground_mesh_path=ground_mesh_path,
                ground_written=false, agh_written=false, ground_mesh_written=false,
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

        # Triangulated ground surface from the dense interpolated lattice (PLY for CloudCompare)
        ground_mesh_written = false
        if cfg.pipeline.enable_ground_mesh
            if npoints(res.ground_points) >= 3
                mesh = build_ground_mesh(res.ground_points;
                                         xy_resolution=cfg.pipeline.xy_resolution,
                                         idw_k=cfg.pipeline.idw_k,
                                         idw_power=cfg.pipeline.idw_power)
                if isempty(mesh.faces)
                    @warn "$_LOG_PREFIX   ground mesh: lattice too small to triangulate; skipping"
                else
                    write_ply_mesh(ground_mesh_path, mesh.vertices, mesh.faces)
                    @info "$_LOG_PREFIX   wrote: $ground_mesh_path"
                    ground_mesh_written = true
                end
            else
                @warn "$_LOG_PREFIX   ground mesh: fewer than 3 ground points; skipping"
            end
        end

        (ground=res.ground_points, agh=res.agh_cloud,
         ground_path=ground_path, agh_path=agh_path, ground_mesh_path=ground_mesh_path,
         ground_written=true, agh_written=agh_written, ground_mesh_written=ground_mesh_written,
         n_preprocess=npoints(res.agh_cloud), n_ground=npoints(res.ground_points))
    end
end

"""
    _stage_tree(cfg, pc_agh) -> NamedTuple

Run tree segmentation (which now internally runs the lean trial QSM + `refine_nbs`
before assembly), write the tree output, and return the full `tree_segmentation`
result plus path / written-flag / n_components metadata. If `pc_agh` is `nothing`,
the input is loaded from disk via `_prepare_stage_input`.

Returns `(result, tree_path, tree_written, n_components)`. `result` is `nothing`
when the stage is disabled.
"""
function _stage_tree(cfg::FLiPConfig, pc_agh)
    fmt = lowercase(cfg.pipeline.output_format)
    tree_path = get_output_path(cfg.pipeline.output_dir, cfg.pipeline.output_prefix, "tree", fmt)

    if !cfg.pipeline.enable_tree_segmentation
        _log_stage_skipped("tree_segmentation")
        return (result=nothing, tree_path=tree_path, tree_written=false, n_components=0)
    end

    return _with_stage_timing("tree_segmentation") do
        tree_input = _prepare_stage_input(pc_agh, cfg, "agh", "AGH output", "tree segmentation")
        GC.gc()
        res = tree_segmentation(tree_input; cfg=cfg)
        tree_input = nothing  # release input

        write_pc(tree_path, res.pc_output); @info "$_LOG_PREFIX   wrote: $tree_path"

        (result=res, tree_path=tree_path, tree_written=true, n_components=res.n_components)
    end
end

"""
    _stage_qsm(cfg, tree_res, config_path) -> NamedTuple

Run the (single, final) modeling + reporting QSM stage (or `(status=:skipped,)` if
disabled). NBS refinement already happened inside the tree stage, so this fits the final
per-node cylinder model (`model_nbs`), stamps `:node_id` onto the tree cloud, and writes the
biometric CSVs (`report/`) + the surface cloud. If `tree_res` is `nothing`, the tree result is
reconstructed from disk via `_load_tree_result(cfg)`.
"""
function _stage_qsm(cfg::FLiPConfig, tree_res, config_path::AbstractString)
    if !cfg.pipeline.enable_qsm
        _log_stage_skipped("qsm")
        return (status=:skipped,)
    end
    return _with_stage_timing("qsm") do
        tr = isnothing(tree_res) ? _load_tree_result(cfg) : tree_res
        pc = tr.pc_output
        fmt = lowercase(cfg.pipeline.output_format)
        dir = cfg.pipeline.output_dir; prefix = cfg.pipeline.output_prefix
        tree_path = get_output_path(dir, prefix, "tree", fmt)

        m = model_nbs(pc=pc, cfg=cfg, group_attr=:tree_nbs_id, node_id_attr=:node_id, emit_surface=true)
        surf = assemble_surface_cloud(m.surface_parts)
        surf_path = joinpath(dir, "$(prefix)qsm_surface.laz")
        # Only the :success path writes outputs (matches the original early-return behavior on
        # :no_data / :no_linear_nbs, which left the tree-stage file untouched).
        bm = (n_trees=0, node_csv_path=joinpath(dir, "$(prefix)qsm_nodes.csv"),
              tree_csv_path=joinpath(dir, "$(prefix)qsm_trees.csv"))
        if m.status == :success
            bm = write_biometrics(m.nodes, cfg; output_dir=dir, output_prefix=prefix)
            if !isempty(dir)
                write_pc(tree_path, pc)                              # tree cloud now carrying :node_id
                npoints(surf) > 0 && write_pc(surf_path, surf)
            end
        end

        (status=m.status, n_nodes=length(m.nodes), n_trees=bm.n_trees,
         node_csv_path=bm.node_csv_path, tree_csv_path=bm.tree_csv_path,
         pc_output=pc, qsm_surface_cloud=surf, surface_cloud_path=surf_path, nodes=m.nodes)
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
4 and 5 when stage 3 was disabled or skipped). Loads the tree output and fills
`filtered_cloud`, `n_components`, `neighbor_radius` with empty / zero defaults.
Throws if the tree output is not on disk.
"""
function _load_tree_result(cfg::FLiPConfig; stem::AbstractString="tree")
    pc_tree = _prepare_stage_input(nothing, cfg, stem,
                                   "tree output", "qsm/report stage")
    return (
        pc_output=pc_tree,
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

"""
    _summarize(cfg, pp_output, g_output, t_output, q_output, r_output) -> NamedTuple

Build the stage-grouped summary returned by `run_pipeline`. Each stage's paths,
written-flags, and counts live in their own sub-NamedTuple (`preprocess`,
`ground`, `agh`, `tree`). NBS refinement runs inside the tree stage, so `qsm` is
the single final QSM result and `report` is forwarded as-is.
"""
function _summarize(cfg::FLiPConfig, pp_output, g_output, t_output, q_output, r_output)
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
        ground_mesh = (path=g_output.ground_mesh_path,
                       written=g_output.ground_mesh_written),
        tree       = (path=t_output.tree_path,
                      written=t_output.tree_written,
                      n_components=t_output.n_components),
        qsm        = q_output,
        report     = r_output,
    )
end
