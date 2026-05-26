"""
    preprocess(; cfg::FLiPConfig=_CFG) -> Union{PointCloud, Nothing}

Discover, clean, and merge the input point clouds.

Workflow:
1. Discover input files via [`find_input_files`](@ref) — either a single
   path or every file in a directory matching `{input_prefix}*.{input_format}`.
2. For each file: read it, optionally distance-subsample, optionally
   statistical-filter, and write a per-scan output. Single-file runs
   produce `{prefix}preprocess.{fmt}`; multi-file runs produce
   `{prefix}preprocess_S{i}.{fmt}` per scan. For E57 inputs the subsample
   is applied to the raw coordinate matrix *before* building the
   `PointCloud`, to avoid materializing the full-size cloud in memory.
3. **Single-file**: return the in-memory cloud directly.
   **Multi-file**: return `nothing`. The downstream stage reloads and
   merges the per-scan files via `_prepare_stage_input`, which keeps peak
   memory at one scan instead of `n_files × scan + merged`.
"""
function preprocess(; cfg::FLiPConfig=_CFG)
    input_files = find_input_files(; cfg=cfg)
    output_dir = cfg.pipeline_output_dir
    output_prefix = cfg.pipeline_output_prefix
    output_fmt = lowercase(cfg.pipeline_output_format)

    mkpath(output_dir)

    n_files = length(input_files)
    T = coord_type(cfg)
    last_pc::Union{PointCloud{T}, Nothing} = nothing

    for (i, fpath) in enumerate(input_files)
        @info "[preprocess] Reading file $i/$n_files: $fpath"
        ext = lowercase(splitext(fpath)[2])

        if ext == ".e57" && cfg.pipeline_enable_preprocess
            # For large E57 files: subsample/filter on raw coords before building
            # PointCloud — avoids peak memory from full-size cloud construction.
            # (Previously only triggered when subsample was also enabled, which
            # left stat-filter-only E57 inputs on the full-cloud path.)
            coords, attrs = _read_e57_to_raw(fpath; precision=T)
            @info "[preprocess]   raw points: $(size(coords, 1))"
            coords, attrs = _apply_preprocess_filters(coords, attrs; cfg=cfg)
            pc = PointCloud(coords, attrs)
        else
            pc = read_pc(fpath)
            if cfg.pipeline_enable_preprocess
                coords, attrs = _apply_preprocess_filters(coordinates(pc),
                                                          _all_attributes(pc); cfg=cfg)
                pc = PointCloud(coords, attrs)
            end
        end

        # Write with _S{i} suffix for multi-file, no suffix for single file
        suffix = n_files > 1 ? "_S$(i)" : ""
        out_path = joinpath(output_dir, "$(output_prefix)preprocess$(suffix).$(output_fmt)")
        write_pc(out_path, pc)
        @info "[preprocess] Wrote: $out_path  ($(npoints(pc)) points)"

        # Single-file: hold the cloud in memory and return it directly.
        # Multi-file: rely on the on-disk per-scan files; downstream reloads
        # via `_prepare_stage_input` rather than accumulating in memory.
        if n_files == 1
            last_pc = pc
        end
    end

    if n_files > 1
        @info "[preprocess] Wrote $n_files scan files; downstream stages will lazy-merge from disk"
    end
    return last_pc
end

# ── Per-scan helper ───────────────────────────────────────────────

"""
    _apply_preprocess_filters(coords::AbstractMatrix, attrs::Dict; cfg::FLiPConfig=_CFG)
        -> (Matrix, Dict)

Apply optional distance subsample + statistical filter to raw `(coords, attrs)`.
Pure: returns new arrays, does not mutate the inputs.

Used by both the LAS path (after `read_pc` → unpacked into raw arrays) and the
E57 path (raw arrays straight from `_read_e57_to_raw`), so the two branches
share a single filter implementation.
"""
function _apply_preprocess_filters(coords::AbstractMatrix, attrs::Dict; cfg::FLiPConfig=_CFG)
    if cfg.pipeline_enable_subsample
        keep = distance_subsample(coords, cfg.pipeline_subsample_res)
        coords = coords[keep, :]
        attrs = Dict(k => v[keep] for (k, v) in attrs)
    end
    if cfg.preprocess_enable_statistical_filter
        keep = statistical_filter(coords,
                                  cfg.statistical_filter_k_neighbors,
                                  cfg.statistical_filter_n_sigma)
        coords = coords[keep, :]
        attrs = Dict(k => v[keep] for (k, v) in attrs)
    end
    return coords, attrs
end
