"""
    _preprocess_single(pc::PointCloud; cfg::FLiPConfig=_CFG) -> PointCloud

Prepare a single point cloud before ground segmentation:
1. Optional distance-based subsampling
2. Optional statistical filtering
"""
function _preprocess_single(pc::PointCloud; cfg::FLiPConfig=_CFG)
    active = cfg.pipeline_enable_subsample ? distance_subsample(pc, cfg.pipeline_subsample_res) : pc

    if cfg.preprocess_enable_statistical_filter
        active = statistical_filter(
            active,
            cfg.statistical_filter_k_neighbors,
            cfg.statistical_filter_n_sigma,
        )
    end

    return active
end

"""
    discover_input_files(; cfg::FLiPConfig=_CFG) -> Vector{String}

Find input point cloud files based on config settings.

If `pipeline_input_path` points to a single file, returns `[input_path]`.
If it points to a directory, returns all files matching
`{input_prefix}*.{input_format}` sorted alphabetically.
"""
function discover_input_files(; cfg::FLiPConfig=_CFG)
    input_path = cfg.pipeline_input_path
    prefix = cfg.pipeline_input_prefix
    fmt = lowercase(cfg.pipeline_input_format)

    # Single file mode
    if isfile(input_path)
        return [input_path]
    end

    # Directory mode
    isdir(input_path) || error("input_path is not a file or directory: $input_path")
    ext = "." * fmt
    files = sort(filter(readdir(input_path; join=true)) do f
        bn = basename(f)
        startswith(bn, prefix) && endswith(lowercase(bn), ext)
    end)
    isempty(files) && error("No .$(fmt) files matching prefix '$(prefix)' found in: $input_path")
    return files
end

"""
    _build_pointcloud_from_coords(coords::AbstractMatrix{<:AbstractFloat}, attrs::Dict{Symbol,Vector}) -> PointCloud

Construct a PointCloud from raw coordinates and attribute vectors.
"""
function _build_pointcloud_from_coords(coords::AbstractMatrix{<:AbstractFloat}, attrs::Dict{Symbol,Vector})
    return PointCloudData(coords, attrs)
end

"""
    preprocess(; cfg::FLiPConfig=_CFG) -> PointCloud

Discover input files, preprocess each individually (subsample + filter),
write individual outputs with `_S{i}` suffix, and return a merged point cloud
for downstream pipeline stages.
"""
function preprocess(; cfg::FLiPConfig=_CFG)
    input_files = discover_input_files(; cfg=cfg)
    output_dir = cfg.pipeline_output_dir
    output_prefix = cfg.pipeline_output_prefix
    output_fmt = lowercase(cfg.pipeline_output_format)

    mkpath(output_dir)

    n_files = length(input_files)
    T = coord_type(cfg)
    all_coords = Vector{Matrix{T}}(undef, n_files)
    all_attrs  = Vector{Dict{Symbol,Vector}}(undef, n_files)

    for (i, fpath) in enumerate(input_files)
        println("[preprocess] Reading file $i/$n_files: $fpath")
        ext = lowercase(splitext(fpath)[2])

        if ext == ".e57" && cfg.pipeline_enable_preprocess && cfg.pipeline_enable_subsample
            # For large E57 files: subsample on raw coords before building PointCloud
            # to avoid peak memory from full-size LAS construction
            coords, attrs = _read_e57_to_raw(fpath; precision=T)
            n_raw = size(coords, 1)
            println("[preprocess]   raw points: $n_raw, subsampling at $(cfg.pipeline_subsample_res)m...")
            keep = distance_subsample_indices(coords, cfg.pipeline_subsample_res)
            coords = coords[keep, :]
            for (k, v) in attrs
                attrs[k] = v[keep]
            end
            println("[preprocess]   after subsample: $(size(coords, 1)) points")
            pc = _build_pointcloud_from_coords(coords, attrs)

            if cfg.preprocess_enable_statistical_filter
                pc = statistical_filter(pc, cfg.statistical_filter_k_neighbors, cfg.statistical_filter_n_sigma)
            end
        else
            pc = read_pc(fpath)
            if cfg.pipeline_enable_preprocess
                pc = _preprocess_single(pc; cfg=cfg)
            end
        end

        # Write with _S{i} suffix for multi-file, no suffix for single file
        suffix = n_files > 1 ? "_S$(i)" : ""
        out_path = joinpath(output_dir, "$(output_prefix)preprocess$(suffix).$(output_fmt)")
        write_pc(out_path, pc)
        println("[preprocess] Wrote: $out_path  ($(npoints(pc)) points)")

        # Extract coords/attrs and release the PointCloud object
        all_coords[i] = coordinates(pc)
        all_attrs[i]  = _all_attributes(pc)
    end

    # Single file: rebuild from extracted data (original pc is out of scope)
    if n_files == 1
        return _build_pointcloud_from_coords(all_coords[1], all_attrs[1])
    end

    # Merge: concatenate coordinates and common attributes
    common_keys = Set(keys(all_attrs[1]))
    for a in all_attrs[2:end]
        intersect!(common_keys, keys(a))
    end

    merged_attrs = Dict{Symbol,Vector}()
    for k in common_keys
        merged_attrs[k] = vcat([a[k] for a in all_attrs]...)
    end
    merged_coords = vcat(all_coords...)
    println("[preprocess] Merged $n_files scans → $(size(merged_coords, 1)) points")
    return _build_pointcloud_from_coords(merged_coords, merged_attrs)
end
