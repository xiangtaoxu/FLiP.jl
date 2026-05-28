"""
Package-wide configuration for FLiP.jl.

`FLiPConfig` is a wrapper struct whose fields mirror the section structure of
`flip_config.toml`. Access is hierarchical (e.g. `cfg.qsm.min_node_size`,
`cfg.pipeline.subsample_res`) and matches the TOML 1:1 — every field lives in
the section that owns its TOML key.

Default values are loaded from `flip_config.toml` in the package root at module
initialization. Call `load_config!` to reload from a different file at runtime.
"""

using TOML

# ── Section structs ──────────────────────────────────────────────────────────
# One mutable struct per TOML `[section]`. Field names match TOML keys exactly.

mutable struct StatisticalFilterCfg
    k_neighbors::Int
    n_sigma::Float64
end
StatisticalFilterCfg(d::Dict) = StatisticalFilterCfg(
    Int(get(d, "k_neighbors", 6)),
    Float64(get(d, "n_sigma",  1.0)),
)

mutable struct SegmentGroundCfg
    voxel_size::Float64
    min_cc_size::Int
    grid_size::Float64
    cone_theta_deg::Float64
    polygon_buffer::Float64
    enable_ground_crop::Bool
end
SegmentGroundCfg(d::Dict) = SegmentGroundCfg(
    Float64(get(d, "voxel_size",         0.5)),
    Int(    get(d, "min_cc_size",        50)),
    Float64(get(d, "grid_size",          0.5)),
    Float64(get(d, "cone_theta_deg",     60.0)),
    Float64(get(d, "polygon_buffer",     5.0)),
    Bool(   get(d, "enable_ground_crop", true)),
)

mutable struct PreprocessCfg
    enable_statistical_filter::Bool
end
PreprocessCfg(d::Dict) = PreprocessCfg(
    Bool(get(d, "enable_statistical_filter", false)),
)

mutable struct TreeSegmentationCfg
    nearground_agh_threshold::Float64
    neighbor_radius::Float64
    frontier_min_cc_size::Int
    nbs_neighbor_distance::Int
    min_nbs_size::Int
    linearity_angle_deg::Float64
    assembly_merge_threshold::Float64
    assembly_occlusion_tolerance::Float64
end
TreeSegmentationCfg(d::Dict) = TreeSegmentationCfg(
    Float64(get(d, "nearground_agh_threshold",     0.3)),
    Float64(get(d, "neighbor_radius",              -1.0)),
    Int(    get(d, "frontier_min_cc_size",         3)),
    Int(    get(d, "nbs_neighbor_distance",        2)),
    Int(    get(d, "min_nbs_size",                 5)),
    Float64(get(d, "linearity_angle_deg",          80.0)),
    Float64(get(d, "assembly_merge_threshold",     0.5)),
    Float64(get(d, "assembly_occlusion_tolerance", 0.1)),
)

mutable struct QSMCfg
    nbs_linearity_threshold::Float64
    slice_height_scalar::Float64
    min_node_size::Int
    phi_bin_min::Int
    phi_bin_max::Int
    surface_res_scalar::Float64
    completeness_threshold::Float64
    breast_height::Float64
    spl_z_smoothing::Float64
    rho_percentile::Float64
    min_octant_taubin::Int
    qc_enable::Bool
    qc_continuity_ratio::Float64
end
QSMCfg(d::Dict) = QSMCfg(
    Float64(get(d, "nbs_linearity_threshold", 0.5)),
    Float64(get(d, "slice_height_scalar",     3.0)),
    Int(    get(d, "min_node_size",           5)),
    Int(    get(d, "phi_bin_min",             36)),
    Int(    get(d, "phi_bin_max",             360)),
    Float64(get(d, "surface_res_scalar",      0.5)),
    Float64(get(d, "completeness_threshold",  0.25)),
    Float64(get(d, "breast_height",           1.3)),
    Float64(get(d, "spl_z_smoothing",         0.3)),
    Float64(get(d, "rho_percentile",          1.0)),
    Int(    get(d, "min_octant_taubin",       3)),
    Bool(   get(d, "qc_enable",               true)),
    Float64(get(d, "qc_continuity_ratio",     0.7)),
)

mutable struct PipelineCfg
    # I/O
    input_path::String
    input_prefix::String
    input_format::String
    output_dir::String
    output_prefix::String
    output_format::String
    coordinate_precision::DataType

    # Data parameters
    subsample_res::Float64
    xy_resolution::Float64
    idw_k::Int
    idw_power::Float64

    # Stage toggles
    enable_subsample::Bool
    enable_preprocess::Bool
    enable_ground_segmentation::Bool
    enable_agh::Bool
    enable_tree_segmentation::Bool
    enable_qsm::Bool
    enable_generate_report::Bool
    enable_skeleton_output::Bool

    # Logging
    enable_debug_info::Bool

    # Threading: positive = exact thread count; 0/1 = serial;
    # negative = Sys.CPU_THREADS + n_thread (e.g. -1 = all-but-one).
    # Always capped at `Threads.nthreads()` at use time.
    n_thread::Int
end
PipelineCfg(d::Dict) = PipelineCfg(
    String(get(d, "input_path",    "")),
    String(get(d, "input_prefix",  "")),
    String(get(d, "input_format",  "las")),
    String(get(d, "output_dir",    "")),
    String(get(d, "output_prefix", "output")),
    String(get(d, "output_format", "las")),
    let prec_str = lowercase(get(d, "coordinate_precision", "Float32"))
        prec_str == "float64" ? Float64 : Float32
    end,
    Float64(get(d, "subsample_res", 0.05)),
    Float64(get(d, "xy_resolution", 0.05)),
    Int(    get(d, "idw_k",         8)),
    Float64(get(d, "idw_power",     2.0)),
    Bool(get(d, "enable_subsample",           false)),
    Bool(get(d, "enable_preprocess",          true)),
    Bool(get(d, "enable_ground_segmentation", true)),
    Bool(get(d, "enable_agh",                 true)),
    Bool(get(d, "enable_tree_segmentation",   true)),
    Bool(get(d, "enable_qsm",                 true)),
    Bool(get(d, "enable_generate_report",     true)),
    Bool(get(d, "enable_skeleton_output",     false)),
    Bool(get(d, "enable_debug_info",          false)),
    Int( get(d, "n_thread",                   1)),
)

# ── Top-level wrapper ────────────────────────────────────────────────────────

"""
    FLiPConfig

Package-wide configuration. Field layout mirrors `flip_config.toml`:

```
cfg.pipeline.subsample_res
cfg.preprocess.enable_statistical_filter
cfg.statistical_filter.k_neighbors
cfg.segment_ground.voxel_size
cfg.tree_segmentation.min_nbs_size
cfg.qsm.min_node_size
```

Mutate sub-struct fields directly or reload from a TOML file with
`load_config!`.
"""
mutable struct FLiPConfig
    pipeline::PipelineCfg
    preprocess::PreprocessCfg
    statistical_filter::StatisticalFilterCfg
    segment_ground::SegmentGroundCfg
    tree_segmentation::TreeSegmentationCfg
    qsm::QSMCfg
end

FLiPConfig(d::Dict) = FLiPConfig(
    PipelineCfg(         get(d, "pipeline",           Dict{String,Any}())),
    PreprocessCfg(       get(d, "preprocess",         Dict{String,Any}())),
    StatisticalFilterCfg(get(d, "statistical_filter", Dict{String,Any}())),
    SegmentGroundCfg(    get(d, "segment_ground",     Dict{String,Any}())),
    TreeSegmentationCfg( get(d, "tree_segmentation",  Dict{String,Any}())),
    QSMCfg(              get(d, "qsm",                Dict{String,Any}())),
)

# ── Loader + singleton ───────────────────────────────────────────────────────

const _DEFAULT_CONFIG_PATH = joinpath(@__DIR__, "..", "flip_config.toml")

"""
    load_config!(path::String) -> FLiPConfig

Load parameter defaults from a TOML configuration file and update the
package-wide `FLiP._CFG` instance in place by copying each sub-struct's
fields. Missing keys fall back to hardcoded defaults.

# Arguments
- `path`: Path to a TOML file (default: `flip_config.toml` in the package root)

# Example
```julia
FLiP.load_config!("my_project/flip_config.toml")
```
"""
function load_config!(path::String=_DEFAULT_CONFIG_PATH)
    d = isfile(path) ? TOML.parsefile(path) : Dict{String,Any}()
    new_cfg = FLiPConfig(d)
    for section in fieldnames(FLiPConfig)
        old_sub = getfield(_CFG, section)
        new_sub = getfield(new_cfg, section)
        for f in fieldnames(typeof(old_sub))
            setfield!(old_sub, f, getfield(new_sub, f))
        end
    end
    return _CFG
end

# Module-level singleton — initialized on first load from file (or fallbacks).
const _CFG = FLiPConfig(
    isfile(_DEFAULT_CONFIG_PATH) ? TOML.parsefile(_DEFAULT_CONFIG_PATH) : Dict{String,Any}()
)

"""
    coord_type(cfg::FLiPConfig=_CFG) -> DataType

Return the configured coordinate precision type (`Float32` or `Float64`).
"""
coord_type(cfg::FLiPConfig=_CFG) = cfg.pipeline.coordinate_precision

"""
    effective_nthreads(cfg::FLiPConfig=_CFG) -> Int

Resolve the configured `pipeline.n_thread` to a concrete positive thread count.

- positive → that many threads
- 0 or 1   → serial (returns 1)
- negative → `Sys.CPU_THREADS + n_thread` (e.g. `-1` = all-but-one logical core)

The result is always capped at `Threads.nthreads()` (the count Julia was
launched with) and never returns less than 1.
"""
function effective_nthreads(cfg::FLiPConfig=_CFG)
    n = cfg.pipeline.n_thread
    requested = if n > 1
        n
    elseif n < 0
        max(1, Sys.CPU_THREADS + n)
    else
        1
    end
    return min(requested, Threads.nthreads())
end
