"""
Package-wide configuration for FLiP.jl.

Default parameter values are loaded from `flip_config.toml` in the package root at
module initialization. Call `load_config!` to reload from a different file at runtime.
"""

using TOML

"""
    FLiPConfig

Mutable struct holding package-wide default parameters for all filtering functions.
Modify fields directly or reload from a TOML file with `load_config!`.
"""
mutable struct FLiPConfig
    # statistical_filter / statistical_filter_indices
    statistical_filter_k_neighbors::Int
    statistical_filter_n_sigma::Float64

    # voxel_connected_component_filter_indices
    voxel_cc_filter_min_cc_size::Int

    # upward_conic_filter_indices
    upward_conic_filter_max_search_delta_z::Float64

    # rnn_filter / rnn_filter_indices
    rnn_filter_min_rnn_size::Int

    # segment_ground
    segment_ground_voxel_size::Float64
    segment_ground_min_cc_size::Int
    segment_ground_grid_size::Float64
    segment_ground_cone_theta_deg::Float64

    # preprocess
    preprocess_enable_statistical_filter::Bool

    # tree segmentation
    tree_nearground_agh_threshold::Float64
    tree_neighbor_radius::Float64
    tree_slice_length::Float64
    tree_min_cc_size::Int
    tree_max_lcs_iterations::Int
    tree_nbs_neighbor_distance::Int
    tree_nbs_min_segment_size::Int
    tree_nbs_max_iterations::Int
    tree_linearity_angle_deg::Float64

    # pipeline runner
    pipeline_input_path::String
    pipeline_output_dir::String
    pipeline_output_prefix::String
    pipeline_subsample_res::Float64
    pipeline_enable_subsample::Bool
    pipeline_enable_preprocess::Bool
    pipeline_enable_agh::Bool
    pipeline_xy_resolution::Float64
    pipeline_idw_k::Int
    pipeline_idw_power::Float64
    pipeline_enable_ground_segmentation::Bool
    pipeline_enable_tree_segmentation::Bool
    pipeline_enable_qsm::Bool
    pipeline_enable_generate_report::Bool
end

function FLiPConfig(d::Dict)
    sf = get(d, "statistical_filter",               Dict{String,Any}())
    vc = get(d, "voxel_connected_component_filter", Dict{String,Any}())
    uc = get(d, "upward_conic_filter",              Dict{String,Any}())
    rn = get(d, "rnn_filter",                       Dict{String,Any}())
    sg = get(d, "segment_ground",                   Dict{String,Any}())
    pp = get(d, "preprocess",                       Dict{String,Any}())
    ts = get(d, "tree_segmentation",                Dict{String,Any}())
    pl = get(d, "pipeline",                         Dict{String,Any}())

    FLiPConfig(
        Int(get(sf, "k_neighbors",         6)),
        Float64(get(sf, "n_sigma",         1.0)),
        Int(get(vc, "min_cc_size",         1)),
        Float64(get(uc, "max_search_delta_z", 5.0)),
        Int(get(rn, "min_rnn_size",        1)),
        Float64(get(sg, "voxel_size",      0.5)),
        Int(get(sg, "min_cc_size",         50)),
        Float64(get(sg, "grid_size",       0.5)),
        Float64(get(sg, "cone_theta_deg",  60.0)),

        Bool(get(pp, "enable_statistical_filter", false)),

        Float64(get(ts, "nearground_agh_threshold", 0.3)),
        Float64(get(ts, "neighbor_radius", -1.0)),
        Float64(get(ts, "slice_length", 0.1)),
        Int(get(ts, "min_cc_size", 3)),
        Int(get(ts, "max_lcs_iterations", 5000)),
        Int(get(ts, "nbs_neighbor_distance", 2)),
        Int(get(ts, "nbs_min_segment_size", 5)),
        Int(get(ts, "nbs_max_iterations", 10000)),
        Float64(get(ts, "linearity_angle_deg", 80.0)),

        String(get(pl, "input_path", "")),
        String(get(pl, "output_dir", "")),
        String(get(pl, "output_prefix", "output")),
        Float64(get(pl, "subsample_res", 0.05)),
        Bool(get(pl, "enable_subsample", false)),
        Bool(get(pl, "enable_preprocess", true)),
        Bool(get(pl, "enable_agh", true)),
        Float64(get(pl, "xy_resolution", 0.05)),
        Int(get(pl, "idw_k", 8)),
        Float64(get(pl, "idw_power", 2.0)),
        Bool(get(pl, "enable_ground_segmentation", true)),
        Bool(get(pl, "enable_tree_segmentation", true)),
        Bool(get(pl, "enable_qsm", true)),
        Bool(get(pl, "enable_generate_report", true)),
    )
end

const _DEFAULT_CONFIG_PATH = joinpath(@__DIR__, "..", "flip_config.toml")

"""
    load_config!(path::String) -> FLiPConfig

Load parameter defaults from a TOML configuration file and update the
package-wide `FLiP._CFG` instance in-place. Missing keys fall back to
hardcoded defaults.

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
    for f in fieldnames(FLiPConfig)
        setfield!(_CFG, f, getfield(new_cfg, f))
    end
    return _CFG
end

# Module-level singleton — initialized on first load from file (or fallbacks).
const _CFG = FLiPConfig(
    isfile(_DEFAULT_CONFIG_PATH) ? TOML.parsefile(_DEFAULT_CONFIG_PATH) : Dict{String,Any}()
)
