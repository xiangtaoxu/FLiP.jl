"""
Stage 3 microbenchmarks: validates the hot-path allocation fixes
(SVector queries, @view filter chain, fused bounds, lazy multi-scan merge).

Usage:
    julia --project=. scripts/bench_stage3.jl [path/to/cloud.las]

If no path is given, falls back to a synthetic 1M-point cloud so the script is
runnable without test data.
"""

using FLiP
using BenchmarkTools
using Random

function load_or_synthesize(path::Union{String,Nothing})
    if path !== nothing && isfile(path)
        @info "Loading $path"
        return read_pc(path)
    end
    @info "Synthesizing 1M-point cloud (no LAS file given)"
    Random.seed!(42)
    n = 1_000_000
    coords = Matrix{Float64}(undef, n, 3)
    @inbounds for i in 1:n
        coords[i, 1] = 100.0 * rand()
        coords[i, 2] = 100.0 * rand()
        coords[i, 3] = 30.0 * rand()
    end
    return FLiP.PointCloud(coords, Dict{Symbol,Vector}())
end

function main()
    path = length(ARGS) >= 1 ? ARGS[1] : nothing
    pc = load_or_synthesize(path)
    coords = coordinates(pc)
    println("Cloud: $(npoints(pc)) points")

    println("\n--- bounds(pc) ---")
    display(@benchmark bounds($pc))

    println("\n--- statistical_filter (k=10, nσ=2.0) ---")
    display(@benchmark statistical_filter($coords, 10, 2.0))

    println("\n--- segment_ground (full chain with @view) ---")
    display(@benchmark segment_ground($pc;
        grid_size=0.5, cone_theta_deg=20.0, voxel_size=0.5, min_cc_size=5))

    println()
end

main()
