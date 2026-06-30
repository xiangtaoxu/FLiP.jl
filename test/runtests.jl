using Test
using FLiP
using LinearAlgebra

function make_test_pointcloud(coords::AbstractMatrix{<:Real}; attrs=Dict{Symbol,Any}())
    size(coords, 2) == 3 || throw(ArgumentError("coords must be N×3"))
    n = size(coords, 1)
    pc_attrs = Dict{Symbol,Vector}()
    for (name, vals_any) in attrs
        vals = collect(vals_any)
        length(vals) == n || throw(ArgumentError("attribute :$name has wrong length"))
        pc_attrs[name] = vals
    end
    T = eltype(coords) <: AbstractFloat ? eltype(coords) : Float64
    return FLiP.PointCloud(T.(coords), pc_attrs)
end

@testset "FLiP.jl" begin
    include("test_types.jl")
    include("test_logging.jl")
    include("test_parallelization.jl")
    include("test_io.jl")
    include("test_subsampling.jl")
    include("test_filtering.jl")
    include("test_geometry.jl")
    include("test_interpolation.jl")
    include("test_mesh.jl")
    include("test_graph.jl")
    include("test_array_utils.jl")
    include("test_tree_segmentation.jl")
    include("test_main.jl")
    include("test_transformations.jl")
    include("test_qsm.jl")
    include("test_refine_nbs.jl")
    include("test_pipeline_integration.jl")
end
