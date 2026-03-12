using Test
using FLiP
using LinearAlgebra
import PointClouds

function make_test_pointcloud(coords::AbstractMatrix{<:Real}; attrs=Dict{Symbol,Any}())
    size(coords, 2) == 3 || throw(ArgumentError("coords must be N×3"))
    n = size(coords, 1)

    points_nt = (
        x = Float64.(coords[:, 1]),
        y = Float64.(coords[:, 2]),
        z = Float64.(coords[:, 3]),
    )
    pc = PointClouds.LAS(
        PointClouds.IO.PointRecord0,
        points_nt;
        coord_scale=(1e-6, 1e-6, 1e-6),
        coord_offset=(0.0, 0.0, 0.0),
    )

    for (name, vals_any) in attrs
        vals = collect(vals_any)
        length(vals) == n || throw(ArgumentError("attribute :$name has wrong length"))
        pc = addattribute(pc, name, vals)
    end

    return pc
end

@testset "FLiP.jl" begin
    include("test_types.jl")
    include("test_io.jl")
    include("test_subsampling.jl")
    include("test_filtering.jl")
    include("test_mesh.jl")
    include("test_transformations.jl")
end
