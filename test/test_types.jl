@testset "Types" begin
    @testset "PointCloud creation helper" begin
        coords = rand(Float64, 100, 3)
        pc = make_test_pointcloud(coords)
        @test pc isa FLiP.PointCloud
        @test npoints(pc) == 100
        @test size(FLiP.coordinates(pc)) == (100, 3)

        pc2 = make_test_pointcloud(coords; attrs=Dict(:test_attr => rand(Float64, 100), :label => rand(Int32(1):Int32(5), 100)))
        @test hasattribute(pc2, :test_attr)
        @test hasattribute(pc2, :label)
        @test length(getattribute(pc2, :test_attr)) == 100
    end

    @testset "PointCloud indexing" begin
        coords = rand(Float64, 100, 3)
        test_attr = rand(Float64, 100)
        pc = make_test_pointcloud(coords; attrs=Dict(:test_attr => test_attr))

        pc_sub = pc[1:10]
        @test npoints(pc_sub) == 10
        @test hasattribute(pc_sub, :test_attr)
        @test length(getattribute(pc_sub, :test_attr)) == 10

        indices = [5, 10, 15]
        pc_subset = pc[indices]
        @test npoints(pc_subset) == 3
        @test FLiP.coordinates(pc_subset)[1, :] ≈ coords[5, :] atol=1e-6
    end

    @testset "PointCloud attributes" begin
        coords = rand(Float64, 100, 3)
        pc = make_test_pointcloud(coords)
        
        # Add attribute
        test_attr = rand(100)
        pc2 = setattribute!(pc, :test_attr, test_attr)
        @test !hasattribute(pc, :test_attr)
        @test hasattribute(pc2, :test_attr)
        @test getattribute(pc2, :test_attr) == test_attr

        # Replace attribute
        test_attr2 = rand(100)
        pc3 = setattribute!(pc2, :test_attr, test_attr2)
        @test getattribute(pc3, :test_attr) == test_attr2

        # Delete custom attribute
        pc4 = deleteattribute(pc3, :test_attr)
        @test !hasattribute(pc4, :test_attr)

        # Invalid attribute length
        @test_throws ArgumentError setattribute!(pc, :wrong, rand(50))
    end

    @testset "PointCloud utility functions" begin
        coords = [
            0.0 0.0 0.0;
            1.0 0.0 0.0;
            0.0 1.0 0.0;
            0.0 0.0 1.0
        ]
        pc = make_test_pointcloud(coords)

        @test npoints(pc) == 4
        @test FLiP.coordinates(pc) == coords

        bbox = bounds(pc)
        @test bbox == (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

        c = center(pc)
        @test c ≈ [0.25, 0.25, 0.25]
    end

    @testset "PointCloud display" begin
        coords = rand(Float64, 100, 3)
        pc = make_test_pointcloud(coords; attrs=Dict(:test_attr => rand(Float64, 100)))

        io = IOBuffer()
        show(io, pc)
        @test length(String(take!(io))) > 0

        show(io, MIME("text/plain"), pc)
        @test length(String(take!(io))) > 0
    end
end
