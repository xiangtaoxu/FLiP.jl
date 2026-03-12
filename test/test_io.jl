@testset "I/O" begin
    # Create temporary test directory
    test_dir = mktempdir()
    
    @testset "LAS/LAZ I/O" begin
        @testset "Write and read LAS" begin
            coords = rand(Float64, 200, 3)
            test_attr = rand(Float64, 200)
            pc = make_test_pointcloud(coords; attrs=Dict(:test_attr => test_attr))

            output_path = joinpath(test_dir, "test_output.las")
            write_las(output_path, pc)
            @test isfile(output_path)

            pc2 = read_las(output_path)
            @test pc2 isa PointCloud
            @test length(pc2) == length(pc)
            @test size(coordinates(pc2)) == size(coordinates(pc))
            @test hasattribute(pc2, :test_attr)
            @test length(getattribute(pc2, :test_attr)) == length(pc2)
        end

        @testset "Write AGH extra attribute" begin
            coords = rand(Float64, 150, 3)
            pc = make_test_pointcloud(coords)
            agh = rand(Float64, length(pc))
            pc_agh = setattribute!(pc, :AGH, agh)

            output_path = joinpath(test_dir, "test_agh_output.las")
            write_las(output_path, pc_agh)
            @test isfile(output_path)

            pc2 = read_las(output_path)
            @test hasattribute(pc2, :AGH)
            @test length(getattribute(pc2, :AGH)) == length(pc2)
        end

        @testset "Subsample and save" begin
            coords = rand(Float64, 500, 3)
            pc = make_test_pointcloud(coords)
            pc_sub = distance_subsample(pc, 0.1)
            @test length(pc_sub) <= length(pc)

            output_path = joinpath(test_dir, "test_subsampled.las")
            write_las(output_path, pc_sub)
            @test isfile(output_path)
        end
    end
    
    # Cleanup
    rm(test_dir, recursive=true)
end
