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
    
    @testset "LAS/LAZ Metadata" begin
        coords = rand(Float64, 200, 3) .* 100.0
        pc = make_test_pointcloud(coords)

        output_path = joinpath(test_dir, "test_meta.las")
        write_las(output_path, pc)

        meta = read_las_metadata(output_path)
        @test meta isa PointCloudMetadata
        @test meta.format == "LAS"
        @test meta.point_count == 200
        @test length(meta.bounds_min) == 3
        @test length(meta.bounds_max) == 3
        @test all(meta.bounds_min .<= meta.bounds_max)
        @test meta.version != ""
        @test meta.point_format >= 0
        @test length(meta.scales) == 3
        @test length(meta.offsets) == 3
        @test isempty(meta.translation)
        @test meta.rotation ≈ Matrix{Float64}(I, 3, 3)
        @test meta.scan_count == 1

        # Dispatcher
        meta2 = read_pc_metadata(output_path)
        @test meta2 isa PointCloudMetadata
        @test meta2.point_count == 200

        # File not found
        @test_throws ErrorException read_las_metadata("__nonexistent__.las")
    end

    @testset "E57 I/O" begin

        @testset "E57 round-trip (coordinates only)" begin
            n = 500
            coords = randn(n, 3) .* 10.0
            pc = make_test_pointcloud(coords)

            tmp = mktempdir()
            e57_path = joinpath(tmp, "test_coords.e57")

            write_e57(e57_path, pc)
            @test isfile(e57_path)
            @test filesize(e57_path) > 0

            pc2 = read_e57(e57_path)
            @test npoints(pc2) == n
            @test coordinates(pc2) ≈ coordinates(pc) atol=1e-4

            rm(tmp; recursive=true, force=true)
        end

        @testset "E57 round-trip (with attributes)" begin
            n = 200
            coords = randn(n, 3) .* 5.0
            attrs = Dict{Symbol,Any}(
                :color_red   => UInt16.(rand(0:255, n)),
                :color_green => UInt16.(rand(0:255, n)),
                :color_blue  => UInt16.(rand(0:255, n)),
            )
            pc = make_test_pointcloud(coords; attrs=attrs)

            tmp = mktempdir()
            e57_path = joinpath(tmp, "test_attrs.e57")

            write_e57(e57_path, pc)
            pc2 = read_e57(e57_path)

            @test npoints(pc2) == n
            @test coordinates(pc2) ≈ coordinates(pc) atol=1e-4

            # RGB round-trip should be exact
            if hasattribute(pc2, :color_red)
                @test getattribute(pc2, :color_red)   == getattribute(pc, :color_red)
                @test getattribute(pc2, :color_green) == getattribute(pc, :color_green)
                @test getattribute(pc2, :color_blue)  == getattribute(pc, :color_blue)
            end

            rm(tmp; recursive=true, force=true)
        end

        @testset "E57 scan_index parameter" begin
            n = 100
            coords = randn(n, 3)
            pc = make_test_pointcloud(coords)

            tmp = mktempdir()
            e57_path = joinpath(tmp, "test_scan.e57")
            write_e57(e57_path, pc)

            # Read specific scan (index 0)
            pc2 = read_e57(e57_path; scan_index=0)
            @test npoints(pc2) == n

            # Out-of-range scan index should error
            @test_throws ErrorException read_e57(e57_path; scan_index=99)

            rm(tmp; recursive=true, force=true)
        end

        @testset "E57 error handling" begin
            @test_throws ErrorException read_e57("__nonexistent_file__.e57")
        end

        @testset "E57 Metadata" begin
            n = 100
            coords = randn(n, 3) .* 5.0
            pc = make_test_pointcloud(coords)

            tmp = mktempdir()
            e57_path = joinpath(tmp, "test_meta.e57")
            write_e57(e57_path, pc)

            # Single scan metadata
            meta = read_e57_metadata(e57_path; scan_index=0)
            @test meta isa PointCloudMetadata
            @test meta.format == "E57"
            @test meta.point_count == n
            @test meta.scan_count == 1
            @test meta.scan_index == 0
            @test meta.point_format == -1

            # All scans (returns vector)
            metas = read_e57_metadata(e57_path)
            @test metas isa Vector{PointCloudMetadata}
            @test length(metas) == 1
            @test metas[1].point_count == n

            # Dispatcher
            metas2 = read_pc_metadata(e57_path)
            @test metas2 isa Vector{PointCloudMetadata}

            # Out-of-range scan index
            @test_throws ErrorException read_e57_metadata(e57_path; scan_index=99)

            # File not found
            @test_throws ErrorException read_e57_metadata("__nonexistent__.e57")

            rm(tmp; recursive=true, force=true)
        end

        @testset "Internal: _merge_scan_attrs" begin
            a1 = Dict{Symbol,Vector}(:color_red => UInt16[1, 2], :color_green => UInt16[10, 20])
            a2 = Dict{Symbol,Vector}(:color_red => UInt16[3, 4], :color_green => UInt16[30, 40])
            merged = FLiP._merge_scan_attrs([a1, a2])
            @test merged[:color_red] == UInt16[1, 2, 3, 4]
            @test merged[:color_green] == UInt16[10, 20, 30, 40]
        end
    end

    # Cleanup
    rm(test_dir, recursive=true)
end
