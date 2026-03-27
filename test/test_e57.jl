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

    @testset "Internal: _merge_scan_attrs" begin
        a1 = Dict{Symbol,Vector}(:color_red => UInt16[1, 2], :color_green => UInt16[10, 20])
        a2 = Dict{Symbol,Vector}(:color_red => UInt16[3, 4], :color_green => UInt16[30, 40])
        merged = FLiP._merge_scan_attrs([a1, a2])
        @test merged[:color_red] == UInt16[1, 2, 3, 4]
        @test merged[:color_green] == UInt16[10, 20, 30, 40]
    end
end
