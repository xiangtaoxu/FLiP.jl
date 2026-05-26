@testset "interpolate_idw" begin
    @testset "exact recovery at known points" begin
        known_xy = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0; 0.5 0.5]
        known_z  = [0.0, 2.0, 3.0, 5.0, 2.5]
        out = interpolate_idw(known_xy, known_z, known_xy; k=1)
        @test out ≈ known_z
        out2 = interpolate_idw(known_xy, known_z, known_xy; k=4)
        @test out2 ≈ known_z  # exact-match short-circuit still wins
    end

    @testset "symmetric midpoint interpolates to mean" begin
        z = interpolate_idw([0.0 0.0; 2.0 0.0], [0.0, 10.0], [1.0 0.0;]; k=2, power=2.0)
        @test length(z) == 1
        @test z[1] ≈ 5.0
    end

    @testset "k > M falls back to M nearest neighbors" begin
        known_xy = [0.0 0.0; 1.0 0.0]
        known_z  = [1.0, 3.0]
        z = interpolate_idw(known_xy, known_z, [0.5 0.0;]; k=100)
        @test length(z) == 1
        @test z[1] ≈ 2.0  # equidistant → mean of 1 and 3
    end

    @testset "max_distance NaNs queries far from known points" begin
        z = interpolate_idw([0.0 0.0;], [42.0], [10.0 10.0;]; k=1, max_distance=1.0)
        @test isnan(z[1])
    end

    @testset "non-finite query returns NaN" begin
        z = interpolate_idw([0.0 0.0; 1.0 1.0; 2.0 0.0], [1.0, 2.0, 3.0],
                            [NaN 0.0; 0.0 NaN; 0.5 0.5])
        @test isnan(z[1])
        @test isnan(z[2])
        @test isfinite(z[3])
    end

    @testset "empty query returns empty vector" begin
        z = interpolate_idw([0.0 0.0; 1.0 0.0], [0.0, 1.0], Matrix{Float64}(undef, 0, 2))
        @test z isa Vector{Float64}
        @test isempty(z)
    end

    @testset "input validation" begin
        @test_throws ArgumentError interpolate_idw([0.0 0.0 0.0;], [1.0], [0.5 0.0;])  # known not M×2
        @test_throws ArgumentError interpolate_idw([0.0 0.0;], [1.0], [0.5 0.0 0.0;])  # query not N×2
        @test_throws ArgumentError interpolate_idw([0.0 0.0;], [1.0, 2.0], [0.5 0.0;])  # length mismatch
        @test_throws ArgumentError interpolate_idw([0.0 0.0;], [1.0], [0.5 0.0;]; k=0)
        @test_throws ArgumentError interpolate_idw([0.0 0.0;], [1.0], [0.5 0.0;]; power=0)
        @test_throws ArgumentError interpolate_idw([0.0 0.0;], [1.0], [0.5 0.0;]; max_distance=0)
    end
end

@testset "calculate_aboveground_height (Stage 4 redesign)" begin
    # Synthetic ground: 10×10 m flat-ish patch, with two canopy returns above.
    nxy = 30
    xs = collect(range(0.0, 10.0; length=nxy))
    ys = collect(range(0.0, 10.0; length=nxy))
    ground = Matrix{Float64}(undef, nxy * nxy, 3)
    idx = 1
    for y in ys, x in xs
        ground[idx, :] .= (x, y, 0.05 * sin(x) * cos(y))
        idx += 1
    end
    # Two canopy points 5 m and 10 m above the ground at known XY
    canopy = [
        5.0 5.0 5.0;
        2.5 7.5 10.0;
    ]
    coords = vcat(ground, canopy)
    pc = make_test_pointcloud(coords)
    ground_pc = make_test_pointcloud(ground)

    agh = calculate_aboveground_height(pc, ground_pc; xy_resolution=0.5)
    @test length(agh) == npoints(pc)
    # Ground points should have AGH ≈ 0 (within IDW interpolation tolerance)
    n_gnd = size(ground, 1)
    @test maximum(abs, agh[1:n_gnd]) < 0.5
    # Canopy points should recover their height above ground
    @test agh[n_gnd + 1] ≈ 5.0 atol=0.5
    @test agh[n_gnd + 2] ≈ 10.0 atol=0.5

    # Queries well outside the ground bbox → NaN
    far = make_test_pointcloud([1e6 1e6 0.0])
    agh_far = calculate_aboveground_height(far, ground_pc; xy_resolution=0.5)
    @test isnan(agh_far[1])

    @testset "in-bbox sparse-ground queries are still interpolated" begin
        # Regression for the bug where the Stage-4 pointwise IDW NaN'd out
        # queries that sit inside the ground bbox but in a sparsely-sampled
        # region. The restored grid-based AGH must give a finite value.
        sparse_ground = [
            0.0     0.0   0.0;
            100.0   0.0   1.0;
            100.0 100.0   2.0;
            0.0   100.0   1.0;
        ]
        sparse_pc = make_test_pointcloud(sparse_ground)
        query = make_test_pointcloud([50.0 50.0 5.0])
        agh_sparse = calculate_aboveground_height(query, sparse_pc; xy_resolution=0.5)
        @test isfinite(agh_sparse[1])
        @test agh_sparse[1] ≈ 4.0 atol=0.5  # z_query=5, IDW of corners ≈ 1
    end

    @testset "ground_polygon mask" begin
        # Tight polygon over only the first quadrant; second canopy point at (2.5, 7.5)
        # falls outside it and should become NaN.
        poly = [0.0 0.0; 6.0 0.0; 6.0 6.0; 0.0 6.0]
        agh_masked = calculate_aboveground_height(pc, ground_pc;
                                                  xy_resolution=0.5,
                                                  ground_polygon=poly)
        @test agh_masked[n_gnd + 1] ≈ 5.0 atol=0.5  # (5, 5) inside
        @test isnan(agh_masked[n_gnd + 2])           # (2.5, 7.5) outside
    end

    @testset "input validation" begin
        @test_throws ArgumentError calculate_aboveground_height(pc, ground_pc; xy_resolution=0.0)
        @test_throws ArgumentError calculate_aboveground_height(pc, ground_pc; xy_resolution=0.5, idw_k=0)
        @test_throws ArgumentError calculate_aboveground_height(pc, ground_pc; xy_resolution=0.5, idw_power=0.0)
    end
end
