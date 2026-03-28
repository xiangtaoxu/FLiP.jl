@testset "QSM" begin

    @testset "Taubin circle fit — full circle" begin
        # Generate points on a known circle (r=0.5, center=(1.0, 2.0))
        n = 100
        theta = range(0, 2π; length=n+1)[1:n]
        r_true = 0.5
        cx_true, cy_true = 1.0, 2.0
        u = cx_true .+ r_true .* cos.(theta) .+ 0.001 .* randn(n)
        v = cy_true .+ r_true .* sin.(theta) .+ 0.001 .* randn(n)

        cx, cy, r = FLiP.taubin_circle_fit(u, v)
        @test abs(cx - cx_true) < 0.02
        @test abs(cy - cy_true) < 0.02
        @test abs(r - r_true) < 0.02
    end

    @testset "Taubin circle fit — partial arc (180°)" begin
        n = 50
        theta = range(0, π; length=n)
        r_true = 0.3
        cx_true, cy_true = 0.0, 0.0
        u = cx_true .+ r_true .* cos.(theta) .+ 0.001 .* randn(n)
        v = cy_true .+ r_true .* sin.(theta) .+ 0.001 .* randn(n)

        cx, cy, r = FLiP.taubin_circle_fit(u, v)
        # Center should not be biased toward the arc midpoint
        @test abs(cx - cx_true) < 0.05
        @test abs(cy - cy_true) < 0.05
        @test abs(r - r_true) < 0.05
    end

    @testset "Perpendicular basis" begin
        d = (0.0, 0.0, 1.0)
        e1, e2 = FLiP._perp_basis(d)
        # Orthogonality
        @test abs(e1[1]*d[1] + e1[2]*d[2] + e1[3]*d[3]) < 1e-10
        @test abs(e2[1]*d[1] + e2[2]*d[2] + e2[3]*d[3]) < 1e-10
        @test abs(e1[1]*e2[1] + e1[2]*e2[2] + e1[3]*e2[3]) < 1e-10
        # Unit length
        @test abs(sqrt(e1[1]^2 + e1[2]^2 + e1[3]^2) - 1.0) < 1e-10
        @test abs(sqrt(e2[1]^2 + e2[2]^2 + e2[3]^2) - 1.0) < 1e-10
    end

    @testset "Smoothed centerline" begin
        centers = [Float64(i) for i in 1:10, j in 1:3]
        centers_copy = copy(centers)
        FLiP._smooth_centerline!(centers_copy, 1)
        # Interior points should be smoothed, endpoints less affected
        @test size(centers_copy) == size(centers)
        # First point: average of [1,2] → 1.5
        @test centers_copy[1, 1] ≈ (centers[1,1] + centers[2,1]) / 2
    end

    @testset "Synthetic cylinder QSM — integration" begin
        # Generate a vertical cylinder: r=0.1m, h=0.5m, 2000 points
        n = 2000
        r_true = 0.1
        h = 0.5

        theta = 2π .* rand(n)
        z = h .* rand(n)
        x = r_true .* cos.(theta) .+ 0.001 .* randn(n)
        y = r_true .* sin.(theta) .+ 0.001 .* randn(n)

        coords = hcat(x, y, z)
        nbs_ids = ones(Int32, n)
        tree_ids = ones(Int32, n)
        agh_values = z  # AGH = z for this simple case

        pc = make_test_pointcloud(coords; attrs=Dict(
            :nbs_id => nbs_ids,
            :tree_id => tree_ids,
            :AGH => agh_values,
            :node_id => ones(Int32, n),
            :tree_nbs_id => ones(Int32, n),
        ))
        tree_result = (
            pc_output=pc,
            skeleton_cloud=FLiP.PointCloudData(zeros(Float64, 0, 3), Dict{Symbol,Vector}()),
            filtered_cloud=pc,
            n_components=1,
            neighbor_radius=0.1,
        )

        # Use default config but make output_dir a tempdir
        cfg = FLiP._CFG
        old_subsample = cfg.pipeline_subsample_res
        cfg.pipeline_subsample_res = 0.05

        result = FLiP.qsm(
            tree_result=tree_result,
            config_path="",
            output_dir=mktempdir(),
            output_prefix="test",
        )

        cfg.pipeline_subsample_res = old_subsample

        @test result.status == :success || result.status == :no_linear_nbs
        if result.status == :success
            @test result.n_nodes > 0
            @test result.n_trees >= 1
            @test isfile(result.node_csv_path)
            @test isfile(result.tree_csv_path)

            # Read back CSV and check radius is reasonable
            lines = readlines(result.node_csv_path)
            @test length(lines) > 1  # header + at least one data row
            headers = split(lines[1], ",")
            r_area_col = findfirst(==("radius_area"), headers)
            if !isnothing(r_area_col) && length(lines) > 1
                vals = split(lines[2], ",")
                r_est = parse(Float64, vals[r_area_col])
                # Radius should be within 30% of true value
                @test abs(r_est - r_true) / r_true < 0.3
            end
        end
    end

    @testset "NBS linearity filtering" begin
        # Linear point set along z-axis: high linearity
        n = 100
        coords_linear = hcat(0.01 .* randn(n), 0.01 .* randn(n), collect(range(0, 1; length=n)))
        nbs_ids_linear = ones(Int32, n)

        cfg = FLiP._CFG
        result = FLiP._filter_linear_nbs(coords_linear, nbs_ids_linear, cfg)
        @test haskey(result, Int32(1))
        if haskey(result, Int32(1))
            @test result[Int32(1)].linearity > 0.9
        end

        # Spherical point set: low linearity
        coords_sphere = randn(n, 3)
        nbs_ids_sphere = ones(Int32, n)
        result_sphere = FLiP._filter_linear_nbs(coords_sphere, nbs_ids_sphere, cfg)
        @test !haskey(result_sphere, Int32(1))
    end

    @testset "Spline slice method" begin
        n = 200
        r = 0.15
        phi = range(-π, π; length=n+1)[1:n]
        rho = fill(r, n) .+ 0.001 .* randn(n)

        cfg = FLiP._CFG
        ca, circ, comp = FLiP._method_spline_slice(collect(rho), collect(phi), cfg)
        @test comp > 0.8
        @test abs(circ - 2π * r) / (2π * r) < 0.15
        @test abs(ca - π * r^2) / (π * r^2) < 0.15
    end

    @testset "Frustum metrics" begin
        # Cylinder (r1 = r2): should equal πr²h and 2πrh
        r = 0.1
        h = 0.5
        vol, sa = FLiP._frustum_metrics(r, r, h)
        @test vol ≈ π * r^2 * h atol=1e-10
        @test sa ≈ 2π * r * h atol=1e-10

        # Cone (r2 = 0): should equal (π/3)r²h
        vol_cone, _ = FLiP._frustum_metrics(r, 0.0, h)
        @test vol_cone ≈ (π / 3) * r^2 * h atol=1e-10
    end

    @testset "QSM with no data" begin
        result = FLiP.qsm(tree_result=nothing, output_dir=mktempdir(), output_prefix="empty")
        @test result.status == :no_data
        @test result.n_nodes == 0
    end

    @testset "2D surface smoothing — uniform surface unchanged" begin
        nphi, nz = 36, 5
        surface = fill(0.1, nphi, nz)
        surface_copy = copy(surface)
        FLiP._smooth_surface_2d!(surface_copy, 0.5, 0.3, 3)
        @test all(abs.(surface_copy .- surface) .< 1e-12)
    end

    @testset "2D surface smoothing — z-gradient reduced but preserved" begin
        nphi, nz = 36, 10
        # Step function: slices 1–5 at r=0.1, slices 6–10 at r=0.2
        surface = fill(0.1, nphi, nz)
        surface[:, 6:10] .= 0.2

        FLiP._smooth_surface_2d!(surface, 0.5, 0.3, 1)

        # Interior of each plateau should remain close to original
        @test all(abs.(surface[:, 1] .- 0.1) .< 1e-10)    # boundary slice, no z-neighbor below
        @test all(abs.(surface[:, 10] .- 0.2) .< 1e-10)   # boundary slice, no z-neighbor above
        # Near the step (slices 5 and 6) values should have moved toward each other
        @test all(surface[:, 5] .> 0.1)
        @test all(surface[:, 6] .< 0.2)
    end

    @testset "2D gap fill — partial coverage filled from neighbors" begin
        nphi, nz = 36, 3
        surface = fill(0.15, nphi, nz)
        # Create a 90° gap (9 bins) in the middle slice only
        gap_bins = 1:9
        surface[gap_bins, 2] .= NaN

        FLiP._fill_gaps_2d!(surface)

        # All NaN cells should now be filled
        @test all(isfinite.(surface))
        # Filled values should be close to 0.15 (neighbors are all 0.15)
        @test all(abs.(surface[gap_bins, 2] .- 0.15) .< 1e-10)
    end

    @testset "2D build_rho_surface — basic binning" begin
        # 4 points, 2 slices, 4 phi bins
        rho = [0.1, 0.2, 0.15, 0.25]
        phi = [-π + 0.1, -π + 0.1, π - 0.1, π - 0.1]  # bins 1 and 4
        pt_slice_ids = [1, 1, 2, 2]
        surface = FLiP._build_rho_surface(rho, phi, pt_slice_ids, 2, 4)
        @test size(surface) == (4, 2)
        # First two points go to bin 1, slice 1 → mean = 0.15
        @test surface[1, 1] ≈ 0.15
        # Cells with no points should be NaN
        @test isnan(surface[2, 1])
    end

    @testset "2D spline method — full cylinder" begin
        # Generate a full cylinder: 3 slices, r=0.15
        r = 0.15
        n_per_slice = 200
        n_slices = 3
        n = n_per_slice * n_slices

        rho = fill(r, n) .+ 0.001 .* randn(n)
        phi = repeat(range(-π, π; length=n_per_slice + 1)[1:n_per_slice], n_slices)
        pt_slice_ids = vcat([fill(s, n_per_slice) for s in 1:n_slices]...)

        cfg = FLiP._CFG
        results = FLiP._method_spline_2d(rho, phi, pt_slice_ids, n_slices, cfg, r)

        @test length(results) == n_slices
        for s in 1:n_slices
            @test results[s].completeness > 0.8
            @test abs(results[s].circumference - 2π * r) / (2π * r) < 0.15
            @test abs(results[s].cross_area - π * r^2) / (π * r^2) < 0.15
        end
    end

end
