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
            skeleton_cloud=FLiP.PointCloud(zeros(Float64, 0, 3), Dict{Symbol,Vector}()),
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

            # Check new quality metric columns exist
            @test "rho_mean" in headers
            @test "rho_std" in headers
            @test "rho_cv" in headers
            # For a clean cylinder, CV should be low
            rho_cv_col = findfirst(==("rho_cv"), headers)
            if !isnothing(rho_cv_col) && length(lines) > 1
                cv_val = parse(Float64, split(lines[2], ",")[rho_cv_col])
                @test cv_val < 0.15
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
        @test length(result) >= 1
        @test result[1] !== nothing
        if result[1] !== nothing
            @test result[1].linearity > 0.9
        end

        # Spherical point set: low linearity
        coords_sphere = randn(n, 3)
        nbs_ids_sphere = ones(Int32, n)
        result_sphere = FLiP._filter_linear_nbs(coords_sphere, nbs_ids_sphere, cfg)
        @test isempty(result_sphere) || result_sphere[1] === nothing
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
        results, surface_grid, phi_bins = FLiP._method_spline_2d(rho, phi, pt_slice_ids, n_slices, cfg, r)

        @test length(results) == n_slices
        for s in 1:n_slices
            @test results[s].completeness > 0.8
            @test abs(results[s].circumference - 2π * r) / (2π * r) < 0.15
            @test abs(results[s].cross_area - π * r^2) / (π * r^2) < 0.15
        end
    end

    @testset "Rho quality metrics — shell has low CV" begin
        using Statistics: mean, std
        # Shell: all points near r=0.15 → CV should be very low
        r = 0.15
        n_pts = 200
        rho_shell = fill(r, n_pts) .+ 0.001 .* randn(n_pts)
        cv_shell = std(rho_shell) / mean(rho_shell)
        @test cv_shell < 0.1

        # Filled disc: uniform from 0 to r → CV should be high
        rho_disc = r .* rand(n_pts)
        cv_disc = std(rho_disc) / mean(rho_disc)
        @test cv_disc > 0.3
    end

    @testset "Rho percentile filter — shrinks toward inner surface" begin
        using Statistics: mean
        # Mix of shell points (r=0.15) and outer noise/leaf returns (r=0.20..0.40)
        n_shell = 100; n_noise = 100
        rho = vcat(fill(0.15, n_shell) .+ 0.002 .* randn(n_shell),
                   0.20 .+ 0.20 .* rand(n_noise))
        phi = collect(range(-π, π; length=n_shell + n_noise + 1)[1:n_shell + n_noise])
        pt_slice_ids = ones(Int, n_shell + n_noise)

        surface_full = FLiP._build_rho_surface(rho, phi, pt_slice_ids, 1, 36, 1.0)
        surface_75   = FLiP._build_rho_surface(rho, phi, pt_slice_ids, 1, 36, 0.75)

        # Percentile-filtered surface should have smaller mean rho (closer to inner shell)
        mean_full = mean(filter(isfinite, surface_full))
        mean_75   = mean(filter(isfinite, surface_75))
        @test mean_75 < mean_full
    end

    @testset "build_rho_surface percentile=1.0 matches default" begin
        rho = [0.1, 0.2, 0.15, 0.25]
        phi = [-π + 0.1, -π + 0.1, π - 0.1, π - 0.1]
        pt_slice_ids = [1, 1, 1, 1]
        s1 = FLiP._build_rho_surface(rho, phi, pt_slice_ids, 1, 4)
        s2 = FLiP._build_rho_surface(rho, phi, pt_slice_ids, 1, 4, 1.0)
        for i in eachindex(s1)
            if isnan(s1[i])
                @test isnan(s2[i])
            else
                @test s1[i] ≈ s2[i]
            end
        end
    end

    @testset "_interpolate_invalid_slices! — midpoint and one-sided fallback" begin
        # 5 slices, slice 3 is NaN; should be interpolated as midpoint of 2 and 4
        centers = Float64[
            0.0  0.0  0.0
            1.0  0.0  0.0
            NaN  NaN  NaN
            3.0  2.0  0.0
            4.0  3.0  0.0
        ]
        valid = Bool[true, true, false, true, true]
        info = FLiP.NBSInfo((1.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            (0.1, 0.2, 1.0), 0.8, Int[])
        FLiP._interpolate_invalid_slices!(centers, valid, info, 0.0, 1.0)
        @test centers[3, 1] ≈ 2.0
        @test centers[3, 2] ≈ 1.0
        @test centers[3, 3] ≈ 0.0

        # NaN at end (slice 5): should copy slice 4
        centers2 = Float64[
            0.0  0.0  0.0
            1.0  0.0  0.0
            2.0  1.0  0.0
            3.0  2.0  0.0
            NaN  NaN  NaN
        ]
        valid2 = Bool[true, true, true, true, false]
        FLiP._interpolate_invalid_slices!(centers2, valid2, info, 0.0, 1.0)
        @test centers2[5, 1] ≈ 3.0
        @test centers2[5, 2] ≈ 2.0
        @test centers2[5, 3] ≈ 0.0
    end

    @testset "_compute_slice_rho_stats — exact two-slice means and std" begin
        # Slice 1: rho ∈ {0.1, 0.2, 0.3}; mean = 0.2, var = 0.01, std = 0.1
        # Slice 2: rho ∈ {1.0, 1.0, 1.0}; mean = 1.0, std = 0
        rho = [0.1, 0.2, 0.3, 1.0, 1.0, 1.0]
        pt_slice_ids = [1, 1, 1, 2, 2, 2]
        mn, sd, cv = FLiP._compute_slice_rho_stats(rho, pt_slice_ids, 2)
        @test mn[1] ≈ 0.2
        @test mn[2] ≈ 1.0
        @test sd[1] ≈ 0.1
        @test sd[2] ≈ 0.0
        @test cv[1] ≈ 0.5
        @test cv[2] ≈ 0.0
    end

    @testset "Terminal-slice rho-percentile fallback shrinks noisy radius" begin
        # Build a single linear NBS along z with a noisy "leafy" outer ring at one slice.
        # When opting into a high terminal_completeness_threshold + low percentile,
        # the noisy slice's radius should shrink below the spline-derived value while
        # a clean fully-covered slice's radius is unchanged.
        n_per_slice = 64
        z_levels = [0.0, 0.5, 1.0, 1.5, 2.0]
        true_r = 0.10
        coords = Float64[]
        for (s, z) in enumerate(z_levels)
            # All clean slices except slice 3 which has a few inflated rho outliers
            for k in 1:n_per_slice
                θ = 2π * (k - 1) / n_per_slice
                r = true_r
                # On slice 3, half the points are leafy (3× the true radius)
                if s == 3 && k <= n_per_slice ÷ 2
                    r = true_r * 3.0
                end
                push!(coords, r * cos(θ), r * sin(θ), z)
            end
        end
        coords_mat = reshape(coords, 3, :)'
        nbs_ids = ones(Int32, size(coords_mat, 1))

        # Run with defaults (no-op terminal fallback)
        cfg_default = deepcopy(FLiP._CFG)
        cfg_default.pipeline_subsample_res = 0.1
        cfg_default.qsm_completeness_threshold = 0.1
        linear_default = FLiP._filter_linear_nbs(coords_mat, nbs_ids, cfg_default)
        @test length(linear_default) >= 1 && linear_default[1] !== nothing
        nodes_default = FLiP.QSMNode[]
        qids = zeros(Int32, size(coords_mat, 1))
        agh = Float64.(coords_mat[:, 3])
        tree_ids_v = ones(Int32, size(coords_mat, 1))
        FLiP._process_single_nbs!(nodes_default, qids, coords_mat,
                                  linear_default[1], Int32(1),
                                  tree_ids_v, agh, cfg_default, 1)

        # Run with opt-in terminal fallback
        cfg_term = deepcopy(cfg_default)
        cfg_term.qsm_terminal_completeness_threshold = 1.5  # always trigger fallback
        cfg_term.qsm_terminal_rho_percentile = 0.25         # inner quartile
        nodes_term = FLiP.QSMNode[]
        qids2 = zeros(Int32, size(coords_mat, 1))
        FLiP._process_single_nbs!(nodes_term, qids2, coords_mat,
                                  linear_default[1], Int32(1),
                                  tree_ids_v, agh, cfg_term, 1)

        # Should have nodes at every clean slice in both runs
        @test length(nodes_default) >= length(z_levels) - 1
        @test length(nodes_term) == length(nodes_default)

        # Find the inflated slice (highest radius_area in default run) and verify shrinkage
        i_max = argmax([nd.radius_area for nd in nodes_default])
        @test nodes_term[i_max].radius_area < nodes_default[i_max].radius_area
        # And the shrunken radius is close to the true inner radius
        @test nodes_term[i_max].radius_area < 0.2
    end

end
