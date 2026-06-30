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
        old_subsample = cfg.pipeline.subsample_res
        cfg.pipeline.subsample_res = 0.05

        result = FLiP.qsm(
            tree_result=tree_result,
            config_path="",
            output_dir=mktempdir(),
            output_prefix="test",
        )

        cfg.pipeline.subsample_res = old_subsample

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

    @testset "Lean trial QSM — fit-only, grouped by nbs_id" begin
        # Same synthetic cylinder, but run the lean trial pass: group by :nbs_id,
        # emit :trial_node_id, and skip surface cloud + per-tree aggregation.
        n = 2000
        r_true = 0.1; h = 0.5
        theta = 2π .* rand(n); z = h .* rand(n)
        x = r_true .* cos.(theta) .+ 0.001 .* randn(n)
        y = r_true .* sin.(theta) .+ 0.001 .* randn(n)
        coords = hcat(x, y, z)

        pc = make_test_pointcloud(coords; attrs=Dict(
            :nbs_id => ones(Int32, n),
            :AGH    => z,
        ))
        tree_result = (
            pc_output=pc,
            skeleton_cloud=FLiP.PointCloud(zeros(Float64, 0, 3), Dict{Symbol,Vector}()),
            filtered_cloud=pc, n_components=1, neighbor_radius=0.1,
        )

        cfg = FLiP._CFG
        old_subsample = cfg.pipeline.subsample_res
        cfg.pipeline.subsample_res = 0.05
        outdir = mktempdir()
        result = FLiP.qsm(tree_result=tree_result, output_dir=outdir, output_prefix="lean",
                          lean=true, group_attr=:nbs_id, node_id_attr=:trial_node_id)
        cfg.pipeline.subsample_res = old_subsample

        @test result.status == :success || result.status == :no_linear_nbs
        if result.status == :success
            @test result.n_nodes > 0
            @test result.n_trees == 0                          # no tree aggregation in lean
            @test !isfile(result.tree_csv_path)                # tree CSV not written
            @test npoints(result.qsm_surface_cloud) == 0       # no surface cloud
            @test hasattribute(pc, :trial_node_id)             # emitted under requested name
            # grouping label flows into both QSMNode.nbs_id and tree_nbs_id
            @test all(nd.nbs_id == nd.tree_nbs_id == Int32(1) for nd in result.nodes)
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
        results, surface_grid, phi_bins = FLiP._method_spline_2d(rho, phi, pt_slice_ids, n_slices, cfg)

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

    @testset "_filter_rho_outliers — drops outliers per slice and compacts arrays" begin
        # Two slices: slice 1 has three shell points + one outlier at rho=0.50;
        # slice 2 is all shell. Percentile=0.75 should drop slice 1's outlier
        # (keeping the 3 shell points whose quantile is 0.10).
        rho = [0.10, 0.10, 0.10, 0.50, 0.10, 0.10, 0.10]
        phi = [0.0, 1.0, 2.0, 3.0, -1.0, 0.0, 1.0]
        pt_slice_ids = [1, 1, 1, 1, 2, 2, 2]
        slice_point_indices = [Int[1, 2, 3, 4], Int[5, 6, 7]]
        indices = collect(1:7)

        rho2, phi2, ids2, spi2, idx2 = FLiP._filter_rho_outliers(
            rho, phi, pt_slice_ids, slice_point_indices, indices, 2, 0.75)
        @test length(rho2) == 6
        @test !(0.50 in rho2)                                  # outlier dropped
        @test maximum(rho2) <= 0.10 + 1e-12
        @test length(spi2[1]) == 3                             # slice 1 lost one
        @test length(spi2[2]) == 3                             # slice 2 unchanged
        @test all(1 .<= spi2[1] .<= length(rho2))              # compacted indices
        @test all(1 .<= spi2[2] .<= length(rho2))

        # Default 1.0 is a no-op
        rho3, _, _, _, _ = FLiP._filter_rho_outliers(
            rho, phi, pt_slice_ids, slice_point_indices, indices, 2, 1.0)
        @test rho3 === rho
    end

    @testset "_build_rho_surface — mean binning, NaN for empty cells" begin
        rho = [0.1, 0.2, 0.15, 0.25]
        phi = [-π + 0.1, -π + 0.1, π - 0.1, π - 0.1]
        pt_slice_ids = [1, 1, 1, 1]
        surface = FLiP._build_rho_surface(rho, phi, pt_slice_ids, 1, 4)
        # Bin 1 (phi ~ -π + 0.1) gets [0.1, 0.2] → mean 0.15
        # Bin 4 (phi ~ π - 0.1)  gets [0.15, 0.25] → mean 0.20
        @test surface[1, 1] ≈ 0.15
        @test surface[4, 1] ≈ 0.20
        @test isnan(surface[2, 1])
        @test isnan(surface[3, 1])
    end

    @testset "_finalize_centerline! — interpolation (window=0)" begin
        # 5 slices, slice 3 is NaN; should be interpolated as midpoint of 2 and 4.
        # window=0 skips smoothing so we can assert exact interpolated values.
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
        FLiP._finalize_centerline!(centers, valid, info, 0.0, 1.0; window=0)
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
        FLiP._finalize_centerline!(centers2, valid2, info, 0.0, 1.0; window=0)
        @test centers2[5, 1] ≈ 3.0
        @test centers2[5, 2] ≈ 2.0
        @test centers2[5, 3] ≈ 0.0
    end

    @testset "_finalize_centerline! — smoothing pass averages neighbors" begin
        # All slices valid (no NaN), so only the smoothing pass mutates centers.
        centers = [Float64(i) for i in 1:10, _ in 1:3]
        valid = trues(10)
        info = FLiP.NBSInfo((1.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            (0.1, 0.2, 1.0), 0.8, Int[])
        FLiP._finalize_centerline!(centers, valid, info, 0.0, 1.0; window=1)
        # First row: avg of rows 1..2 → 1.5
        @test centers[1, 1] ≈ 1.5
        # Interior row 5: avg of rows 4..6 → 5.0
        @test centers[5, 1] ≈ 5.0
        # Last row: avg of rows 9..10 → 9.5
        @test centers[10, 1] ≈ 9.5
    end

    @testset "QC: fused NBS — two parallel cylinders share one nbs_id" begin
        using Statistics: mean
        # Two vertical cylinders 0.30 m apart in x, both r=0.05, h=1.0, sharing
        # one nbs_id. With QC off, the per-slice fit straddles both clusters and
        # inflates the fitted radius. With QC on, the 3D CC step drops the
        # secondary cluster and the fit returns ≈ true radius.
        n_per_level = 36
        z_levels    = collect(range(0.0, 1.0; length=11))
        true_r      = 0.05
        gap         = 0.30
        coords = Float64[]
        for z in z_levels, k in 1:n_per_level
            θ = 2π * (k - 1) / n_per_level
            push!(coords, true_r * cos(θ),       true_r * sin(θ), z)   # cyl A
            push!(coords, gap + true_r * cos(θ), true_r * sin(θ), z)   # cyl B
        end
        coords_mat = collect(reshape(coords, 3, :)')
        n_pts = size(coords_mat, 1)
        nbs_ids = ones(Int32, n_pts)

        cfg_off = deepcopy(FLiP._CFG)
        cfg_off.pipeline.subsample_res     = 0.05
        cfg_off.qsm.completeness_threshold = 0.1
        cfg_off.qsm.qc_enable              = false
        cfg_on = deepcopy(cfg_off)
        cfg_on.qsm.qc_enable               = true
        # CC link radius = QC_CC_RADIUS_SCALAR (=2.0) × subsample_res = 0.10 m < cluster gap 0.20 m

        linear_off = FLiP._filter_linear_nbs(coords_mat, nbs_ids, cfg_off)
        linear_on  = FLiP._filter_linear_nbs(coords_mat, nbs_ids, cfg_on)
        @test linear_off[1] !== nothing
        @test linear_on[1]  !== nothing

        nodes_off = FLiP.QSMNode[]; qids_off = zeros(Int32, n_pts)
        nodes_on  = FLiP.QSMNode[]; qids_on  = zeros(Int32, n_pts)
        agh  = Float64.(coords_mat[:, 3])
        tids = ones(Int32, n_pts)
        FLiP._process_single_nbs!(nodes_off, qids_off, coords_mat, linear_off[1], Int32(1),
                                  tids, agh, cfg_off, 1)
        FLiP._process_single_nbs!(nodes_on,  qids_on,  coords_mat, linear_on[1],  Int32(1),
                                  tids, agh, cfg_on,  1)

        @test !isempty(nodes_off) && !isempty(nodes_on)
        r_off = mean([nd.radius_area for nd in nodes_off])
        r_on  = mean([nd.radius_area for nd in nodes_on])
        @test r_off > 2.0 * true_r          # straddles both cylinders
        @test r_on  < 1.5 * true_r          # hugs one cylinder
        @test r_on  < 0.5 * r_off           # meaningful reduction
        # QC should drop the secondary cluster's points
        @test count(>(0), qids_on) < count(>(0), qids_off)
    end

    @testset "QC: noisy blob — outliers around a clean cylinder" begin
        using Statistics: mean
        # Single cylinder r=0.05 + 10% outlier points scattered in a 0.30 m
        # box around it. Without QC the outer points inflate the fitted
        # radius; with QC the small-CC drop + 3D SOR strip them and the fit
        # returns ≈ true radius.
        n_per_level = 36
        z_levels    = collect(range(0.0, 1.0; length=11))
        true_r      = 0.05
        coords = Float64[]
        for z in z_levels, k in 1:n_per_level
            θ = 2π * (k - 1) / n_per_level
            push!(coords, true_r * cos(θ), true_r * sin(θ), z)
        end
        n_clean = length(coords) ÷ 3
        for _ in 1:(n_clean ÷ 10)
            push!(coords, 0.3 * (rand() - 0.5), 0.3 * (rand() - 0.5), rand())
        end
        coords_mat = collect(reshape(coords, 3, :)')
        n_pts = size(coords_mat, 1)
        nbs_ids = ones(Int32, n_pts)

        cfg_off = deepcopy(FLiP._CFG)
        cfg_off.pipeline.subsample_res     = 0.05
        cfg_off.qsm.completeness_threshold = 0.1
        cfg_off.qsm.qc_enable              = false
        cfg_on = deepcopy(cfg_off)
        cfg_on.qsm.qc_enable               = true

        linear_off = FLiP._filter_linear_nbs(coords_mat, nbs_ids, cfg_off)
        linear_on  = FLiP._filter_linear_nbs(coords_mat, nbs_ids, cfg_on)
        @test linear_off[1] !== nothing
        @test linear_on[1]  !== nothing

        nodes_off = FLiP.QSMNode[]; qids_off = zeros(Int32, n_pts)
        nodes_on  = FLiP.QSMNode[]; qids_on  = zeros(Int32, n_pts)
        agh  = Float64.(coords_mat[:, 3])
        tids = ones(Int32, n_pts)
        FLiP._process_single_nbs!(nodes_off, qids_off, coords_mat, linear_off[1], Int32(1),
                                  tids, agh, cfg_off, 1)
        FLiP._process_single_nbs!(nodes_on,  qids_on,  coords_mat, linear_on[1],  Int32(1),
                                  tids, agh, cfg_on,  1)

        @test !isempty(nodes_off) && !isempty(nodes_on)
        r_off = mean([nd.radius_area for nd in nodes_off])
        r_on  = mean([nd.radius_area for nd in nodes_on])
        @test r_on < r_off                           # QC reduces inflation
        @test abs(r_on - true_r) / true_r < 0.5      # close to truth after QC
        # QC drops at least some of the noise points
        @test count(>(0), qids_on) < count(>(0), qids_off)
    end

    @testset "QC: clean cylinder is near-no-op" begin
        using Statistics: mean
        # Clean cylinder — QC on vs off should produce essentially the same
        # fitted radius. QC must not "shave" valid stem surfaces.
        n_per_level = 36
        z_levels    = collect(range(0.0, 1.0; length=11))
        true_r      = 0.08
        coords = Float64[]
        for z in z_levels, k in 1:n_per_level
            θ = 2π * (k - 1) / n_per_level
            push!(coords, true_r * cos(θ) + 0.001*randn(),
                          true_r * sin(θ) + 0.001*randn(), z)
        end
        coords_mat = collect(reshape(coords, 3, :)')
        n_pts = size(coords_mat, 1)
        nbs_ids = ones(Int32, n_pts)

        cfg_off = deepcopy(FLiP._CFG)
        cfg_off.pipeline.subsample_res     = 0.05
        cfg_off.qsm.completeness_threshold = 0.1
        cfg_off.qsm.qc_enable              = false
        cfg_on = deepcopy(cfg_off)
        cfg_on.qsm.qc_enable               = true

        linear_off = FLiP._filter_linear_nbs(coords_mat, nbs_ids, cfg_off)
        linear_on  = FLiP._filter_linear_nbs(coords_mat, nbs_ids, cfg_on)
        nodes_off = FLiP.QSMNode[]; qids_off = zeros(Int32, n_pts)
        nodes_on  = FLiP.QSMNode[]; qids_on  = zeros(Int32, n_pts)
        agh  = Float64.(coords_mat[:, 3])
        tids = ones(Int32, n_pts)
        FLiP._process_single_nbs!(nodes_off, qids_off, coords_mat, linear_off[1], Int32(1),
                                  tids, agh, cfg_off, 1)
        FLiP._process_single_nbs!(nodes_on,  qids_on,  coords_mat, linear_on[1],  Int32(1),
                                  tids, agh, cfg_on,  1)

        @test length(nodes_off) == length(nodes_on)
        r_off = mean([nd.radius_area for nd in nodes_off])
        r_on  = mean([nd.radius_area for nd in nodes_on])
        @test abs(r_on - r_off) / r_off < 0.05      # within 5%
        @test abs(r_on - true_r) / true_r < 0.10    # within 10% of truth
    end

end
