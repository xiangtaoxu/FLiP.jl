@testset "Main pipeline" begin
    test_dir = mktempdir()

    try
        input_path = joinpath(test_dir, "input.las")
        output_dir = joinpath(test_dir, "out")
        config_path = joinpath(test_dir, "flip_config.toml")

        xs = collect(0.0:0.5:9.5)
        ys = collect(0.0:0.5:9.5)

        nxy = length(xs) * length(ys)
        ground = Matrix{Float64}(undef, nxy, 3)
        canopy = Matrix{Float64}(undef, nxy, 3)

        idx = 1
        for y in ys, x in xs
            zg = 0.2 * sin(0.3 * x) * cos(0.3 * y)
            ground[idx, :] .= (x, y, zg)
            canopy[idx, :] .= (x, y, zg + 8.0)
            idx += 1
        end

        coords = vcat(ground, canopy)
        pc = make_test_pointcloud(coords)
        write_las(input_path, pc)

        open(config_path, "w") do io
            write(io, """
[pipeline]
input_path = \"$input_path\"
output_dir = \"$output_dir\"
output_prefix = \"test_\"

subsample_res = 0.1
enable_subsample = true
enable_preprocess = true
enable_agh = true

xy_resolution = 0.5
idw_k = 8
idw_power = 2.0

enable_ground_segmentation = true
enable_tree_segmentation = false
enable_qsm = false
enable_generate_report = false

[segment_ground]
voxel_size = 0.5
min_cc_size = 20
grid_size = 0.5
cone_theta_deg = 60.0

[preprocess]
enable_statistical_filter = false
""")
        end

        result = run_pipeline(config_path)

        @test result.preprocess.written
        @test result.ground.written
        @test result.agh.written
        @test result.tree.written == false
        @test result.qsm.status == :skipped
        @test result.report.status == :skipped

        @test basename(result.preprocess.path) == "test_preprocess.las"
        @test basename(result.ground.path) == "test_ground.las"
        @test basename(result.agh.path) == "test_agh.las"

        @test isfile(result.preprocess.path)
        @test isfile(result.ground.path)
        @test isfile(result.agh.path)
    finally
        rm(test_dir; recursive=true, force=true)
    end
end

@testset "ground_segmentation does not mutate input cloud" begin
    # Regression test for Stage 1.3: when crop is disabled and AGH is enabled,
    # the previous code mutated the input PointCloud via setattribute!(pc_use, :AGH, ...).
    test_dir = mktempdir()
    try
        nxy = 20
        xs = collect(range(0.0, 9.5; length=nxy))
        ys = collect(range(0.0, 9.5; length=nxy))
        ground = Matrix{Float64}(undef, nxy * nxy, 3)
        canopy = Matrix{Float64}(undef, nxy * nxy, 3)
        idx = 1
        for y in ys, x in xs
            ground[idx, :] .= (x, y, 0.0)
            canopy[idx, :] .= (x, y, 5.0)
            idx += 1
        end
        coords = vcat(ground, canopy)
        pc = make_test_pointcloud(coords; attrs=Dict(:intensity => fill(UInt16(0), size(coords, 1))))

        # Load a config with crop disabled, AGH enabled — the case that previously mutated pc.
        input_path = joinpath(test_dir, "input.las")
        output_dir = joinpath(test_dir, "out")
        config_path = joinpath(test_dir, "flip_config.toml")
        write_las(input_path, pc)
        open(config_path, "w") do io
            write(io, """
[pipeline]
input_path = \"$input_path\"
output_dir = \"$output_dir\"
output_prefix = \"test_\"
enable_preprocess = false
enable_subsample = false
enable_ground_segmentation = true
enable_ground_crop = false
enable_agh = true
enable_tree_segmentation = false
enable_qsm = false
enable_generate_report = false
xy_resolution = 0.5
idw_k = 8
idw_power = 2.0
[segment_ground]
voxel_size = 0.5
min_cc_size = 5
grid_size = 0.5
cone_theta_deg = 60.0
[preprocess]
enable_statistical_filter = false
""")
        end
        FLiP.load_config!(config_path)
        cfg = FLiP._CFG

        orig_keys = sort(collect(keys(pc.attrs)))
        res = ground_segmentation(pc; cfg=cfg)

        # The input cloud's attribute dict must be unchanged.
        @test sort(collect(keys(pc.attrs))) == orig_keys
        @test !hasattribute(pc, :AGH)
        # But the returned agh_cloud must carry :AGH.
        @test hasattribute(res.agh_cloud, :AGH)
    finally
        rm(test_dir; recursive=true, force=true)
    end
end

@testset "_apply_preprocess_filters honors both filter flags (Stage 5)" begin
    # Regression test: previously the E57 raw path only triggered when subsample
    # was enabled, leaving stat-filter-only E57 inputs stuck on the full-cloud
    # path. The shared helper now runs whichever subset of filters is enabled.
    coords = randn(500, 3)
    # Add 5 obvious outliers
    coords[end-4:end, :] .= [1e3 1e3 1e3; -1e3 -1e3 -1e3; 1e3 -1e3 0.0; 0.0 1e3 -1e3; -1e3 0.0 1e3]
    attrs = Dict{Symbol,Vector}(:intensity => collect(UInt16(1):UInt16(500)))

    cfg = deepcopy(FLiP._CFG)

    # Both off → identity
    cfg.pipeline.enable_subsample = false
    cfg.preprocess.enable_statistical_filter = false
    c, a = FLiP._apply_preprocess_filters(coords, attrs; cfg=cfg)
    @test size(c) == size(coords)
    @test length(a[:intensity]) == 500

    # Subsample only
    cfg.pipeline.enable_subsample = true
    cfg.preprocess.enable_statistical_filter = false
    cfg.pipeline.subsample_res = 0.5
    c1, a1 = FLiP._apply_preprocess_filters(coords, attrs; cfg=cfg)
    @test size(c1, 1) <= 500
    @test length(a1[:intensity]) == size(c1, 1)
    @test a1 isa Dict{Symbol,Vector}

    # Stat-filter only (the previously-broken E57 case)
    cfg.pipeline.enable_subsample = false
    cfg.preprocess.enable_statistical_filter = true
    cfg.statistical_filter.k_neighbors = 10
    cfg.statistical_filter.n_sigma = 2.0
    c2, a2 = FLiP._apply_preprocess_filters(coords, attrs; cfg=cfg)
    @test size(c2, 1) < 500            # outliers removed
    @test size(c2, 1) >= 490           # most clean points kept
    @test length(a2[:intensity]) == size(c2, 1)
    @test a2 isa Dict{Symbol,Vector}

    # Both on
    cfg.pipeline.enable_subsample = true
    cfg.preprocess.enable_statistical_filter = true
    c3, a3 = FLiP._apply_preprocess_filters(coords, attrs; cfg=cfg)
    @test size(c3, 1) <= size(c1, 1)
    @test length(a3[:intensity]) == size(c3, 1)
    @test a3 isa Dict{Symbol,Vector}

    # Helper does not mutate its inputs
    @test length(attrs[:intensity]) == 500

    # Regression: a filtered cloud with NO attributes (e.g. an E57 with neither
    # intensity nor color) must still yield a Dict{Symbol,Vector}, so the
    # downstream `PointCloud(coords, attrs)` constructor accepts it. Previously
    # the untyped comprehension produced a Dict{Any,Any} here and threw a
    # MethodError at construction time.
    empty_attrs = Dict{Symbol,Vector}()
    cfg.pipeline.enable_subsample = false
    cfg.preprocess.enable_statistical_filter = true
    c4, a4 = FLiP._apply_preprocess_filters(Float32.(coords), empty_attrs; cfg=cfg)
    @test a4 isa Dict{Symbol,Vector}
    @test isempty(a4)
    @test FLiP.PointCloud(c4, a4) isa FLiP.PointCloud   # the real call site
end

@testset "Main pipeline aborts when upstream data missing" begin
    test_dir = mktempdir()

    try
        input_path = joinpath(test_dir, "input.las")
        output_dir = joinpath(test_dir, "out")
        config_path = joinpath(test_dir, "flip_config.toml")

        coords = [
            0.0 0.0 0.0;
            0.0 0.0 5.0;
            1.0 0.0 0.0;
            1.0 0.0 5.0;
        ]
        pc = make_test_pointcloud(coords)
        write_las(input_path, pc)

        open(config_path, "w") do io
            write(io, """
[pipeline]
input_path = \"$input_path\"
output_dir = \"$output_dir\"
output_prefix = \"test_\"

subsample_res = 0.1
enable_subsample = false
enable_preprocess = true
enable_agh = false

xy_resolution = 0.5
idw_k = 8
idw_power = 2.0

enable_ground_segmentation = false
enable_tree_segmentation = true
enable_qsm = false
enable_generate_report = false

[segment_ground]
voxel_size = 0.5
min_cc_size = 2
grid_size = 0.5
cone_theta_deg = 60.0

[preprocess]
enable_statistical_filter = false

[tree_segmentation]
nearground_agh_threshold = 0.3
neighbor_radius = 1.0
slice_length = 0.1
min_cc_size = 2
max_lcs_iterations = 10
nbs_neighbor_distance = 2
nbs_min_segment_size = 2
linearity_angle_deg = 60.0
""")
        end

        @test_throws ArgumentError run_pipeline(config_path)
    finally
        rm(test_dir; recursive=true, force=true)
    end
end