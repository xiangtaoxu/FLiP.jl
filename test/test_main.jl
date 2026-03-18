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

        @test result.preprocess_written
        @test result.ground_written
        @test result.agh_written
        @test result.tree_written == false
        @test result.tree_skeleton_written == false
        @test result.qsm_result.status == :skipped
        @test result.report_result.status == :skipped

        @test basename(result.preprocess_path) == "test_preprocess.las"
        @test basename(result.ground_path) == "test_ground.las"
        @test basename(result.agh_path) == "test_agh.las"

        @test isfile(result.preprocess_path)
        @test isfile(result.ground_path)
        @test isfile(result.agh_path)
    finally
        rm(test_dir; recursive=true, force=true)
    end
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
""")
        end

        @test_throws ArgumentError run_pipeline(config_path)
    finally
        rm(test_dir; recursive=true, force=true)
    end
end