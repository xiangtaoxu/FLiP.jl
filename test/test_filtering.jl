@testset "Filtering" begin
    @testset "Statistical outlier removal" begin
        # Create clean data with outliers
        coords = randn(Float64, 100, 3)
        # Add some outliers far from the cluster
        outliers = [
            10.0 10.0 10.0;
            -10.0 -10.0 -10.0;
            15.0 0.0 0.0;
        ]
        coords_with_outliers = vcat(coords, outliers)
        
        # Filter with default parameters
        indices = statistical_filter_indices(coords_with_outliers, 10, 2.0)
        @test length(indices) < size(coords_with_outliers, 1)
        # Most clean points should be kept
        @test length(indices) >= size(coords, 1) * 0.9
        
        # Test with PointCloud
        pc = make_test_pointcloud(coords_with_outliers)
        pc_clean = statistical_filter(pc, 10, 2.0)
        @test length(pc_clean) < length(pc)
        
        # Test with too few points
        small_coords = rand(5, 3)
        indices = statistical_filter_indices(small_coords, 10, 2.0)
        @test length(indices) == 5  # Should keep all points
        
        # Test error handling
        @test_throws ArgumentError statistical_filter_indices(coords, 0, 2.0)
        @test_throws ArgumentError statistical_filter_indices(coords, 10, -1.0)
        @test_throws ArgumentError statistical_filter_indices(rand(10, 2), 10, 2.0)
    end
    
    @testset "Grid z-min filter" begin
        coords = [
            0.1 0.1 2.0;
            0.2 0.3 1.0;
            1.2 0.2 5.0;
            1.4 0.6 4.0;
            2.1 2.1 7.0;
        ]

        indices = grid_zmin_filter_indices(coords, 1.0)
        @test indices == [2, 4, 5]

        # Deterministic tie-breaking: smallest index for equal z within cell
        tie_coords = [
            0.1 0.1 1.0;
            0.2 0.2 1.0;
            1.2 1.2 0.5;
        ]
        tie_indices = grid_zmin_filter_indices(tie_coords, 1.0)
        @test tie_indices == [1, 3]

        @test isempty(grid_zmin_filter_indices(zeros(0, 3), 1.0))
        @test_throws ArgumentError grid_zmin_filter_indices(coords, 0.0)
        @test_throws ArgumentError grid_zmin_filter_indices(coords, -1.0)
        @test_throws ArgumentError grid_zmin_filter_indices(rand(10, 2), 1.0)
    end

    @testset "Upward conic filter" begin
        coords = [
             0.0  0.0  0.0;   # kept anchor
             0.2  0.0  1.0;   # removed by point 1 (inside cone)
             2.0  0.0  1.0;   # kept (outside cone of point 1)
             0.0  0.0  2.0;   # removed by point 1
             2.0  0.0  2.0;   # removed by point 3
             5.0  5.0 -1.0;   # kept (lowest, far away)
        ]

        indices = upward_conic_filter_indices(coords, 45.0)
        @test indices == [1, 3, 6]

        # Equal-z points should not suppress each other (requires Δz > 0)
        flat_coords = [
            0.0 0.0 1.0;
            0.0 0.0 1.0;
        ]
        @test upward_conic_filter_indices(flat_coords, 45.0) == [1, 2]

        @test isempty(upward_conic_filter_indices(zeros(0, 3), 45.0))
        @test_throws ArgumentError upward_conic_filter_indices(coords, 0.0)
        @test_throws ArgumentError upward_conic_filter_indices(coords, 90.0)
        @test_throws ArgumentError upward_conic_filter_indices(coords, -10.0)
        @test_throws ArgumentError upward_conic_filter_indices(coords, 45.0, max_search_delta_z=0.0)
        @test_throws ArgumentError upward_conic_filter_indices(rand(10, 2), 45.0)

        # Maximum search radius cap: with default max_search_delta_z=5 and theta=45,
        # max_search_radius is 5 m. A point 10 m away in XY should not be suppressed.
        capped_coords = [
            0.0  0.0  0.0;
            10.0 0.0 20.0;
        ]
        @test upward_conic_filter_indices(capped_coords, 45.0) == [1, 2]
        @test upward_conic_filter_indices(capped_coords, 45.0, max_search_delta_z=20.0) == [1]

        # Spatial bins must still find suppressors near bin boundaries.
        boundary_coords = [
            0.0 0.0 0.0;
            4.9 0.0 5.0;
            0.0 4.9 5.0;
        ]
        @test upward_conic_filter_indices(boundary_coords, 45.0, max_search_delta_z=5.0) == [1]
    end

    @testset "RNN (radius neighbor count) filter" begin
        # Create a sparse cluster with varying density
        # Center cluster: 5 points within 0.5m
        center_points = [
            0.0  0.0  0.0;
            0.2  0.0  0.0;
            0.1  0.1  0.0;
            0.15 0.2  0.0;
            0.3  0.1  0.0;
        ]
        # Outliers: isolated points far away
        outliers = [
            10.0  0.0  0.0;
            20.0  0.0  0.0;
        ]
        coords = vcat(center_points, outliers)

        # With radius 0.5, center cluster has 5 points each (including self)
        indices = rnn_filter_indices(coords, 0.5, min_rnn_size=1)
        @test length(indices) == size(coords, 1)  # All have at least 1 neighbor
        
        # Keep only points with at least 3 neighbors within 0.5m
        indices = rnn_filter_indices(coords, 0.5, min_rnn_size=3)
        @test indices == [1, 2, 3, 4, 5]  # Only center cluster qualifies
        
        # Keep only points with at least 5 neighbors within 0.5m
        indices = rnn_filter_indices(coords, 0.5, min_rnn_size=5)
        @test indices == [1, 2, 3, 4, 5]  # All have exactly 5 neighbors
        
        # Keep only points with at least 6 neighbors
        indices = rnn_filter_indices(coords, 0.5, min_rnn_size=6)
        @test isempty(indices)  # No points have 6 neighbors

        # Empty input
        @test isempty(rnn_filter_indices(zeros(0, 3), 0.5))
        
        # Error handling
        @test_throws ArgumentError rnn_filter_indices(coords, 0.0)
        @test_throws ArgumentError rnn_filter_indices(coords, -0.5)
        @test_throws ArgumentError rnn_filter_indices(coords, 0.5, min_rnn_size=0)
        @test_throws ArgumentError rnn_filter_indices(rand(10, 2), 0.5)
    end
    
    @testset "Filtering preserves attributes" begin
        coords = randn(Float64, 100, 3)
        test_attr = rand(Float64, 100)
        pc = make_test_pointcloud(coords; attrs=Dict(:test_attr => test_attr))
        
        # Statistical filter
        pc_stat = statistical_filter(pc, 10, 2.0)
        @test hasattribute(pc_stat, :test_attr)
        @test length(getattribute(pc_stat, :test_attr)) == length(pc_stat)
    end

    @testset "segment_ground" begin
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
        test_attr = rand(Float64, size(coords, 1))
        pc = make_test_pointcloud(coords; attrs=Dict(:test_attr => test_attr))

        pc_segmented = segment_ground(pc; voxel_size=0.5, min_cc_size=20, grid_size=0.5, cone_theta_deg=60.0)
        @test length(pc_segmented) > 0
        @test length(pc_segmented) < length(pc)

        pc_explicit = segment_ground(pc; voxel_size=0.5, min_cc_size=20, grid_size=0.5, cone_theta_deg=60.0)
        @test length(pc_segmented) == length(pc_explicit)
        @test coordinates(pc_segmented) == coordinates(pc_explicit)

        agh = calculate_aboveground_height(pc, pc_segmented; xy_resolution=0.5)
        @test length(agh) == length(pc)
        @test any(isfinite, agh)

        @test_throws ArgumentError calculate_aboveground_height(pc, pc_segmented; xy_resolution=0.0)
        @test_throws ArgumentError calculate_aboveground_height(pc, pc_segmented; xy_resolution=0.5, idw_k=0)
        @test_throws ArgumentError calculate_aboveground_height(pc, pc_segmented; xy_resolution=0.5, idw_power=0.0)
    end

    @testset "Convex hull 2D" begin
        # Square points
        pts = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0; 0.5 0.5]
        hull = convex_hull_2d(pts)
        @test size(hull, 2) == 2
        @test size(hull, 1) == 4  # interior point excluded

        # Works with N×3 input
        pts3d = [pts[:, 1] pts[:, 2] zeros(5)]
        hull3d = convex_hull_2d(pts3d)
        @test size(hull3d, 1) == 4

        # Too few points
        @test_throws ArgumentError convex_hull_2d(rand(2, 2))
    end

    @testset "Buffer polygon" begin
        # Unit square CCW
        poly = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
        buffered = buffer_polygon(poly, 1.0)
        @test size(buffered) == size(poly)

        # Buffered polygon should be strictly larger in all directions
        @test minimum(buffered[:, 1]) < minimum(poly[:, 1])
        @test maximum(buffered[:, 1]) > maximum(poly[:, 1])
        @test minimum(buffered[:, 2]) < minimum(poly[:, 2])
        @test maximum(buffered[:, 2]) > maximum(poly[:, 2])

        # Error cases
        @test_throws ArgumentError buffer_polygon(poly, -1.0)
        @test_throws ArgumentError buffer_polygon(rand(2, 2), 1.0)
    end

    @testset "XY polygon filter" begin
        # Square polygon [0,1] × [0,1]
        poly = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]

        # Points: inside, outside, and borderline
        pts = [0.5 0.5 0.0;   # inside
               2.0 2.0 0.0;   # outside
               0.5 0.5 10.0;  # inside (z irrelevant)
               -1.0 0.5 0.0]  # outside
        idx = XY_polygon_filter_indices(pts, poly)
        @test 1 in idx
        @test 3 in idx
        @test !(2 in idx)
        @test !(4 in idx)

        # PointCloud wrapper
        pc = make_test_pointcloud(pts)
        pc_filtered = XY_polygon_filter(pc, poly)
        @test npoints(pc_filtered) == 2

        # Empty input
        @test isempty(XY_polygon_filter_indices(zeros(0, 3), poly))

        # Error cases
        @test_throws ArgumentError XY_polygon_filter_indices(rand(5, 1), poly)
        @test_throws ArgumentError XY_polygon_filter_indices(rand(5, 3), rand(2, 2))
    end

    @testset "Polygon area" begin
        # Unit square
        poly = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
        @test polygon_area(poly) ≈ 1.0
        # Triangle
        tri = [0.0 0.0; 4.0 0.0; 0.0 3.0]
        @test polygon_area(tri) ≈ 6.0
        @test_throws ArgumentError polygon_area(rand(2, 2))
    end

    @testset "crop_by_ground_polygon" begin
        # Ground points in a small square region
        n_gnd = 200
        gnd_coords = hcat(rand(n_gnd) .* 10.0, rand(n_gnd) .* 10.0, zeros(n_gnd))
        ground = make_test_pointcloud(gnd_coords)

        # Full cloud: ground region + far-away outlier points
        outlier_coords = [50.0 50.0 5.0; 60.0 60.0 3.0; -40.0 -40.0 2.0]
        all_coords = vcat(gnd_coords, outlier_coords)
        pc = make_test_pointcloud(all_coords)

        result = crop_by_ground_polygon(pc, ground; buffer=2.0)
        @test npoints(result.pc_cropped) < npoints(pc)
        @test npoints(result.pc_cropped) >= n_gnd
        @test result.ground_area > 0.0
        # Buffered [0,10]×[0,10] should be roughly (10+4)×(10+4) = 196 m²
        @test result.ground_area > 100.0
    end
end
