@testset "Subsampling" begin
    @testset "Minimum distance subsampling" begin
        # Create test data where some points are too close
        coords = [
            0.0 0.0 0.0;
            0.01 0.0 0.0;    # Too close to first
            0.5 0.0 0.0;     # Far enough
            0.51 0.0 0.0;    # Too close to previous
        ]
        
        # With min distance 0.1, should keep points 1 and 3
        indices = distance_subsample_indices(coords, 0.1)
        @test length(indices) == 2
        @test 1 in indices
        @test 3 in indices
        
        # Test with PointCloud
        pc = make_test_pointcloud(coords)
        pc_sub = distance_subsample(pc, 0.1)
        @test length(pc_sub) == 2
        
        # Test that kept points maintain minimum distance
        sub_coords = coordinates(pc_sub)
        for i in 1:size(sub_coords, 1)
            for j in (i+1):size(sub_coords, 1)
                dist = norm(sub_coords[i, :] - sub_coords[j, :])
                @test dist >= 0.1
            end
        end
        
        # Test error handling
        @test_throws ArgumentError distance_subsample_indices(coords, 0.0)
        @test_throws ArgumentError distance_subsample_indices(coords, -1.0)
        @test_throws ArgumentError distance_subsample_indices(rand(10, 2), 0.1)
    end
    
    @testset "Subsampling preserves attributes" begin
        coords = rand(Float64, 100, 3)
        test_attr = rand(Float64, 100)
        pc = make_test_pointcloud(coords; attrs=Dict(:test_attr => test_attr))
        
        # Distance
        pc_dist = distance_subsample(pc, 0.1)
        @test hasattribute(pc_dist, :test_attr)
        @test length(getattribute(pc_dist, :test_attr)) == length(pc_dist)
    end
end
