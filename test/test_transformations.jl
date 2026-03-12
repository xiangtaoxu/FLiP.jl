@testset "Transformations" begin
    @testset "Translation" begin
        coords = [
            0.0 0.0 0.0;
            1.0 0.0 0.0;
            0.0 1.0 0.0;
            0.0 0.0 1.0;
        ]
        pc = make_test_pointcloud(coords)
        
        # Translate by (1, 2, 3)
        pc_trans = translate(pc, 1.0, 2.0, 3.0)
        expected = coords .+ [1.0 2.0 3.0]
        @test coordinates(pc_trans) ≈ expected
        
        # Original should be unchanged
        @test coordinates(pc) == coords
        
        # In-place translation
        pc2 = make_test_pointcloud(copy(coords))
        pc2 = translate!(pc2, 1.0, 2.0, 3.0)
        @test coordinates(pc2) ≈ expected
        
        # Attributes should be preserved
        pc3 = make_test_pointcloud(coords; attrs=Dict(:test_attr => [1.0, 2.0, 3.0, 4.0]))
        pc3_trans = translate(pc3, 1.0, 2.0, 3.0)
        @test hasattribute(pc3_trans, :test_attr)
        @test getattribute(pc3_trans, :test_attr) == [1.0, 2.0, 3.0, 4.0]
    end
    
    @testset "Scaling" begin
        coords = [
            1.0 2.0 3.0;
            2.0 4.0 6.0;
        ]
        pc = make_test_pointcloud(coords)
        
        # Uniform scaling
        pc_scaled = scale(pc, 2.0)
        @test coordinates(pc_scaled) ≈ coords .* 2.0
        
        # Non-uniform scaling
        pc_scaled2 = scale(pc, 2.0, 3.0, 0.5)
        expected = coords .* [2.0 3.0 0.5]
        @test coordinates(pc_scaled2) ≈ expected
        
        # Error on invalid scale factor
        @test_throws ArgumentError scale(pc, 0.0)
        @test_throws ArgumentError scale(pc, -1.0)
        @test_throws ArgumentError scale(pc, 1.0, 0.0, 1.0)
    end
    
    @testset "Rotation" begin
        coords = [
            1.0 0.0 0.0;
            0.0 1.0 0.0;
            0.0 0.0 1.0;
        ]
        pc = make_test_pointcloud(coords)
        
        # Rotate 90 degrees around Z axis
        pc_rot = rotate(pc, :z, π/2)
        expected = [
            0.0 1.0 0.0;
            -1.0 0.0 0.0;
            0.0 0.0 1.0;
        ]
        @test coordinates(pc_rot)[:, 1] ≈ expected[:, 1] atol=1e-10
        @test coordinates(pc_rot)[:, 2] ≈ expected[:, 2] atol=1e-10
        @test coordinates(pc_rot)[:, 3] ≈ expected[:, 3] atol=1e-10
        
        # Rotate around X axis
        pc_rot_x = rotate(pc, :x, π/2)
        @test coordinates(pc_rot_x)[1, :] ≈ [1.0, 0.0, 0.0] atol=1e-10
        @test coordinates(pc_rot_x)[2, :] ≈ [0.0, 0.0, 1.0] atol=1e-10
        
        # Rotate around custom axis
        pc_rot_custom = rotate(pc, [0, 0, 1], π/2)
        @test coordinates(pc_rot_custom)[:, 1] ≈ expected[:, 1] atol=1e-10
        
        # Error on invalid axis
        @test_throws ArgumentError rotate(pc, :w, π/2)
        @test_throws ArgumentError rotate(pc, [1, 2], π/2)
    end
    
    @testset "Center at origin" begin
        coords = [
            1.0 1.0 1.0;
            2.0 2.0 2.0;
            3.0 3.0 3.0;
        ]
        pc = make_test_pointcloud(coords)
        
        pc_centered = center_at_origin(pc)
        c = center(pc_centered)
        @test all(abs.(c) .< 1e-10)
        
        # Check relative positions preserved
        pc_center_orig = center(pc)
        for i in 1:3
            offset = coordinates(pc)[i, :] .- pc_center_orig
            new_pos = coordinates(pc_centered)[i, :]
            @test new_pos ≈ offset atol=1e-10
        end
    end
    
    @testset "Arbitrary transformation" begin
        using CoordinateTransformations, Rotations
        
        coords = [
            0.0 0.0 0.0;
            1.0 0.0 0.0;
            0.0 1.0 0.0;
        ]
        pc = make_test_pointcloud(coords)
        
        # Compose translation and rotation
        tfm = Translation(1, 2, 3) ∘ LinearMap(RotZ(π/2))
        pc_tfm = transform(pc, tfm)
        
        @test size(coordinates(pc_tfm)) == size(coords)
        # First point should be at (1, 2, 3)
        @test coordinates(pc_tfm)[1, :] ≈ [1.0, 2.0, 3.0] atol=1e-10
        
        # Test apply_transform on raw coordinates
        coords_tfm = apply_transform(coords, tfm)
        @test coords_tfm ≈ coordinates(pc_tfm) atol=1e-10
        
        # Test in-place transform
        coords_copy = copy(coords)
        apply_transform!(coords_copy, Translation(1, 0, 0))
        @test coords_copy[:, 1] ≈ coords[:, 1] .+ 1.0
    end
    
    @testset "Bounding box crop" begin
        coords = rand(Float64, 100, 3) .* 10.0  # Points in [0, 10]^3
        pc = make_test_pointcloud(coords)
        
        # Crop to [2, 8] in all dimensions
        pc_cropped = bounding_box_crop(pc, [2, 2, 2], [8, 8, 8])
        
        @test length(pc_cropped) < length(pc)
        crop_coords = coordinates(pc_cropped)
        @test all(2.0 .<= crop_coords[:, 1] .<= 8.0)
        @test all(2.0 .<= crop_coords[:, 2] .<= 8.0)
        @test all(2.0 .<= crop_coords[:, 3] .<= 8.0)
        
        # Error on invalid dimensions
        @test_throws ArgumentError bounding_box_crop(pc, [0, 0], [10, 10])
        @test_throws ArgumentError bounding_box_crop(pc, [0, 0, 0], [10, 10])
    end
    
    @testset "Transformations preserve attributes" begin
        coords = rand(Float64, 50, 3)
        test_attr = rand(Float64, 50)
        pc = make_test_pointcloud(coords; attrs=Dict(:test_attr => test_attr))
        
        # Translation
        pc_trans = translate(pc, 1.0, 2.0, 3.0)
        @test hasattribute(pc_trans, :test_attr)
        @test getattribute(pc_trans, :test_attr) == test_attr
        
        # Rotation
        pc_rot = rotate(pc, :z, π/4)
        @test hasattribute(pc_rot, :test_attr)
        @test getattribute(pc_rot, :test_attr) == test_attr
        
        # Scaling
        pc_scaled = scale(pc, 2.0)
        @test hasattribute(pc_scaled, :test_attr)
        @test getattribute(pc_scaled, :test_attr) == test_attr
    end
end
