@testset "Mesh" begin
    @testset "delaunay_triangulation_xy" begin
        pts = [
            0.0 0.0 0.0;
            1.0 0.0 1.0;
            0.0 1.0 1.0;
            1.0 1.0 2.0;
        ]

        mesh = delaunay_triangulation_xy(pts)

        @test length(mesh.points_xy) == 4
        @test length(mesh.z) == 4
        @test !isempty(mesh.triangles)
        @test !isempty(mesh.triangles)

        @test_throws ArgumentError delaunay_triangulation_xy(rand(10, 2))
        @test_throws ArgumentError delaunay_triangulation_xy(rand(2, 3))
    end

    @testset "cloud_to_mesh_distance_z" begin
        # Plane: z = x + y over unit square corners
        ref = [
            0.0 0.0 0.0;
            1.0 0.0 1.0;
            0.0 1.0 1.0;
            1.0 1.0 2.0;
        ]
        mesh = delaunay_triangulation_xy(ref)

        queries = [
            0.25 0.25 0.50;  # on plane
            0.75 0.25 1.00;  # on plane
            0.50 0.25 2.00;  # above plane (plane z=0.75 -> +1.25)
            0.50 0.25 0.00;  # below plane (-> -0.75)
            2.00 2.00 3.00;  # outside hull -> NaN
        ]

        dz = cloud_to_mesh_distance_z(queries, mesh)

        @test isapprox(dz[1], 0.0; atol=1e-10)
        @test isapprox(dz[2], 0.0; atol=1e-10)
        @test isapprox(dz[3], 1.25; atol=1e-10)
        @test isapprox(dz[4], -0.75; atol=1e-10)
        @test isnan(dz[5])

        @test_throws ArgumentError cloud_to_mesh_distance_z(rand(10, 2), mesh)
    end

    @testset "sample_mesh_xy" begin
        ref = [
            0.0 0.0 0.0;
            1.0 0.0 1.0;
            0.0 1.0 1.0;
            1.0 1.0 2.0;
        ]
        mesh = delaunay_triangulation_xy(ref)

        sampled = FLiP.sample_mesh_xy(mesh, 0.5)
        @test size(sampled, 2) == 3
        @test size(sampled, 1) >= 4

        # On this plane z = x + y; sampled z should match closely
        @test maximum(abs.(sampled[:, 3] .- (sampled[:, 1] .+ sampled[:, 2]))) < 1e-8

        @test_throws ArgumentError FLiP.sample_mesh_xy(mesh, 0.0)
        @test_throws ArgumentError FLiP.sample_mesh_xy(mesh, -0.1)
    end
end
