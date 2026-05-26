@testset "Geometry" begin

    @testset "pca_linearity — linear point set along z" begin
        # Points along z-axis with tiny x/y jitter → high linearity, direction ≈ ±z
        n = 100
        zs = collect(range(0.0, 1.0; length=n))
        coords = hcat(0.001 .* (zs .- 0.5),     # tiny linear x
                      0.001 .* (zs .- 0.5),     # tiny linear y
                      zs)
        result = FLiP.pca_linearity(coords, 1:n, 0.5)
        @test result !== nothing
        @test result.linearity > 0.99
        @test abs(result.direction[3]) > 0.99    # PC1 is essentially z
        @test result.eigenvalues[1] <= result.eigenvalues[2] <= result.eigenvalues[3]
        # Centroid should be midpoint
        @test result.center[3] ≈ 0.5 atol=1e-10
    end

    @testset "pca_linearity — spherical cloud returns nothing" begin
        # Roughly uniform 3D distribution → linearity << 0.5
        coords = [0.0 0.0 0.0;
                  1.0 0.0 0.0;
                  -1.0 0.0 0.0;
                  0.0 1.0 0.0;
                  0.0 -1.0 0.0;
                  0.0 0.0 1.0;
                  0.0 0.0 -1.0]
        result = FLiP.pca_linearity(coords, 1:7, 0.5)
        @test result === nothing
    end

    @testset "pca_linearity — too few points returns nothing" begin
        coords = [0.0 0.0 0.0; 1.0 0.0 0.0]
        result = FLiP.pca_linearity(coords, 1:2, 0.5)
        @test result === nothing
    end

    @testset "_build_perpendicular_basis — orthonormal to direction" begin
        # Axis-aligned input
        d = (0.0, 0.0, 1.0)
        e1, e2 = FLiP._build_perpendicular_basis(d)
        @test abs(e1[1]*d[1] + e1[2]*d[2] + e1[3]*d[3]) < 1e-12
        @test abs(e2[1]*d[1] + e2[2]*d[2] + e2[3]*d[3]) < 1e-12
        @test abs(e1[1]*e2[1] + e1[2]*e2[2] + e1[3]*e2[3]) < 1e-12
        @test abs(sqrt(e1[1]^2 + e1[2]^2 + e1[3]^2) - 1.0) < 1e-12
        @test abs(sqrt(e2[1]^2 + e2[2]^2 + e2[3]^2) - 1.0) < 1e-12

        # Oblique input — normalize first
        n = sqrt(1.0 + 4.0 + 9.0)
        d2 = (1.0/n, 2.0/n, 3.0/n)
        e1b, e2b = FLiP._build_perpendicular_basis(d2)
        @test abs(e1b[1]*d2[1] + e1b[2]*d2[2] + e1b[3]*d2[3]) < 1e-12
        @test abs(e2b[1]*d2[1] + e2b[2]*d2[2] + e2b[3]*d2[3]) < 1e-12
        @test abs(e1b[1]*e2b[1] + e1b[2]*e2b[2] + e1b[3]*e2b[3]) < 1e-12
        @test abs(sqrt(e1b[1]^2 + e1b[2]^2 + e1b[3]^2) - 1.0) < 1e-12
        @test abs(sqrt(e2b[1]^2 + e2b[2]^2 + e2b[3]^2) - 1.0) < 1e-12
    end

end
