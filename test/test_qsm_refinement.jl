@testset "QSM refinement" begin

    # Build a QSMNode with just the fields the refinement model reads.
    mknode(; id, seg, tree, agh, h, comp, npts, cx, cy, cz, dx=0.0, dy=0.0, dz=1.0, r) =
        FLiP.QSMNode(id, Int32(seg), Int32(tree), Int32(seg),
                     agh, h, comp, npts, cx, cy, cz, dx, dy, dz,
                     π * r^2, 2π * r, r, r)

    # Cylinder element compatible with the geometry primitives.
    mkcyl(cx, cy, cz, r, hh; ax=(0.0, 0.0, 1.0)) =
        (center=(cx, cy, cz), axis=ax, radius=r, half_height=hh)

    # ── point_in_cylinder ──────────────────────────────────────────────────
    @testset "point_in_cylinder" begin
        c = (0.0, 0.0, 0.0); ax = (0.0, 0.0, 1.0); r = 0.5; hh = 1.0
        @test FLiP.point_in_cylinder((0.0, 0.0, 0.0), c, ax, r, hh)         # center
        @test FLiP.point_in_cylinder((0.4, 0.0, 0.5), c, ax, r, hh)         # inside
        @test FLiP.point_in_cylinder((0.5, 0.0, 0.0), c, ax, r, hh)         # on radial boundary
        @test !FLiP.point_in_cylinder((0.6, 0.0, 0.0), c, ax, r, hh)        # outside radius
        @test !FLiP.point_in_cylinder((0.0, 0.0, 1.5), c, ax, r, hh)        # beyond half_height
        @test !FLiP.point_in_cylinder((0.0, 0.0, 0.0), c, ax, 0.0, hh)      # radius 0
        @test !FLiP.point_in_cylinder((0.0, 0.0, 0.0), c, ax, NaN, hh)      # non-finite
    end

    # ── cylinder_aabb ──────────────────────────────────────────────────────
    @testset "cylinder_aabb" begin
        bb = FLiP.cylinder_aabb((1.0, 2.0, 3.0), (0.0, 0.0, 1.0), 0.5, 1.0)
        @test all(isapprox.(bb, (0.5, 1.5, 1.5, 2.5, 2.0, 4.0); atol=1e-12))

        # 45° tilt in x-z: half-extent = hh*|d|+r*sqrt(1-d^2)
        s = 1 / sqrt(2)
        bb2 = FLiP.cylinder_aabb((0.0, 0.0, 0.0), (s, 0.0, s), 0.5, 1.0)
        ex = 1.0 * s + 0.5 * sqrt(1 - s^2)
        @test isapprox(bb2[2], ex; atol=1e-10)   # xmax
        @test isapprox(bb2[6], ex; atol=1e-10)   # zmax
        @test isapprox(bb2[4], 0.5; atol=1e-10)  # ymax (axis has no y component)
    end

    @testset "aabbs_overlap" begin
        a = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        @test FLiP.aabbs_overlap(a, (0.5, 2.0, 0.5, 2.0, 0.5, 2.0))
        @test !FLiP.aabbs_overlap(a, (2.0, 3.0, 0.0, 1.0, 0.0, 1.0))
    end

    # ── voxelized_cylinder_volume ──────────────────────────────────────────
    @testset "voxelized_cylinder_volume ≈ π r² h" begin
        r = 0.3; hh = 0.5; v = 0.02
        cyl = mkcyl(0.0, 0.0, 0.0, r, hh)
        box = FLiP.cylinder_aabb(cyl.center, cyl.axis, cyl.radius, cyl.half_height)
        vol = FLiP.voxelized_cylinder_volume([cyl], box, v)
        analytic = π * r^2 * (2hh)
        @test isapprox(vol, analytic; rtol=0.03)
        @test FLiP.voxelized_cylinder_volume(typeof(cyl)[], box, v) == 0.0
    end

    # ── intersection / overlap ratio (analytic) ────────────────────────────
    @testset "overlap ratio" begin
        r = 0.3; hh = 0.5; v = 0.02
        A = mkcyl(0.0, 0.0, 0.0, r, hh)
        bbA = FLiP.cylinder_aabb(A.center, A.axis, A.radius, A.half_height)
        selfvol = FLiP.voxelized_cylinder_volume([A], bbA, v)

        # identical coaxial → full overlap
        iv_same = FLiP._voxel_intersection_volume([A], [A], bbA, v)
        @test isapprox(iv_same / selfvol, 1.0; atol=1e-9)

        # coaxial, axially offset by half_height → ~half overlap
        B = mkcyl(0.0, 0.0, hh, r, hh)
        bbB = FLiP.cylinder_aabb(B.center, B.axis, B.radius, B.half_height)
        box = FLiP._aabb_intersection(bbA, bbB)
        iv_half = FLiP._voxel_intersection_volume([A], [B], box, v)
        @test isapprox(iv_half / selfvol, 0.5; rtol=0.05)

        # far apart → no overlap
        C = mkcyl(0.0, 0.0, 10.0, r, hh)
        bbC = FLiP.cylinder_aabb(C.center, C.axis, C.radius, C.half_height)
        @test !FLiP.aabbs_overlap(bbA, bbC)
    end

    # Two overlapping single-cylinder segments in the same tree → should merge.
    function two_overlap_setup(; comp=0.9, npts=100, dx_off=0.1, tree2=1)
        nodes = [
            mknode(id=1, seg=1, tree=1,     agh=5.0, h=1.0, comp=comp, npts=npts, cx=0.0,    cy=0.0, cz=0.0, r=0.3),
            mknode(id=2, seg=2, tree=tree2, agh=5.0, h=1.0, comp=comp, npts=npts, cx=dx_off, cy=0.0, cz=0.0, r=0.3),
        ]
        coords = [0.0 0.0 0.0; 0.0 0.0 0.1; 0.1 0.0 0.0; 0.1 0.0 0.1]
        pc = make_test_pointcloud(coords; attrs=Dict(
            :tree_nbs_id => Int32[1, 1, 2, 2],
            :tree_id     => Int32[1, 1, tree2, tree2],
        ))
        return nodes, pc
    end

    base_cfg() = begin
        cfg = FLiP.FLiPConfig(Dict{String,Any}())
        cfg.pipeline.subsample_res = 0.05
        cfg
    end

    @testset "basic within-tree merge" begin
        nodes, pc = two_overlap_setup()
        cfg = base_cfg()
        res = FLiP.nbs_merge_by_volume_overlap(pc=pc, nodes=nodes, cfg=cfg)
        @test res.status == :success
        @test res.n_segments_in == 2
        @test res.n_segments_out == 1
        @test res.n_groups_merged == 1
        @test res.merged
        tnbs = getattribute(pc, :tree_nbs_id)
        @test length(unique(tnbs)) == 1          # both segments now share one id
    end

    @testset "completeness gate blocks merge" begin
        nodes, pc = two_overlap_setup(comp=0.9)
        cfg = base_cfg(); cfg.qsm_refinement.completeness_gate = 0.95
        res = FLiP.nbs_merge_by_volume_overlap(pc=pc, nodes=nodes, cfg=cfg)
        @test !res.merged
        @test res.n_groups_merged == 0
    end

    @testset "min_points gate blocks merge" begin
        nodes, pc = two_overlap_setup(npts=100)
        cfg = base_cfg(); cfg.qsm_refinement.min_points_gate = 500
        res = FLiP.nbs_merge_by_volume_overlap(pc=pc, nodes=nodes, cfg=cfg)
        @test !res.merged
    end

    @testset "absorb guard blocks sparse-absorbs-dense" begin
        # Larger-volume (radius 0.6) segment is sparse (10 pts); smaller (0.3) is dense (100).
        nodes = [
            mknode(id=1, seg=1, tree=1, agh=5.0, h=1.0, comp=0.9, npts=100, cx=0.0, cy=0.0, cz=0.0, r=0.3),
            mknode(id=2, seg=2, tree=1, agh=5.0, h=1.0, comp=0.9, npts=30,  cx=0.1, cy=0.0, cz=0.0, r=0.6),
        ]
        coords = [0.0 0.0 0.0; 0.1 0.0 0.0]
        pc = make_test_pointcloud(coords; attrs=Dict(
            :tree_nbs_id => Int32[1, 2], :tree_id => Int32[1, 1]))
        cfg = base_cfg()  # absorber=seg2 (larger vol) has 30 pts < 0.5*100 → absorb_guard
        res = FLiP.nbs_merge_by_volume_overlap(pc=pc, nodes=nodes, cfg=cfg)
        @test !res.merged
    end

    @testset "union-find transitivity (A-B, B-C; A∩C tangent)" begin
        nodes = [
            mknode(id=1, seg=1, tree=1, agh=5.0, h=1.0, comp=0.9, npts=100, cx=0.0, cy=0.0, cz=0.0, r=0.3),
            mknode(id=2, seg=2, tree=1, agh=5.0, h=1.0, comp=0.9, npts=100, cx=0.3, cy=0.0, cz=0.0, r=0.3),
            mknode(id=3, seg=3, tree=1, agh=5.0, h=1.0, comp=0.9, npts=100, cx=0.6, cy=0.0, cz=0.0, r=0.3),
        ]
        coords = [0.0 0.0 0.0; 0.3 0.0 0.0; 0.6 0.0 0.0]
        pc = make_test_pointcloud(coords; attrs=Dict(
            :tree_nbs_id => Int32[1, 2, 3], :tree_id => Int32[1, 1, 1]))
        cfg = base_cfg()
        res = FLiP.nbs_merge_by_volume_overlap(pc=pc, nodes=nodes, cfg=cfg)
        @test res.merged
        @test res.n_segments_in == 3
        @test res.n_segments_out == 1       # all three collapse via B
        @test length(unique(getattribute(pc, :tree_nbs_id))) == 1
    end

    @testset "cross-tree merge toggle" begin
        # Same geometry, different trees.
        nodes, pc = two_overlap_setup(tree2=2)
        cfg = base_cfg(); cfg.qsm_refinement.cross_tree = true
        # Both segments are high above ground (agh 5.0) so protect_grounded_trunks is inactive.
        res = FLiP.nbs_merge_by_volume_overlap(pc=pc, nodes=nodes, cfg=cfg)
        @test res.merged
        @test res.n_cross_tree_merges == 1
        tid = getattribute(pc, :tree_id)
        tnbs = getattribute(pc, :tree_nbs_id)
        @test length(unique(tid)) == 1
        # No tree_nbs_id maps to two different tree_ids.
        m = Dict{Int32,Int32}()
        ok = true
        for i in eachindex(tnbs)
            t = tnbs[i]
            if haskey(m, t); ok &= (m[t] == tid[i]); else; m[t] = tid[i]; end
        end
        @test ok

        # Disabled → no merge.
        nodes2, pc2 = two_overlap_setup(tree2=2)
        cfg2 = base_cfg(); cfg2.qsm_refinement.cross_tree = false
        res2 = FLiP.nbs_merge_by_volume_overlap(pc=pc2, nodes=nodes2, cfg=cfg2)
        @test !res2.merged
        @test length(unique(getattribute(pc2, :tree_id))) == 2
    end

    @testset "flag_only writes report but does not relabel" begin
        nodes, pc = two_overlap_setup()
        cfg = base_cfg(); cfg.qsm_refinement.mode = "flag_only"
        tmp = mktempdir()
        res = FLiP.nbs_merge_by_volume_overlap(pc=pc, nodes=nodes, cfg=cfg,
                                               output_dir=tmp, output_prefix="t_")
        @test !res.merged                                  # cloud not relabeled
        @test length(unique(getattribute(pc, :tree_nbs_id))) == 2
        @test isfile(joinpath(tmp, "t_nbs_merge_report.csv"))
        rm(tmp; recursive=true, force=true)
    end

    # ── Integration: real QSM output → refinement merge ────────────────────
    @testset "integration: qsm() then merge two overlapping cylinders" begin
        function cyl_points(cx, cy, r, h, n, seg)
            θ = 2π .* rand(n)
            z = h .* rand(n)
            x = cx .+ r .* cos.(θ) .+ 0.002 .* randn(n)
            y = cy .+ r .* sin.(θ) .+ 0.002 .* randn(n)
            return hcat(x, y, z), fill(Int32(seg), n), z
        end
        n = 1500; r = 0.15; h = 0.6
        c1, s1, z1 = cyl_points(0.0, 0.0, r, h, n, 1)
        c2, s2, z2 = cyl_points(0.1, 0.0, r, h, n, 2)   # 0.1 apart → overlapping
        coords = vcat(c1, c2)
        seg = vcat(s1, s2)
        zz = vcat(z1, z2)
        pc = make_test_pointcloud(coords; attrs=Dict(
            :tree_nbs_id => seg,
            :tree_id     => ones(Int32, 2n),
            :AGH         => zz,
            :node_id     => ones(Int32, 2n),
            :nbs_id      => seg,
        ))
        tree_result = (pc_output=pc,
                       skeleton_cloud=FLiP.PointCloud(zeros(Float64, 0, 3), Dict{Symbol,Vector}()),
                       filtered_cloud=pc, n_components=1, neighbor_radius=0.1)

        cfg = FLiP._CFG
        old = cfg.pipeline.subsample_res
        cfg.pipeline.subsample_res = 0.03
        tmp = mktempdir()
        q = FLiP.qsm(tree_result=tree_result, config_path="", output_dir=tmp, output_prefix="i_")
        if q.status == :success && length(q.nodes) > 0
            res = FLiP.nbs_merge_by_volume_overlap(pc=pc, nodes=q.nodes, cfg=cfg,
                                                   output_dir=tmp, output_prefix="i_")
            @test res.status == :success
            @test isfile(joinpath(tmp, "i_nbs_merge_report.csv"))
            # The two overlapping cylinders should merge into one segment.
            if res.n_segments_in >= 2
                @test res.n_segments_out < res.n_segments_in
            end

            # CSV resume reconstructs the same node count.
            recovered = FLiP._read_qsm_nodes_csv(joinpath(tmp, "i_qsm_nodes.csv"))
            @test length(recovered) == length(q.nodes)
        end
        cfg.pipeline.subsample_res = old
        rm(tmp; recursive=true, force=true)
    end
end
