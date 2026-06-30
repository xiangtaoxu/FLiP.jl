@testset "refine_nbs" begin

    # ── helpers ────────────────────────────────────────────────────────────────
    # A trial-QSM node: a finite cylinder (vertical by default) at (cx,cy,cz).
    mknode(id, nbs, cx, cy, cz; r=0.10, h=0.30, comp=0.9, npts=30, agh=1.0,
           dir=(0.0, 0.0, 1.0)) =
        FLiP.QSMNode(id, Int32(nbs), Int32(0), Int32(nbs), agh, h, comp, npts,
                     Float64(cx), Float64(cy), Float64(cz),
                     dir[1], dir[2], dir[3], π*r^2, 2π*r, r, r)

    function base_cfg(; mode="apply", merge_threshold=0.8, overlap=0.2,
                        comp_gate=0.25, min_pts=1)
        cfg = FLiP.FLiPConfig(Dict{String,Any}())
        cfg.pipeline.subsample_res                = 0.05
        cfg.tree.refine.mode                       = mode
        cfg.tree.refine.overlap_threshold          = overlap
        cfg.tree.refine.completeness_gate          = comp_gate
        cfg.tree.refine.min_points_gate            = min_pts
        cfg.tree.refine.voxel_res_scalar           = 1.0
        cfg.tree.refine.candidate_radius_scalar    = 1.0
        cfg.tree.assembly.merge_threshold = merge_threshold
        return cfg
    end

    sg(n, edges) = (g = FLiP.SimpleGraph(n); for (u, v) in edges; FLiP.add_edge!(g, u, v); end; g)

    # ── geometry primitives still used by refine_nbs ───────────────────────────
    @testset "geometry primitives" begin
        cyl = (center=(0.0,0.0,0.0), axis=(0.0,0.0,1.0), radius=0.2, half_height=0.5)
        # point_in_cylinder: inside, outside-radially, outside-axially
        @test FLiP.point_in_cylinder((0.1,0.0,0.3), cyl.center, cyl.axis, cyl.radius, cyl.half_height)
        @test !FLiP.point_in_cylinder((0.3,0.0,0.0), cyl.center, cyl.axis, cyl.radius, cyl.half_height)
        @test !FLiP.point_in_cylinder((0.0,0.0,0.9), cyl.center, cyl.axis, cyl.radius, cyl.half_height)
        bb  = FLiP.cylinder_aabb(cyl.center, cyl.axis, cyl.radius, cyl.half_height)
        # aabbs_overlap: overlapping vs disjoint
        @test FLiP.aabbs_overlap(bb, FLiP.cylinder_aabb((0.1,0.0,0.0), cyl.axis, 0.2, 0.5))
        @test !FLiP.aabbs_overlap(bb, FLiP.cylinder_aabb((5.0,0.0,0.0), cyl.axis, 0.2, 0.5))
        v   = FLiP.voxelized_cylinder_volume([cyl], bb, 0.05)
        @test v > 0
        @test abs(v - π*0.2^2*1.0) / (π*0.2^2*1.0) < 0.2   # ≈ analytic cylinder volume
        # self-intersection equals self-volume on the shared lattice
        iv = FLiP._voxel_intersection_volume([cyl], [cyl], bb, 0.05)
        @test iv ≈ v
    end

    # ── Step 1: Rule B (whole-NBS connectivity merge by volume) ────────────────
    @testset "Rule B: collinear NBS merges into larger-volume neighbor" begin
        nodes = [mknode(10, 1, 0.0,0.0,0.0; r=0.15), mknode(20, 2, 0.0,0.0,0.6; r=0.07)]
        res = FLiP.refine_nbs(nbs_id=Int32[1,1,2,2], node_id=Int32[1,1,2,2],
                              trial_node_id=Int32[10,10,20,20], nodes=nodes,
                              graph_skeleton=sg(2, [(1,2)]), skel_to_nbs=Int32[1,2],
                              cfg=base_cfg())
        @test res.n_rule_b_merges == 1
        @test res.nbs_id == Int32[1,1,1,1]        # smaller-id, larger-volume nbs 1 absorbs
    end

    @testset "Rule B: transitive chain A→B→C collapses to the largest" begin
        nodes = [mknode(10,1, 0,0,0.0; r=0.06), mknode(20,2, 0,0,0.6; r=0.10),
                 mknode(30,3, 0,0,1.2; r=0.15)]
        res = FLiP.refine_nbs(nbs_id=Int32[1,1,2,2,3,3], node_id=Int32[1,1,2,2,3,3],
                              trial_node_id=Int32[10,10,20,20,30,30], nodes=nodes,
                              graph_skeleton=sg(3, [(1,2),(2,3)]), skel_to_nbs=Int32[1,2,3],
                              cfg=base_cfg())
        @test res.n_rule_b_merges == 2
        @test res.nbs_id == Int32[3,3,3,3,3,3]    # everything folds into C (largest volume)
    end

    @testset "Rule B: equal-volume tie → smaller nbs id is the receiver" begin
        nodes = [mknode(10,1, 0,0,0.0; r=0.10), mknode(20,2, 0,0,0.6; r=0.10)]
        res = FLiP.refine_nbs(nbs_id=Int32[1,1,2,2], node_id=Int32[1,1,2,2],
                              trial_node_id=Int32[10,10,20,20], nodes=nodes,
                              graph_skeleton=sg(2, [(1,2)]), skel_to_nbs=Int32[1,2],
                              cfg=base_cfg())
        @test res.nbs_id == Int32[1,1,1,1]
    end

    @testset "Rule B: zero-volume (non-linear) NBS merges via connectivity" begin
        # nbs 2 produced no trial cylinder (volume 0) but is skeleton-adjacent to nbs 1.
        nodes = [mknode(10,1, 0,0,0.0; r=0.12)]
        res = FLiP.refine_nbs(nbs_id=Int32[1,1,2,2], node_id=Int32[1,1,2,2],
                              trial_node_id=Int32[10,10,0,0], nodes=nodes,
                              graph_skeleton=sg(2, [(1,2)]), skel_to_nbs=Int32[1,2],
                              cfg=base_cfg())
        @test res.n_rule_b_merges == 1
        @test res.nbs_id == Int32[1,1,1,1]
    end

    @testset "Rule B: below threshold does not merge" begin
        # nbs 2 has two skeleton nodes; only one touches nbs 1 → frac 0.5 < 0.8.
        nodes = [mknode(10,1, 0,0,0.0; r=0.15),
                 mknode(20,2, 0,0,0.6; r=0.07), mknode(21,2, 0,0,1.2; r=0.07)]
        res = FLiP.refine_nbs(nbs_id=Int32[1,1,2,2,2,2], node_id=Int32[1,1,2,2,3,3],
                              trial_node_id=Int32[10,10,20,20,21,21], nodes=nodes,
                              graph_skeleton=sg(3, [(1,2),(2,3)]), skel_to_nbs=Int32[1,2,2],
                              cfg=base_cfg())
        @test res.n_rule_b_merges == 0
        @test sort(unique(res.nbs_id)) == Int32[1,2]
    end

    # ── Step 2: node-level volume overlap (no skeleton edges → no Rule B) ──────
    @testset "Volume merge: overlapping node is claimed by the larger NBS" begin
        nodes = [mknode(10,1, 0,0,0.0; r=0.25, h=1.0, npts=200),   # fat focal
                 mknode(20,2, 0,0,0.0; r=0.05, h=0.2, npts=20)]     # small, inside it
        res = FLiP.refine_nbs(nbs_id=Int32[1,1,2,2], node_id=Int32[1,1,2,2],
                              trial_node_id=Int32[10,10,20,20], nodes=nodes,
                              graph_skeleton=sg(2, Tuple{Int,Int}[]), skel_to_nbs=Int32[1,2],
                              cfg=base_cfg())
        @test res.n_rule_b_merges == 0
        @test res.n_nodes_moved == 1
        @test res.nbs_id == Int32[1,1,1,1]
    end

    @testset "Volume merge: non-overlapping node stays put" begin
        nodes = [mknode(10,1, 0,0,0.0; r=0.25, h=1.0, npts=200),
                 mknode(20,2, 3.0,0.0,0.0; r=0.05, h=0.2, npts=20)]   # far away
        res = FLiP.refine_nbs(nbs_id=Int32[1,1,2,2], node_id=Int32[1,1,2,2],
                              trial_node_id=Int32[10,10,20,20], nodes=nodes,
                              graph_skeleton=sg(2, Tuple{Int,Int}[]), skel_to_nbs=Int32[1,2],
                              cfg=base_cfg())
        @test res.n_nodes_moved == 0
        @test res.nbs_id == Int32[1,1,2,2]
    end

    # ── Fix #1: skeleton-node plurality snap ───────────────────────────────────
    @testset "Snap: a skeleton node spanning two trial slices never splits" begin
        # nbs 2 = ONE skeleton node (node_id 2) but TWO trial slices (20 overlaps the
        # focal, 21 is far). Moving only slice 20 would split skeleton node 2; the snap
        # forces its whole point set to a single nbs.
        nodes = [mknode(10,1, 0,0,0.0; r=0.30, h=1.2, npts=300),
                 mknode(20,2, 0,0,0.0; r=0.05, h=0.2, npts=20),
                 mknode(21,2, 3.0,0.0,0.0; r=0.05, h=0.2, npts=20)]
        node_id = Int32[1,1, 2,2, 2,2]      # pts 3-6 are all one skeleton node
        res = FLiP.refine_nbs(nbs_id=Int32[1,1,2,2,2,2], node_id=node_id,
                              trial_node_id=Int32[10,10,20,20,21,21], nodes=nodes,
                              graph_skeleton=sg(2, Tuple{Int,Int}[]), skel_to_nbs=Int32[1,2],
                              cfg=base_cfg())
        @test length(unique(res.nbs_id[node_id .== 2])) == 1   # not split
    end

    # ── determinism + flag_only ────────────────────────────────────────────────
    @testset "Determinism: identical inputs → identical labels" begin
        nodes = [mknode(10,1, 0,0,0.0; r=0.15), mknode(20,2, 0,0,0.6; r=0.07),
                 mknode(30,3, 0,0,0.0; r=0.05, h=0.2, npts=20)]
        mk() = FLiP.refine_nbs(nbs_id=Int32[1,1,2,2,3,3], node_id=Int32[1,1,2,2,3,3],
                               trial_node_id=Int32[10,10,20,20,30,30], nodes=nodes,
                               graph_skeleton=sg(3, [(1,2)]), skel_to_nbs=Int32[1,2,3],
                               cfg=base_cfg())
        @test mk().nbs_id == mk().nbs_id
    end

    @testset "flag_only: labels unchanged, moves still reported" begin
        nodes = [mknode(10,1, 0,0,0.0; r=0.15), mknode(20,2, 0,0,0.6; r=0.07)]
        res = FLiP.refine_nbs(nbs_id=Int32[1,1,2,2], node_id=Int32[1,1,2,2],
                              trial_node_id=Int32[10,10,20,20], nodes=nodes,
                              graph_skeleton=sg(2, [(1,2)]), skel_to_nbs=Int32[1,2],
                              cfg=base_cfg(mode="flag_only"))
        @test res.nbs_id == Int32[1,1,2,2]      # cloud untouched
        @test res.n_rule_b_merges == 1          # but the merge is still detected
    end

end
