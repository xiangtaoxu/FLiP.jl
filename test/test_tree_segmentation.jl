@testset "tree_segmentation internals" begin
    @testset "_seed_trees_from_nearground!" begin
        # Two NBS: NBS 1 near ground, NBS 2 above the ceiling.
        # nbs_points is Vector{Vector{Int}} indexed by NBS id (1..K_nbs).
        nbs_points = [[1, 2, 3], [4, 5]]
        agh        = [0.1, 0.2, 0.15, 3.0, 3.5]
        tree_id    = zeros(Int32, 5)

        seeded = FLiP._seed_trees_from_nearground!(tree_id, nbs_points, agh, 0.5)

        @test seeded.nbs_tree == Int32[1, 0]   # NBS 1 assigned, NBS 2 not
        @test seeded.assigned_nbs == BitVector([true, false])
        @test tree_id[1:3] == Int32[1, 1, 1]
        @test all(==(Int32(0)), tree_id[4:5])
        @test seeded.next_tree_id == Int32(2)

        # All NBS near-ground — both seeded, tree ids 1 and 2 (dense iteration order)
        tree_id2 = zeros(Int32, 5)
        seeded2  = FLiP._seed_trees_from_nearground!(tree_id2, nbs_points,
                                                     [0.1, 0.2, 0.15, 0.2, 0.3], 0.5)
        @test seeded2.next_tree_id == Int32(3)
        @test seeded2.nbs_tree == Int32[1, 2]
        @test count(seeded2.assigned_nbs) == 2
        @test all(>(Int32(0)), tree_id2)

        # No NBS near-ground — nothing seeded
        tree_id3 = zeros(Int32, 5)
        seeded3  = FLiP._seed_trees_from_nearground!(tree_id3, nbs_points,
                                                     [5.0, 5.0, 5.0, 5.0, 5.0], 0.5)
        @test seeded3.next_tree_id == Int32(1)
        @test all(==(Int32(0)), seeded3.nbs_tree)
        @test !any(seeded3.assigned_nbs)
        @test all(==(Int32(0)), tree_id3)

        # Empty entries in nbs_points are skipped without error
        nbs_points_sparse = [Int[], [1, 2], Int[]]
        tree_id4 = zeros(Int32, 2)
        seeded4  = FLiP._seed_trees_from_nearground!(tree_id4, nbs_points_sparse,
                                                     [0.1, 0.2], 0.5)
        @test seeded4.nbs_tree == Int32[0, 1, 0]
        @test seeded4.assigned_nbs == BitVector([false, true, false])
        @test tree_id4 == Int32[1, 1]
    end

    @testset "_init_assembly_info" begin
        # 5 points, 2 NBS, 2 skeleton nodes:
        #  points 1,2 → NBS 1, node 1
        #  points 3,4 → NBS 2, node 2
        #  point 5    → unassigned (nbs_id=0, node_id=0)
        # Point-graph edges: 1-2 (intra-NBS 1), 2-3 (cross 1↔2), 3-4 (intra-NBS 2)
        nbs_id  = [1, 1, 2, 2, 0]
        node_id = [1, 1, 2, 2, 0]
        graph   = Graphs.SimpleGraph(5)
        Graphs.add_edge!(graph, 1, 2)
        Graphs.add_edge!(graph, 2, 3)
        Graphs.add_edge!(graph, 3, 4)

        # Skeleton: one vertex per node, one edge between them
        graph_skeleton = Graphs.SimpleGraph(2)
        Graphs.add_edge!(graph_skeleton, 1, 2)
        skel_coords = [0.0 0.0 0.0; 1.0 0.0 0.0]
        skel_pc     = FLiP.PointCloud(skel_coords, Dict{Symbol,Vector}())
        FLiP.setattribute!(skel_pc, :node_id, Int32[1, 2])

        info = FLiP._init_assembly_info(graph, nbs_id, node_id,
                                        graph_skeleton, skel_pc, 2, 2)

        @test info.nbs_points == [[1, 2], [3, 4]]
        @test info.node_to_skel == [1, 2]
        @test info.skel_to_nbs == [1, 2]
        @test info.nbs_skel_nodes == [[1], [2]]
        # Symmetric: one cross-NBS edge between NBS 1 and NBS 2
        @test info.nbs_adj[1, 2] == 1
        @test info.nbs_adj[2, 1] == 1
        @test info.nbs_adj[1, 1] == 0     # intra-NBS edges not stored
    end

    @testset "_seed_largest_nbs!" begin
        # Three NBS, the second is largest by point count.
        nbs_points   = [Int[1, 2, 3], Int[4, 5, 6, 7, 8], Int[]]
        assigned_nbs = falses(3)
        nbs_tree     = zeros(Int32, 3)
        tree_id      = zeros(Int32, 8)

        tid = FLiP._seed_largest_nbs!(tree_id, nbs_tree, assigned_nbs,
                                      nbs_points, Int32(1))

        @test tid == Int32(1)
        @test nbs_tree == Int32[0, 1, 0]
        @test assigned_nbs == BitVector([false, true, false])
        @test tree_id[1:3] == zeros(Int32, 3)         # NBS 1 untouched
        @test tree_id[4:8] == ones(Int32, 5)          # NBS 2 seeded as tree 1

        # Ties pick the first NBS at the max count (strict > in argmax)
        nbs_points2   = [Int[1, 2], Int[3, 4]]
        assigned_nbs2 = falses(2)
        nbs_tree2     = zeros(Int32, 2)
        tree_id2      = zeros(Int32, 4)
        @test FLiP._seed_largest_nbs!(tree_id2, nbs_tree2, assigned_nbs2,
                                      nbs_points2, Int32(7)) == Int32(7)
        @test nbs_tree2 == Int32[7, 0]
        @test tree_id2  == Int32[7, 7, 0, 0]

        # All-empty NBS slot list returns 0 and leaves everything untouched
        nbs_points3   = [Int[], Int[]]
        assigned_nbs3 = falses(2)
        nbs_tree3     = zeros(Int32, 2)
        tree_id3      = zeros(Int32, 0)
        @test FLiP._seed_largest_nbs!(tree_id3, nbs_tree3, assigned_nbs3,
                                      nbs_points3, Int32(3)) == Int32(0)
        @test !any(assigned_nbs3)
        @test all(==(Int32(0)), nbs_tree3)
    end

    # Shared fixture: a 3-NBS linear chain. Each NBS has 2 skeleton nodes so
    # `frac_connected` can land on the Rule-A side of the merge threshold.
    #
    # NBS 1: pts 1-4  → nodes 1,1,2,2 → skel verts 1,2
    # NBS 2: pts 5-8  → nodes 3,3,4,4 → skel verts 3,4
    # NBS 3: pts 9-12 → nodes 5,5,6,6 → skel verts 5,6
    #
    # Point-graph edges: intra-node + intra-NBS (between the two nodes) +
    # one cross-NBS edge each at the NBS boundary.
    # Skeleton edges: 1-2, 3-4, 5-6 (intra-NBS) and 2-3, 4-5 (NBS boundaries).
    function _make_chain_fixture()
        nbs_id  = Int[1,1,1,1, 2,2,2,2, 3,3,3,3]
        node_id = Int[1,1,2,2, 3,3,4,4, 5,5,6,6]

        graph = Graphs.SimpleGraph(12)
        # Intra-node edges
        for (u, v) in [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12)]
            Graphs.add_edge!(graph, u, v)
        end
        # Intra-NBS, cross-node edges (so the two nodes of each NBS are linked)
        for (u, v) in [(2,3), (6,7), (10,11)]
            Graphs.add_edge!(graph, u, v)
        end
        # Cross-NBS boundary edges
        Graphs.add_edge!(graph, 4, 5)   # NBS 1 ↔ NBS 2
        Graphs.add_edge!(graph, 8, 9)   # NBS 2 ↔ NBS 3

        # 6 skeleton vertices, one per node
        graph_skeleton = Graphs.SimpleGraph(6)
        for (u, v) in [(1,2), (3,4), (5,6), (2,3), (4,5)]
            Graphs.add_edge!(graph_skeleton, u, v)
        end
        skel_coords = Float64[
            0.0 0.0 0.0;
            1.0 0.0 0.0;
            2.0 0.0 0.0;
            3.0 0.0 0.0;
            4.0 0.0 0.0;
            5.0 0.0 0.0;
        ]
        skel_pc = FLiP.PointCloud(skel_coords, Dict{Symbol,Vector}())
        FLiP.setattribute!(skel_pc, :node_id,  Int32[1, 2, 3, 4, 5, 6])
        FLiP.setattribute!(skel_pc, :n_points, Int32[2, 2, 2, 2, 2, 2])

        # Coords just need to be N×3 finite numbers; values don't influence assembly.
        coords = hcat(collect(1.0:12.0), zeros(12), zeros(12))

        return (graph=graph, coords=coords, nbs_id=nbs_id, node_id=node_id,
                graph_skeleton=graph_skeleton, skel_pc=skel_pc)
    end

    @testset "_iterative_tree_growth!" begin
        f = _make_chain_fixture()

        K_nbs = 3
        info  = FLiP._init_assembly_info(f.graph, f.nbs_id, f.node_id,
                                         f.graph_skeleton, f.skel_pc, K_nbs, 6)

        # Pre-seed NBS 1 as tree 1 (manual stand-in for step 4.1)
        tree_id      = zeros(Int32, 12)
        tree_nbs_id  = Int32.(f.nbs_id)
        nbs_tree     = zeros(Int32, K_nbs)
        assigned_nbs = falses(K_nbs)

        nbs_tree[1]     = Int32(1)
        assigned_nbs[1] = true
        for i in info.nbs_points[1]
            tree_id[i] = Int32(1)
        end

        n_iter = FLiP._iterative_tree_growth!(
            tree_id, tree_nbs_id, nbs_tree, assigned_nbs,
            K_nbs, info.nbs_points, info.nbs_skel_nodes, info.skel_to_nbs,
            info.nbs_adj, f.graph_skeleton, 0.5,
        )

        @test n_iter >= 1
        @test all(assigned_nbs)                        # every NBS got a tree
        @test all(nbs_tree .== Int32(1))               # propagated from NBS 1
        @test all(tree_id  .== Int32(1))               # every point assigned
    end

    @testset "assemble_segments: ungrounded components become orphans" begin
        f = _make_chain_fixture()

        # No near-ground NBS (lowest AGH above the ceiling 0.3 + 2*0.05 = 0.4): the
        # largest NBS is seeded and the component is grown so branches get a
        # `tree_nbs_id`, but the temporary `tree_id` is zeroed → every point becomes an
        # orphan (tree_id==0, tree_nbs_id>0) for the occlusion rescue.
        agh_no_ground = fill(1.0, 12)
        res_ung = FLiP.assemble_segments(f.graph, f.coords, f.nbs_id, f.node_id,
                                         agh_no_ground, f.graph_skeleton, f.skel_pc)
        @test all(==(Int32(0)), res_ung.tree_id)       # no positive tree_id
        @test all(>(Int32(0)), res_ung.tree_nbs_id)    # branches kept

        # With a near-ground NBS: grounded → positive tree_id, one tree.
        agh_with_ground = vcat(fill(0.1, 4), fill(1.0, 4), fill(2.0, 4))
        res_g = FLiP.assemble_segments(f.graph, f.coords, f.nbs_id, f.node_id,
                                       agh_with_ground, f.graph_skeleton, f.skel_pc)
        @test all(>(Int32(0)), res_g.tree_id)
        @test length(unique(res_g.tree_id[res_g.tree_id .> 0])) == 1

        # K_nbs == 0 (all NBS discarded): nothing assigned.
        nbs_none = zeros(Int, 12)
        res_none = FLiP.assemble_segments(f.graph, f.coords, nbs_none, f.node_id,
                                          agh_no_ground, f.graph_skeleton, f.skel_pc)
        @test all(==(Int32(0)), res_none.tree_id)
        @test all(==(Int32(0)), res_none.tree_nbs_id)
    end

    @testset "assemble_segments: enable_rule_b toggles connectivity merge" begin
        # NBS1 (pts 1-2, node 1) grounded; NBS2 (pts 3-6, nodes 2,3) fully straddles
        # NBS1 — both NBS2 skeleton nodes touch node 1, so frac_connected = 1.0 > 0.5.
        nbs_id  = Int[1,1, 2,2, 2,2]
        node_id = Int[1,1, 2,2, 3,3]
        graph = Graphs.SimpleGraph(6)
        for (u, v) in [(1,2), (3,4), (5,6), (4,5), (2,3)]   # incl. one cross-NBS edge (2-3)
            Graphs.add_edge!(graph, u, v)
        end
        graph_skeleton = Graphs.SimpleGraph(3)
        for (u, v) in [(1,2), (1,3), (2,3)]
            Graphs.add_edge!(graph_skeleton, u, v)
        end
        skel_pc = FLiP.PointCloud(Float64[0 0 0; 1 0 0; 1 1 0], Dict{Symbol,Vector}())
        FLiP.setattribute!(skel_pc, :node_id,  Int32[1, 2, 3])
        FLiP.setattribute!(skel_pc, :n_points, Int32[2, 2, 2])
        coords = hcat(collect(1.0:6.0), zeros(6), zeros(6))
        agh = Float64[0.1, 0.1, 1.0, 1.0, 2.0, 2.0]   # only NBS1 near ground

        # Rule B ON (default): NBS2 merges into NBS1 → a single branch in one tree.
        res_on = FLiP.assemble_segments(graph, coords, nbs_id, node_id, agh,
                                        graph_skeleton, skel_pc)
        @test all(>(Int32(0)), res_on.tree_id)
        @test length(unique(res_on.tree_nbs_id[res_on.tree_nbs_id .> 0])) == 1

        # Rule B OFF: NBS2 stays its own branch → two branches within the one tree.
        res_off = FLiP.assemble_segments(graph, coords, nbs_id, node_id, agh,
                                         graph_skeleton, skel_pc; enable_rule_b=false)
        @test length(unique(res_off.tree_id[res_off.tree_id .> 0])) == 1      # still one tree
        @test length(unique(res_off.tree_nbs_id[res_off.tree_nbs_id .> 0])) == 2  # two branches
    end

    # Defaults: occlusion_tol=0.1, sub_res=0.05 → exact link distance 0.15, voxel 0.2.
    @testset "assemble_occluded_segments: orphan branch adjacent to grounded tree" begin
        coords = Float64[
            0.00 0.0 0.0; 0.00 0.0 1.0; 0.00 0.0 2.0; 0.00 0.0 3.0;  # grounded tree 1
            0.05 0.0 0.0; 0.05 0.0 1.0;                              # orphan branch 5
        ]
        tree_id     = Int32[1, 1, 1, 1, 0, 0]
        tree_nbs_id = Int32[1, 1, 1, 1, 5, 5]
        cfg = FLiP.FLiPConfig(Dict{String,Any}())
        FLiP.assemble_occluded_segments(coords, tree_id, tree_nbs_id; cfg=cfg)
        @test tree_id[5:6]     == Int32[1, 1]   # adopted the grounded tree
        @test tree_nbs_id[5:6] == Int32[5, 5]   # kept its own branch id
    end

    @testset "assemble_occluded_segments: bridges orphan→orphan→grounded chain" begin
        # pt2 (0.1 from grounded pt1) links directly; pt3 (0.2 from pt1) only reaches
        # ground through pt2 once pt2 is itself assigned.
        coords = Float64[0.0 0.0 0.0; 0.1 0.0 0.0; 0.2 0.0 0.0]
        tree_id     = Int32[1, 0, 0]
        tree_nbs_id = Int32[1, 5, 6]
        cfg = FLiP.FLiPConfig(Dict{String,Any}())
        FLiP.assemble_occluded_segments(coords, tree_id, tree_nbs_id; cfg=cfg)
        @test tree_id     == Int32[1, 1, 1]   # whole chain grounded
        @test tree_nbs_id == Int32[1, 5, 6]   # branch ids preserved
    end

    @testset "assemble_occluded_segments: branches migrate to nearest grounded tree" begin
        coords = Float64[0.0 0.0 0.0; 10.0 0.0 0.0; 0.1 0.0 0.0; 10.1 0.0 0.0]
        tree_id     = Int32[1, 2, 0, 0]
        tree_nbs_id = Int32[1, 2, 5, 6]
        cfg = FLiP.FLiPConfig(Dict{String,Any}())
        FLiP.assemble_occluded_segments(coords, tree_id, tree_nbs_id; cfg=cfg)
        @test tree_id[3] == Int32(1)            # branch 5 → tree 1
        @test tree_id[4] == Int32(2)            # branch 6 → tree 2 (separately)
        @test tree_nbs_id == Int32[1, 2, 5, 6]
    end

    @testset "assemble_occluded_segments: isolated orphan stays ungrounded" begin
        coords = Float64[0.0 0.0 0.0; 100.0 0.0 0.0]
        tree_id     = Int32[1, 0]
        tree_nbs_id = Int32[1, 5]
        cfg = FLiP.FLiPConfig(Dict{String,Any}())
        FLiP.assemble_occluded_segments(coords, tree_id, tree_nbs_id; cfg=cfg)
        @test tree_id     == Int32[1, 0]   # not rescued
        @test tree_nbs_id == Int32[1, 5]   # branch id kept
    end

    @testset "assemble_occluded_segments: disabled when occlusion_tol ≤ 0" begin
        coords = Float64[0.0 0.0 0.0; 0.05 0.0 0.0]
        tree_id     = Int32[1, 0]
        tree_nbs_id = Int32[1, 5]
        cfg = FLiP.FLiPConfig(Dict{String,Any}())
        cfg.tree_segmentation.assembly_occlusion_tolerance = 0.0
        FLiP.assemble_occluded_segments(coords, tree_id, tree_nbs_id; cfg=cfg)
        @test tree_id     == Int32[1, 0]   # untouched
        @test tree_nbs_id == Int32[1, 5]
    end

    @testset "_relabel_tree_nbs_within_trees!" begin
        # tree 1:{10×2, 20×2}, tree 2:{7}, tree 0:{99}; tnbs 0 stays 0.
        tree_id     = Int32[1, 1, 1, 1, 2, 0, 0]
        tree_nbs_id = Int32[10, 10, 20, 20, 7, 99, 0]
        FLiP._relabel_tree_nbs_within_trees!(tree_id, tree_nbs_id)
        # Valid trees first (tree_id asc, count desc, tnbs asc); the tree_id==0 group LAST:
        # (1,10)→1, (1,20)→2, (2,7)→3, then (0,99)→4.
        @test tree_nbs_id == Int32[1, 1, 2, 2, 3, 4, 0]
        # globally unique: each new tnbs maps to exactly one tree.
        @test length(unique(tree_nbs_id[tree_nbs_id .> 0])) == 4
    end
end
