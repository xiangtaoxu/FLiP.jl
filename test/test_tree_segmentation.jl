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

    @testset "resolve_isolated_branches via assemble_segments" begin
        f = _make_chain_fixture()

        # CC-B: no near-ground NBS. Even the lowest AGH is above the ceiling
        # (default threshold 0.3 + 2 * subsample_res 0.05 = 0.4).
        agh_no_ground = fill(1.0, 12)

        cfg_off = FLiP.FLiPConfig(Dict{String,Any}())
        cfg_off.tree_resolve_isolated_branches = false
        res_off = FLiP.assemble_segments(f.graph, f.coords, f.nbs_id, f.node_id,
                                         agh_no_ground, f.graph_skeleton, f.skel_pc;
                                         cfg=cfg_off)
        @test all(==(Int32(0)), res_off.tree_id)       # nothing assigned without the flag

        cfg_on = FLiP.FLiPConfig(Dict{String,Any}())
        cfg_on.tree_resolve_isolated_branches = true
        res_on = FLiP.assemble_segments(f.graph, f.coords, f.nbs_id, f.node_id,
                                        agh_no_ground, f.graph_skeleton, f.skel_pc;
                                        cfg=cfg_on)
        @test all(>(Int32(0)), res_on.tree_id)         # every point now has a tree id
        @test length(unique(res_on.tree_id)) == 1      # all merged into one tree

        # CC-A regression: with a near-ground NBS, the fallback must NOT fire,
        # so the result with the flag on equals the result with it off.
        agh_with_ground = vcat(fill(0.1, 4), fill(1.0, 4), fill(2.0, 4))

        res_gate_off = FLiP.assemble_segments(f.graph, f.coords, f.nbs_id, f.node_id,
                                              agh_with_ground, f.graph_skeleton, f.skel_pc;
                                              cfg=cfg_off)
        res_gate_on  = FLiP.assemble_segments(f.graph, f.coords, f.nbs_id, f.node_id,
                                              agh_with_ground, f.graph_skeleton, f.skel_pc;
                                              cfg=cfg_on)
        @test res_gate_off.tree_id == res_gate_on.tree_id
        @test length(unique(res_gate_on.tree_id[res_gate_on.tree_id .> 0])) == 1
    end

    @testset "_propagate_orphan_labels" begin
        # 3 orphan NBS:
        #  orphan 1: directly connected to tree 7 via KDTree pass
        #  orphan 2: orphan-orphan edge to orphan 1 only (no direct tree edge)
        #  orphan 3: no edges at all → stays unrescued
        coarse_o2o = [
            Dict{Int,Int}(2 => 5),    # orphan 1 ↔ orphan 2 (count 5)
            Dict{Int,Int}(1 => 5),
            Dict{Int,Int}(),          # orphan 3 isolated
        ]
        coarse_o2t = [
            Dict{Int32,Int}(Int32(7) => 10),   # orphan 1 → tree 7
            Dict{Int32,Int}(),
            Dict{Int32,Int}(),
        ]
        orphan_nbs_points = [[1, 2], [3, 4, 5], [6]]

        result = FLiP._propagate_orphan_labels(coarse_o2o, coarse_o2t, orphan_nbs_points)

        @test result[1] == Int32(7)   # rescued via direct tree edge
        @test result[2] == Int32(7)   # rescued via orphan 1 (after orphan 1 is in)
        @test result[3] == Int32(0)   # never rescued — no edges

        # Tie-breaking: most-voted tree wins
        coarse_o2o2 = [Dict{Int,Int}()]
        coarse_o2t2 = [Dict{Int32,Int}(Int32(3) => 1, Int32(9) => 7)]
        @test FLiP._propagate_orphan_labels(coarse_o2o2, coarse_o2t2, [[1]])[1] == Int32(9)

        # Empty input
        @test FLiP._propagate_orphan_labels(Dict{Int,Int}[], Dict{Int32,Int}[],
                                            Vector{Int}[]) == Int32[]
    end

    @testset "process_orphan_segments merge_threshold" begin
        # Pre-assigned tree 1: 4 points along the z-axis at x=y=0, NBS 1, node 1.
        # Orphan NBS 2 has 4 points spanning 2 distinct node_ids:
        #   node 2 (close, within occlusion_tol of tree):  pts 5,6
        #   node 3 (far from any assigned pt):              pts 7,8
        # Node-based frac_connected = 1 connected node of 2 total = 0.5.
        coords = Float64[
            0.00 0.0 0.0;
            0.00 0.0 1.0;
            0.00 0.0 2.0;
            0.00 0.0 3.0;
            0.05 0.0 0.0;
            0.05 0.0 1.0;
            10.0 0.0 0.0;
            10.0 0.0 1.0;
        ]
        nbs_id  = Int32[1, 1, 1, 1, 2, 2, 2, 2]
        node_id = Int32[1, 1, 1, 1, 2, 2, 3, 3]

        _initial_state() = (
            tree_id     = Int32[1, 1, 1, 1, 0, 0, 0, 0],
            tree_nbs_id = Int32[1, 1, 1, 1, 0, 0, 0, 0],
        )

        # Rule B: frac=0.5 > 0.0 → orphan merges into existing tree_nbs_id 1.
        cfg_b = FLiP.FLiPConfig(Dict{String,Any}())
        cfg_b.tree_assembly_merge_threshold = 0.0
        s_b = _initial_state()
        FLiP.process_orphan_segments(coords, nbs_id, node_id,
                                     s_b.tree_id, s_b.tree_nbs_id; cfg=cfg_b)
        @test s_b.tree_id[5:8]     == Int32[1, 1, 1, 1]   # orphan joins tree 1
        @test s_b.tree_nbs_id[5:8] == Int32[1, 1, 1, 1]   # merged into existing tnid

        # Rule A: frac=0.5 ≤ 0.5 → orphan keeps tree 1 but gets a fresh tnid (> 1).
        cfg_a = FLiP.FLiPConfig(Dict{String,Any}())
        cfg_a.tree_assembly_merge_threshold = 0.5
        s_a = _initial_state()
        FLiP.process_orphan_segments(coords, nbs_id, node_id,
                                     s_a.tree_id, s_a.tree_nbs_id; cfg=cfg_a)
        @test s_a.tree_id[5:8]                == Int32[1, 1, 1, 1]   # same tree_id
        @test all(>(Int32(1)), s_a.tree_nbs_id[5:8])                 # not merged into tnid 1
        @test length(unique(s_a.tree_nbs_id[5:8])) == 1              # all share one fresh tnid

        # merge_threshold = 1.0 fully suppresses orphan→tnid merging.
        cfg_strict = FLiP.FLiPConfig(Dict{String,Any}())
        cfg_strict.tree_assembly_merge_threshold = 1.0
        s_strict = _initial_state()
        FLiP.process_orphan_segments(coords, nbs_id, node_id,
                                     s_strict.tree_id, s_strict.tree_nbs_id; cfg=cfg_strict)
        @test all(>(Int32(1)), s_strict.tree_nbs_id[5:8])
    end
end
