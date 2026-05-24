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
end
