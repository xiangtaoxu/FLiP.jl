@testset "tree_segmentation helpers" begin
    @testset "relabel_by_occurrence (extended semantics)" begin
        # Default: count all values, rank by descending count
        @test relabel_by_occurrence(Int[]) == Int[]
        @test relabel_by_occurrence([7, 4, 9, 7, 4, 7]) == [1, 2, 3, 1, 2, 1]

        # positive_only: zero and negatives stay 0 regardless of count
        @test relabel_by_occurrence([-1, 5, -1, 5, 5]; positive_only=true) ==
              [0, 1, 0, 1, 1]
        @test relabel_by_occurrence([0, 7, 0, 7, 4, 7]; positive_only=true) ==
              [0, 1, 0, 1, 2, 1]

        # T_out kwarg
        @test eltype(relabel_by_occurrence([2, 2, 3]; T_out=Int32)) == Int32
        @test relabel_by_occurrence(Int32[2, 2, 3, 3, 3]; T_out=Int32) ==
              Int32[2, 2, 1, 1, 1]

        # min_count drops below-threshold values
        @test relabel_by_occurrence([1, 1, 2, 3, 3, 3], 2) == [2, 2, 0, 1, 1, 1]

        # Invalid min_count
        @test_throws ArgumentError relabel_by_occurrence([1, 2, 3], 0)
    end

    @testset "group_indices_by_label" begin
        # Empty
        @test group_indices_by_label(Int[], Int[]) == Vector{Vector{Int}}()

        # Basic grouping — preserves label order
        @test group_indices_by_label([10, 20, 30, 40], [1, 1, 2, 0]) ==
              [[10, 20], [30]]

        # Label 0 entries dropped; otherwise empty result
        @test group_indices_by_label([5, 6, 7], [0, 0, 0]) == Vector{Vector{Int}}()

        # Length-mismatch should error
        @test_throws ArgumentError group_indices_by_label([1, 2], [1, 2, 3])

        # Non-contiguous labels — empty buckets get filtered out
        @test group_indices_by_label([100, 200, 300], [1, 3, 3]) ==
              [[100], [200, 300]]
    end

    @testset "_seed_trees_from_nearground!" begin
        # Two NBS: one with all points near ground, one above the ceiling
        nbs_points = Dict(1 => [1, 2, 3], 2 => [4, 5])
        agh        = [0.1, 0.2, 0.15, 3.0, 3.5]
        tree_id    = zeros(Int32, 5)

        seeded = FLiP._seed_trees_from_nearground!(tree_id, nbs_points, agh, 0.5)

        @test seeded.nbs_tree[1] == Int32(1)
        @test !haskey(seeded.nbs_tree, 2)
        @test 1 in seeded.assigned_nbs
        @test !(2 in seeded.assigned_nbs)
        @test tree_id[1:3] == Int32[1, 1, 1]
        @test all(==(Int32(0)), tree_id[4:5])
        @test seeded.next_tree_id == Int32(2)

        # All NBS near-ground — both seeded, tree ids 1 and 2
        tree_id2 = zeros(Int32, 5)
        seeded2  = FLiP._seed_trees_from_nearground!(tree_id2, nbs_points,
                                                     [0.1, 0.2, 0.15, 0.2, 0.3], 0.5)
        @test seeded2.next_tree_id == Int32(3)
        @test length(seeded2.assigned_nbs) == 2
        @test all(>(Int32(0)), tree_id2)

        # No NBS near-ground — nothing seeded
        tree_id3 = zeros(Int32, 5)
        seeded3  = FLiP._seed_trees_from_nearground!(tree_id3, nbs_points,
                                                     [5.0, 5.0, 5.0, 5.0, 5.0], 0.5)
        @test seeded3.next_tree_id == Int32(1)
        @test isempty(seeded3.assigned_nbs)
        @test all(==(Int32(0)), tree_id3)
    end

    @testset "Step 4.3 reorder: tree_nbs_id alone matches (tree_id, tree_nbs_id)" begin
        # 7 points across 3 trees and 4 distinct (tree_id, tree_nbs_id) groups:
        # group A (tree 1, nbs 7) has 3 points → should rank 1
        # group B (tree 2, nbs 4) has 2 points → should rank 2
        # group C (tree 3, nbs 9) has 1 point  → should rank 3
        # plus a (tree 0, nbs 5) unassigned point → should end at 0
        tree_id     = Int32[1, 1, 1, 2, 2, 3, 0]
        tree_nbs_id = Int32[7, 7, 7, 4, 4, 9, 5]

        # Reference: compound-key reorder (mirrors the old logic exactly)
        groups = Dict{Tuple{Int32,Int32}, Vector{Int}}()
        for i in eachindex(tree_id)
            (tree_id[i] > 0 && tree_nbs_id[i] > 0) || continue
            push!(get!(groups, (tree_id[i], tree_nbs_id[i]), Int[]), i)
        end
        sorted = sort!(collect(groups); by = kv -> -length(kv[2]))
        expected = zeros(Int32, length(tree_id))
        for (lbl, (_, pts)) in enumerate(sorted)
            for i in pts
                expected[i] = Int32(lbl)
            end
        end

        # New path: zero out unassigned first, then relabel_by_occurrence
        new_tnbs = copy(tree_nbs_id)
        for i in eachindex(tree_id)
            tree_id[i] == 0 && (new_tnbs[i] = Int32(0))
        end
        new_tnbs = relabel_by_occurrence(new_tnbs; positive_only=true, T_out=Int32)

        @test new_tnbs == expected
    end
end
