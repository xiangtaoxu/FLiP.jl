import Graphs

@testset "Graph" begin
    @testset "Connected component labels" begin
        coords = [
            0.0 0.0 0.0;
            0.1 0.0 0.0;
            5.0 5.0 5.0;
            5.2 5.0 5.0;
            10.0 0.0 0.0;
        ]

        labels = connected_component_labels(coords, 0.25)
        @test labels == [1, 1, 2, 2, 3]

        coords_rank = [
            0.0 0.0 0.0;
            5.0 0.0 0.0;
            5.1 0.0 0.0;
            5.2 0.0 0.0;
            10.0 0.0 0.0;
            10.1 0.0 0.0;
        ]
        labels_rank = connected_component_labels(coords_rank, 0.15)
        @test labels_rank == [3, 1, 1, 1, 2, 2]

        labels_min2 = connected_component_labels(coords, 0.25, 2)
        @test labels_min2 == [1, 1, 2, 2, 0]

        labels_min3 = connected_component_labels(coords, 0.25, 3)
        @test labels_min3 == [0, 0, 0, 0, 0]

        @test connected_component_labels(zeros(0, 3), 0.5) == Int[]
        @test connected_component_labels([1.0 2.0 3.0], 0.5) == [1]
        @test connected_component_labels([1.0 2.0 3.0], 0.5, 2) == [0]

        @test_throws ArgumentError connected_component_labels(rand(10, 2), 0.5)
        @test_throws ArgumentError connected_component_labels(coords, 0.0)
        @test_throws ArgumentError connected_component_labels(coords, 0.5, 0)
    end

    @testset "Connected component labels from graph" begin
        graph = Graphs.SimpleGraph(6)
        Graphs.add_edge!(graph, 1, 2)
        Graphs.add_edge!(graph, 2, 3)
        Graphs.add_edge!(graph, 4, 5)

        labels = connected_component_labels(graph)
        @test labels == [1, 1, 1, 2, 2, 3]

        labels_min2 = connected_component_labels(graph, 2)
        @test labels_min2 == [1, 1, 1, 2, 2, 0]

        labels_min4 = connected_component_labels(graph, 4)
        @test labels_min4 == [0, 0, 0, 0, 0, 0]

        @test connected_component_labels(Graphs.SimpleGraph(0)) == Int[]
        @test_throws ArgumentError connected_component_labels(graph, 0)
    end

    @testset "Connected components on subset" begin
        graph = Graphs.SimpleGraph(8)
        Graphs.add_edge!(graph, 1, 2)
        Graphs.add_edge!(graph, 2, 3)
        Graphs.add_edge!(graph, 4, 5)
        Graphs.add_edge!(graph, 6, 7)

        subset = [2, 1, 3, 5, 4, 8]
        ws = ConnectedComponentSubsetWorkspace(Graphs.nv(graph))

        labels = connected_component_subset!(ws, graph, subset, 1)
        @test labels == [1, 1, 1, 2, 2, 3]

        labels_min2 = connected_component_subset!(ws, graph, subset, 2)
        @test labels_min2 == [1, 1, 1, 2, 2, 0]

        @test connected_component_subset!(ws, graph, Int[], 1) == Int[]
        @test_throws ArgumentError connected_component_subset!(ws, graph, [0, 1], 1)
        @test_throws ArgumentError connected_component_subset!(ws, graph, [1, 1, 2], 0)
    end

    @testset "Radius graph construction" begin
        coords = [
            0.0 0.0 0.0;
            1.0 0.0 0.0;
            2.5 0.0 0.0;
        ]

        graph_data = build_radius_graph(coords, 1.1)
        graph = graph_data.graph
        weights = graph_data.weights

        @test Graphs.nv(graph) == 3
        @test Graphs.ne(graph) == 1
        @test Graphs.has_edge(graph, 1, 2)
        @test !Graphs.has_edge(graph, 2, 3)
        @test isapprox(weights[1, 2], 1.0; atol=1e-10)
        @test isapprox(weights[2, 1], 1.0; atol=1e-10)
        @test weights[1, 3] == 0.0

        empty_graph = build_radius_graph(zeros(0, 3), 1.0)
        @test Graphs.nv(empty_graph.graph) == 0
        @test size(empty_graph.weights) == (0, 0)

        @test_throws ArgumentError build_radius_graph(coords, 0.0)
        @test_throws ArgumentError build_radius_graph(rand(10, 2), 1.0)
    end

    @testset "Quotient graph" begin
        coords = [
            0.0 0.0 0.0;
            1.0 0.0 0.0;
            0.0 1.0 0.0;
            10.0 0.0 0.0;
            11.0 0.0 0.0;
        ]
        graph = Graphs.SimpleGraph(5)
        Graphs.add_edge!(graph, 1, 2)
        Graphs.add_edge!(graph, 2, 3)
        Graphs.add_edge!(graph, 3, 4)
        Graphs.add_edge!(graph, 4, 5)
        labels = [10, 10, 20, 30, 30]

        result = quotient_graph(coords, graph, labels)

        @test result.labels == [10, 20, 30]
        @test result.points ≈ [
            0.5 0.0 0.0;
            0.0 1.0 0.0;
            10.5 0.0 0.0;
        ] atol=1e-10
        @test Graphs.nv(result.graph) == 3
        @test Graphs.ne(result.graph) == 2
        @test Graphs.has_edge(result.graph, 1, 2)
        @test Graphs.has_edge(result.graph, 2, 3)
        @test !Graphs.has_edge(result.graph, 1, 3)
        @test all(isapprox.(collect(result.edge_vectors[(1, 2)]), [-1.0, 1.0, 0.0]; atol=1e-10))
        @test all(isapprox.(collect(result.edge_vectors[(2, 1)]), [1.0, -1.0, 0.0]; atol=1e-10))
        @test all(isapprox.(collect(result.edge_vectors[(2, 3)]), [10.0, -1.0, 0.0]; atol=1e-10))

        empty_result = quotient_graph(zeros(0, 3), Graphs.SimpleGraph(0), Int[])
        @test Graphs.nv(empty_result.graph) == 0
        @test size(empty_result.points) == (0, 3)
        @test isempty(empty_result.labels)
        @test isempty(empty_result.edge_vectors)

        @test_throws ArgumentError quotient_graph(coords, graph, [10, 10, 20, 30])
    end

    @testset "Shortest-path distances and slicing" begin
        coords = [
            0.0 0.0 0.0;
            1.0 0.0 0.0;
            2.0 0.0 0.0;
            3.0 0.0 0.0;
        ]
        graph_data = build_radius_graph(coords, 1.01)

        sp_idx = shortest_path_distances(coords, graph_data.graph, graph_data.weights, 1)
        @test sp_idx.target_idx == 1
        @test sp_idx.distances ≈ [0.0, 1.0, 2.0, 3.0]

        sp_point = shortest_path_distances(coords, graph_data.graph, graph_data.weights, [0.1, 0.0, 0.0])
        @test sp_point.target_idx == 1
        @test sp_point.distances ≈ [0.0, 1.0, 2.0, 3.0]

        sliced = slice_by_shortest_path(coords, graph_data.graph, graph_data.weights, 1, 1.5)
        @test sliced.slice_labels == [1, 1, 2, 3]

        coords_disc = [
            0.0 0.0 0.0;
            1.0 0.0 0.0;
            10.0 0.0 0.0;
        ]
        graph_disc = build_radius_graph(coords_disc, 1.1)
        sliced_disc = slice_by_shortest_path(coords_disc, graph_disc.graph, graph_disc.weights, [0.0, 0.0, 0.0], 1.0)
        @test sliced_disc.slice_labels == [1, 2, 0]
        @test isinf(sliced_disc.distances[3])

        empty_graph = build_radius_graph(zeros(0, 3), 1.0)

        @test_throws ArgumentError shortest_path_distances(coords, graph_data.graph, graph_data.weights, 0)
        @test_throws ArgumentError shortest_path_distances(zeros(0, 3), empty_graph.graph, empty_graph.weights, [0.0, 0.0, 0.0])
        @test_throws ArgumentError slice_by_shortest_path(coords, graph_data.graph, graph_data.weights, 1, 0.0)
    end

    @testset "Shortest paths on subset" begin
        coords = [
            0.0 0.0 0.0;
            1.0 0.0 0.0;
            2.0 0.0 0.0;
            3.0 0.0 0.0;
            4.0 0.0 0.0;
            1.0 1.0 0.0;
        ]
        graph_data = build_radius_graph(coords, 1.01)
        graph = graph_data.graph
        weights = graph_data.weights

        subset = [4, 3, 2, 1]
        ws = ShortestPathSubsetWorkspace(Graphs.nv(graph))
        res = shortest_path_subset!(ws, graph, weights, subset, 1)

        @test res.subset == subset
        @test res.target_idx == 4
        @test res.distances ≈ [3.0, 2.0, 1.0, 0.0]
        @test res.parents == [2, 3, 4, 0]

        subset_disc = [6, 1]
        res_disc = shortest_path_subset!(ws, graph, weights, subset_disc, 1)
        @test res_disc.distances[2] == 0.0
        @test isinf(res_disc.distances[1])

        @test_throws ArgumentError shortest_path_subset!(ws, graph, weights, [2, 3], 1)
    end

    @testset "Proto nodes from slice labels" begin
        coords = [
            0.00 0.0 0.0;
            0.10 0.0 0.0;
            0.20 0.0 0.0;
            1.00 0.0 0.0;
            5.00 0.0 0.0;
            5.10 0.0 0.0;
            8.00 0.0 0.0;
        ]
        slice_labels = [1, 1, 3, 3, 2, 2, 0]

        proto_nodes = generate_proto_nodes_from_slice_label(coords, slice_labels, 0.15)
        @test proto_nodes == [1, 1, 1, 3, 2, 2, 0]

        coords_multi = [
            0.00 0.0 0.0;
            0.10 0.0 0.0;
            0.20 0.0 0.0;
            3.00 0.0 0.0;
            4.00 0.0 0.0;
            4.10 0.0 0.0;
        ]
        slice_labels_multi = [1, 1, 3, 2, 2, 4]
        proto_nodes_multi = generate_proto_nodes_from_slice_label(coords_multi, slice_labels_multi, 0.15)
        @test proto_nodes_multi == [1, 1, 1, 2, 3, 3]

        coords_mincc = [
            0.00 0.0 0.0;
            0.10 0.0 0.0;
            3.00 0.0 0.0;
            6.00 0.0 0.0;
            6.05 0.0 0.0;
            9.00 0.0 0.0;
        ]
        slice_labels_mincc = [1, 1, 3, 2, 2, 4]
        proto_nodes_mincc = generate_proto_nodes_from_slice_label(coords_mincc, slice_labels_mincc, 0.15, 2)
        @test proto_nodes_mincc == [1, 1, 0, 2, 2, 0]

        pc = make_test_pointcloud(coords; attrs=Dict(:slice_label => Int32.(slice_labels)))
        @test generate_proto_nodes_from_slice_label(FLiP.coordinates(pc), FLiP.getattribute(pc, :slice_label), 0.15) == proto_nodes
        pc_mincc = make_test_pointcloud(coords_mincc; attrs=Dict(:slice_label => Int32.(slice_labels_mincc)))
        @test generate_proto_nodes_from_slice_label(FLiP.coordinates(pc_mincc), FLiP.getattribute(pc_mincc, :slice_label), 0.15, 2) == proto_nodes_mincc

        @test generate_proto_nodes_from_slice_label(zeros(0, 3), Int[], 0.15) == Int[]
        @test generate_proto_nodes_from_slice_label([1.0 2.0 3.0], [0], 0.15) == [0]
        @test generate_proto_nodes_from_slice_label([1.0 2.0 3.0], [5], 0.15) == [1]

        @test_throws ArgumentError generate_proto_nodes_from_slice_label(coords, slice_labels[1:6], 0.15)
        @test_throws ArgumentError generate_proto_nodes_from_slice_label(coords, slice_labels, 0.0)
        @test_throws ArgumentError generate_proto_nodes_from_slice_label(coords, slice_labels, 0.15, 0)
        @test_throws ArgumentError generate_proto_nodes_from_slice_label(rand(7, 2), slice_labels, 0.15)

        graph_res = build_radius_graph(coords, 0.15)
        proto_nodes_graph = generate_proto_nodes_from_slice_label(coords, graph_res.graph, slice_labels, min_cc_size=1)
        @test proto_nodes_graph == proto_nodes

        @test_throws ArgumentError generate_proto_nodes_from_slice_label(coords, graph_res.graph, slice_labels[1:6], min_cc_size=1)
        @test_throws ArgumentError generate_proto_nodes_from_slice_label(rand(7, 2), graph_res.graph, slice_labels, min_cc_size=1)
    end

    @testset "Longest linear path" begin
        coords = [
            0.0 0.0 0.0;
            0.0 0.0 1.0;
            0.0 0.0 2.0;
            1.0 0.0 2.0;
            0.0 0.0 3.0;
            2.0 0.0 2.0;
            3.0 0.0 2.0;
        ]

        tree = Graphs.SimpleGraph(7)
        Graphs.add_edge!(tree, 1, 2)
        Graphs.add_edge!(tree, 2, 3)
        Graphs.add_edge!(tree, 3, 5)
        Graphs.add_edge!(tree, 2, 4)
        Graphs.add_edge!(tree, 4, 6)
        Graphs.add_edge!(tree, 6, 7)

        path = longest_linear_path(tree, coords, 1)
        @test path.vertices == [1, 2, 3, 5]
        @test isapprox(path.length, 3.0; atol=1e-10)

        cycle_graph = Graphs.SimpleGraph(3)
        Graphs.add_edge!(cycle_graph, 1, 2)
        Graphs.add_edge!(cycle_graph, 2, 3)
        Graphs.add_edge!(cycle_graph, 1, 3)
        cycle_path = longest_linear_path(cycle_graph, coords[1:3, :], 1)
        @test cycle_path.vertices == [1, 2, 3]
        @test isapprox(cycle_path.length, 2.0; atol=1e-10)

        # Path must stop when continuation would exceed 60°
        sharp_graph = Graphs.SimpleGraph(3)
        Graphs.add_edge!(sharp_graph, 1, 2)
        Graphs.add_edge!(sharp_graph, 2, 3)
        sharp_coords = [
            0.0 0.0 0.0;
            0.0 0.0 1.0;
            1.0 0.0 1.0;
        ]
        sharp_path = longest_linear_path(sharp_graph, sharp_coords, 1)
        @test sharp_path.vertices == [1, 2]
        @test isapprox(sharp_path.length, 1.0; atol=1e-10)

        @test_throws ArgumentError longest_linear_path(tree, coords, 0)
    end
end