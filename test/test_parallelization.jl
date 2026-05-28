@testset "Parallelization primitives" begin

    @testset "_parallel_for — dynamic scheduling" begin
        # Empty / degenerate ranges.
        let hits = Threads.Atomic{Int}(0)
            FLiP._parallel_for(0, 4) do _; Threads.atomic_add!(hits, 1); end
            @test hits[] == 0
        end
        let hits = Threads.Atomic{Int}(0)
            FLiP._parallel_for(1, 4) do _; Threads.atomic_add!(hits, 1); end
            @test hits[] == 1
        end

        # Every index 1:n is visited exactly once, across thread budgets and for
        # n both smaller than and much larger than the budget. Each worker writes
        # its own disjoint slot, so no atomics are needed for correctness.
        for nt in (1, 3), n in (2, 1000)
            visited = zeros(Int, n)
            FLiP._parallel_for(n, nt) do i
                visited[i] += 1
            end
            @test all(==(1), visited)
        end

        # Skewed per-index cost (early indices sleep longer) must not change the
        # result: dynamic scheduling reorders execution but the per-slot output
        # matches the serial computation.
        let n = 200
            out = Vector{Int}(undef, n)
            FLiP._parallel_for(n, 3) do i
                i <= 5 && sleep(0.005)   # heavy items concentrated at the front
                out[i] = i * i
            end
            @test out == [i * i for i in 1:n]
        end
    end

    @testset "_parallel_findall — deterministic ascending collection" begin
        # Degenerate ranges.
        @test FLiP._parallel_findall(_ -> true, 0, 4) == Int[]
        @test FLiP._parallel_findall(isodd, 1, 4) == [1]
        @test FLiP._parallel_findall(_ -> false, 50, 4) == Int[]

        # Result must equal Base.findall(pred, 1:n) regardless of thread budget,
        # for n both below and well above the budget, and stay ascending.
        for nt in (1, 3), n in (5, 1000)
            expected = [i for i in 1:n if iseven(i)]
            for got in (FLiP._parallel_findall(iseven, n, nt),)
                @test got == expected
                @test issorted(got)
            end
        end

        # Serial and threaded paths agree on a non-trivial predicate.
        pred = i -> (i % 7 == 0) || (i % 11 == 0)
        @test FLiP._parallel_findall(pred, 500, 1) == FLiP._parallel_findall(pred, 500, 4)
    end
end
