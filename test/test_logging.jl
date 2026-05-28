@testset "Logging helpers" begin

    @testset "_fmt_elapsed — adaptive units" begin
        @test FLiP._fmt_elapsed(0.34)  == "0.34s"
        @test FLiP._fmt_elapsed(1.234) == "1.2s"
        @test FLiP._fmt_elapsed(59.5)  == "59.5s"
        @test FLiP._fmt_elapsed(84.0)  == "1m 24s"
        @test FLiP._fmt_elapsed(3600.0) == "60m 0s"
    end

    @testset "ProgressReporter — single threaded" begin
        p = FLiP.ProgressReporter("unit-test", 100)
        @test p.last_pct[] == -5
        for i in 1:100
            FLiP.report!(p, i)
        end
        # After processing all 100, the last reported boundary should be 100%.
        @test p.last_pct[] == 100
    end

    @testset "ProgressReporter — zero total is a no-op" begin
        p = FLiP.ProgressReporter("empty", 0)
        FLiP.report!(p, 0)   # should not throw
        FLiP.report!(p, 5)   # should not throw
        @test p.last_pct[] == -5
    end

    @testset "ProgressReporter — thread-safe under @threads" begin
        # Multiple threads racing on report! with a shared counter; the CAS
        # gate guarantees at most one print per 5% boundary and the atomic
        # ends at exactly 100%.
        p = FLiP.ProgressReporter("threaded", 1000)
        done = Threads.Atomic{Int}(0)
        Threads.@threads for i in 1:1000
            n = Threads.atomic_add!(done, 1) + 1
            FLiP.report!(p, n)
        end
        @test p.last_pct[] == 100
        @test done[] == 1000
    end

    @testset "_LOG_PREFIX constant defined" begin
        @test FLiP._LOG_PREFIX == "[FLiP]"
    end
end
