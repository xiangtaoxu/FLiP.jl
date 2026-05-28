# ── Parallelization primitives ────────────────────────────────────────────
#
# The only place in FLiP.jl that orchestrates threads with raw `Threads.@spawn` /
# `@sync`. Every other parallel routine in the package is built on top of these
# two primitives; callers pass an explicit thread budget (typically from
# `effective_nthreads(cfg)`), so the primitives never consult global config.

"""
    _parallel_for(f, n::Integer, n_thread::Integer)

Run `f(i)` for `i in 1:n`, splitting work across at most `n_thread` concurrent
tasks via `@sync` / `Threads.@spawn`. Uses **dynamic scheduling**: the `nt`
workers each pull the next index from a shared atomic counter, so skewed
per-index costs balance across threads (no thread is left holding all the heavy
items). Falls back to a plain serial loop when `n_thread <= 1`, `n <= 1`, or
`Threads.nthreads() == 1`.

Each `i` is processed by exactly one task, but **iteration order is
nondeterministic** — `f` must be independent across indices (write to disjoint
slots / per-index containers), never relying on `i` running before/after `i±1`.

`n_thread` should be the resolved thread budget from `effective_nthreads(cfg)`.
The function does not consult any global config; callers pass the count
explicitly.

`f` is specialized via the `where {F}` type parameter to avoid closure boxing.
"""
function _parallel_for(f::F, n::Integer, n_thread::Integer) where {F}
    n <= 0 && return nothing
    N  = Int(n)
    nt = min(Int(n_thread), N)
    if nt <= 1 || Threads.nthreads() == 1
        @inbounds for i in 1:N
            f(i)
        end
        return nothing
    end
    next = Threads.Atomic{Int}(1)
    @sync for _ in 1:nt
        Threads.@spawn while true
            i = Threads.atomic_add!(next, 1)   # returns old value, then +1
            i > N && break
            @inbounds f(i)
        end
    end
    return nothing
end

"""
    _parallel_findall(pred, N, n_thread) -> Vector{Int}

Collect, in ascending order, every `i in 1:N` for which `pred(i)` is true, scanning the
range in up to `n_thread` contiguous chunks concurrently. Each task fills its own buffer
and the buffers are concatenated in chunk order, so the result is deterministic and
independent of thread scheduling. Falls back to a serial scan for small `N` or one thread.
"""
function _parallel_findall(pred::F, N::Integer, n_thread::Integer) where {F}
    n = Int(N)
    n <= 0 && return Int[]
    nt = min(Int(n_thread), n)
    if nt <= 1 || Threads.nthreads() == 1
        out = Int[]
        @inbounds for i in 1:n
            pred(i) && push!(out, i)
        end
        return out
    end
    chunk   = cld(n, nt)
    nchunks = cld(n, chunk)
    parts   = Vector{Vector{Int}}(undef, nchunks)
    @sync for c in 1:nchunks
        lo = (c - 1) * chunk + 1
        hi = min(c * chunk, n)
        Threads.@spawn begin
            buf = Int[]
            @inbounds for i in lo:hi
                pred(i) && push!(buf, i)
            end
            parts[c] = buf
        end
    end
    out = Int[]
    sizehint!(out, sum(length, parts))
    for p in parts
        append!(out, p)
    end
    return out
end
