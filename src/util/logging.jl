"""
Logging helpers for the FLiP pipeline.

Provides three primitives used across all stages:
- `_LOG_PREFIX` — uniform `[FLiP]` prefix for all stage messages.
- `_with_stage_timing(name, f)` — wraps a stage body with start / end
  announcements and wall-clock elapsed timing.
- `_log_stage_skipped(name, reason)` — one-liner for disabled stages.
- `ProgressReporter` + `report!` — thread-safe percentage-throttled progress
  reporter that emits at 5% boundaries. Single-CAS gate; safe under
  `Threads.@threads`.
- `_fmt_elapsed(dt)` — adaptive seconds → minutes formatting.

All log calls go through Julia's standard `@info`; helpers do not bypass the
active logger. Helpers are internals — no exports.
"""

# Prefix used by every stage log line — keeps `grep "[FLiP]"` useful.
const _LOG_PREFIX = "[FLiP]"

# ── Elapsed-time formatting ───────────────────────────────────────────────

"""
    _fmt_elapsed(dt::Real) -> String

Adaptive elapsed-time formatter: sub-second → `"0.34s"`, seconds →
`"1.2s"`, above 1 minute → `"1m 24s"`.
"""
function _fmt_elapsed(dt::Real)
    dt < 1.0  && return string(round(dt, digits=2), "s")
    dt < 60.0 && return string(round(dt, digits=1), "s")
    m, s = divrem(dt, 60.0)
    return string(Int(m), "m ", Int(round(s)), "s")
end

# ── Byte formatting ───────────────────────────────────────────────────────

"""
    _fmt_bytes(n::Integer) -> String

Format a byte count as GiB with one decimal, e.g. `"64.0 GiB"`.
"""
function _fmt_bytes(n::Integer)
    return string(round(n / 2^30, digits=1), " GiB")
end

# ── Session resource banner ───────────────────────────────────────────────

"""
    _log_session_info(cfg)

Emit a two-line `@info` summary of the host resources and thread budget for this
run. The first line reports CPU cores and RAM (free / total). The second line
makes the three distinct thread quantities explicit:

- `Threads.nthreads()` — threads available to Julia (launch-time `-t` / `JULIA_NUM_THREADS` cap).
- `cfg.pipeline.n_thread` — the raw value requested in the config.
- `effective_nthreads(cfg)` — the resolved budget actually used (capped at the above).
"""
function _log_session_info(cfg)
    @info "$_LOG_PREFIX session: $(Sys.CPU_THREADS) CPU cores, " *
          "$(_fmt_bytes(Sys.free_memory())) free / $(_fmt_bytes(Sys.total_memory())) total RAM"
    @info "$_LOG_PREFIX threads: $(Threads.nthreads()) available to Julia, " *
          "config n_thread=$(cfg.pipeline.n_thread), using $(effective_nthreads(cfg))"
end

# ── Stage timing wrappers ─────────────────────────────────────────────────

"""
    _with_stage_timing(f, stage_name) -> result-of-f

Emit a `>> <stage> starting` line, run `f()`, then emit a `<< <stage>
done (elapsed)` line. Returns whatever `f` returned. Start/end markers are
always visible — they are not gated by `enable_debug_info`.

Function-first signature so the standard do-block syntax works:
```julia
_with_stage_timing("preprocess") do
    preprocess(; cfg=cfg)
end
```
"""
function _with_stage_timing(f::Function, stage_name::AbstractString)
    @info "$_LOG_PREFIX >> $stage_name starting"
    t0 = time()
    result = f()
    @info "$_LOG_PREFIX << $stage_name done ($(_fmt_elapsed(time() - t0)))"
    return result
end

"""
    _log_stage_skipped(stage_name, reason="disabled by config")

One-liner for stages that are present in the pipeline but did not run.
"""
function _log_stage_skipped(stage_name::AbstractString,
                            reason::AbstractString="disabled by config")
    @info "$_LOG_PREFIX -- $stage_name skipped ($reason)"
end

# ── Thread-safe progress reporter ─────────────────────────────────────────

"""
    ProgressReporter(label, total)

Throttled progress reporter that emits `@info` only when a 5% boundary is
crossed on `n_done / total`. Thread-safe: the percentage gate is held in a
`Threads.Atomic{Int}` and the CAS in `report!` guarantees that at most one
thread emits per boundary.

Callers tracking `n_done` from inside a `Threads.@threads` loop should use a
shared `Threads.Atomic{Int}` for the count (atomic_add! per iteration) and
pass the snapshot value as `n_done`.
"""
mutable struct ProgressReporter
    label::String
    total::Int
    last_pct::Threads.Atomic{Int}
    t_start::Float64
end

ProgressReporter(label::AbstractString, total::Integer) =
    ProgressReporter(String(label), Int(total), Threads.Atomic{Int}(-5), time())

"""
    report!(p::ProgressReporter, n_done; extra="")

Emit a throttled progress line if `n_done / p.total` crossed the next 5%
boundary. No-op when `p.total == 0`. Optional `extra` is appended to the
message in parentheses.
"""
function report!(p::ProgressReporter, n_done::Integer; extra::AbstractString="")
    p.total > 0 || return
    pct = clamp(round(Int, 100.0 * n_done / p.total), 0, 100)
    while true
        last = p.last_pct[]
        pct < last + 5 && return
        new_last = pct - (pct % 5)
        # CAS: only one thread wins the boundary and prints.
        Threads.atomic_cas!(p.last_pct, last, new_last) === last || continue
        elapsed = _fmt_elapsed(time() - p.t_start)
        msg = isempty(extra) ?
              "$_LOG_PREFIX   $(p.label): $(new_last)% ($n_done/$(p.total), $elapsed)" :
              "$_LOG_PREFIX   $(p.label): $(new_last)% ($n_done/$(p.total), $elapsed, $extra)"
        @info msg
        return
    end
end
