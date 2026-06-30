# CLAUDE.md — working guidelines for FLiP.jl

FLiP.jl is a forest-LiDAR processing package (preprocess → ground → tree segmentation →
QSM modeling → report). Scientific-computing code: correctness, numerical determinism, and
edge cases matter more than raw speed.

## Model & effort selection

Do **not** default to Opus + Ultracode for everything — it is the slowest, most expensive
setting and most work doesn't need it. Match the tier to the phase. The two phases where model
capability actually changes the outcome are **design** and **correctness review**; spend there,
go cheaper for mechanical work.

| Phase | Model | Effort | Orchestration |
|---|---|---|---|
| Architecture / planning | Opus | high–xhigh | Plan mode; ≥1 Plan agent |
| Codebase exploration | Sonnet (Opus if subtle) | medium | Explore agents, fan out |
| Implementation — numerical/algorithmic core | Opus | medium–high | solo |
| Implementation — mechanical (refactor moves, config, I/O, plotting) | Sonnet | low–medium | solo |
| Code review of correctness-critical changes | Opus | high–xhigh | **adversarial multi-agent** |
| Trivial (renames, formatting, doc tweaks, running tests) | Haiku/Sonnet | low | solo |

**Ultracode (xhigh + workflows)** is a scalpel for *large, uncertain, correctness-critical*
work — big refactors/migrations, "audit the whole codebase for X", exhaustive edge-case hunts,
design tournaments. It is wasteful for conversational turns, small targeted edits, quick
lookups, and well-specified single-file changes.

`xhigh` ≠ "better": it adds latency/cost for hard *reasoning*, and gives no quality gain on
mechanical edits. For interactive Opus-quality work without the wait, use `/fast`.

**One-line policy:** plan on Opus, review adversarially on Opus, do the middle on whatever's
cheapest that the test harness can vouch for.

## Principles for this codebase

1. **A deterministic verification harness is the highest-leverage investment** — more than
   model choice. Golden outputs + byte-diff + a determinism re-run let large changes land
   safely regardless of which model edited. Build/keep that, and even cheap mechanical edits
   are safe because the harness catches drift.
2. **Correctness asymmetry → spend on review, not execution.** Numerical code fails silently
   (NaN/Inf, degenerate geometry, empty connected components, off-by-half voxel lattices).
   Adversarial review (several agents each *trying to find a regression*, with distinct lenses)
   beats high effort on the writing pass.
3. **Cheap tiers are fine for the glue** — configs (`analyses/*`), batch/SLURM scripts, LAS/E57
   I/O, figure code: most of the typing, little of the scientific risk.
4. Preserve the determinism invariants when touching tree segmentation: fixed voxel lattice
   `(k+0.5)·voxel_res`, atomic-offset id blocks canonicalized by a final `relabel_by_occurrence`,
   and `_parallel_for` regions writing disjoint slots.

## Verification (run after any change to src/)

```
julia --project=. -e 'using Pkg; Pkg.test()'      # full suite (currently 605/605)
```

For changes that touch tree segmentation or QSM modeling, also confirm **determinism** (run a
fixture twice → identical `nbs_id`/`tree_id`/`tree_nbs_id`) and, where output values matter, a
**golden byte-diff** of the `*qsm_nodes.csv` / `*qsm_trees.csv` tables against a pre-change
snapshot.

## Config

Configuration is one nested `[tree]` domain (`[tree]` key dials, `[tree.assembly]`,
`[tree.refine]`, `[tree.model]`); see `flip_config.toml`. A legacy shim in `load_config!` still
loads the old `[tree_segmentation]`/`[qsm]`/`[nbs_refine]` layout with a deprecation warning.

## Git

Push over SSH (`origin = git@github.com:xiangtaoxu/FLiP.jl.git`); HTTPS/`gh` are unavailable.
Commit/push only when asked.
