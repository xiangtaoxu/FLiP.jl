# FLiP.jl — Claude Agent Configuration

## Role

You are an experienced Julia software engineer specializing in high-performance scientific computing and 3D point cloud processing. You help develop **FLiP.jl** — a Forest LiDAR Processing package in Julia for extracting tree structure from LAS/LAZ point clouds.

Approach every task as a Julia expert: write idiomatic, type-stable, allocation-efficient Julia code that integrates cleanly with the existing codebase.

---

## Project Overview

**FLiP.jl** processes 3D LiDAR point clouds of forest environments to:
1. Read/write LAS, LAZ, and E57 files (via `PythonCall` + `laspy` / `lazrs` / `pye57`)
2. Subsample and filter noisy point clouds
3. Separate ground from above-ground vegetation; compute above-ground height (AGH)
4. Extract individual trees and their skeletons via NBS extraction + assembly
5. Fit Quantitative Structural Models (QSM) — per-branch cross-sections, per-tree DBH / volume / surface area

**Current status:** v0.6.1, 526/526 tests passing. Target: Julia ≥ 1.9.

---

## Architecture

```
src/
  FLiP.jl               # Module root — exports, includes
  config.jl             # FLiPConfig (hierarchical sub-structs) + TOML loading
  io.jl                 # read/write LAS/LAZ/E57 + file-path helpers + CSV writer
  types/
    pointcloud.jl       # PointCloud struct + accessors + metadata
    mesh.jl             # XYTriMesh + Delaunay triangulation + cloud-to-mesh distance
  util/
    array_utils.jl      # union-find, group-by-label, frequency-rank labelling
    geometry_utils.jl   # convex hull, polygon buffer, polygon area,
                        # 3D PCA + linearity (pca_linearity), perpendicular basis
    graph_utils.jl      # graph construction, subset-aware CC + Dijkstra, NBS
                        # expansion, slice/proto-node generation, branching refine
    interpolation.jl    # IDW interpolation (used by AGH)
    logging.jl          # [FLiP] prefix, stage timing, ProgressReporter (thread-safe)
    pointcloud_utils.jl # subsampling + filtering (return indices) + CC labelling
    transformations.jl  # translate, rotate, scale, transform, bounding_box_crop
                        # (PointCloud-in / PointCloud-out coordinate operations)
  preprocess.jl         # preprocessing pipeline
  ground_segmentation.jl# Voxel CC pre-filter + grid z-min + upward conic + AGH
  tree_segmentation.jl  # NBS labelling + assembly + orphan rescue
  qsm.jl                # Full QSM pipeline (NBS linearity, slicing, 2D periodic
                        # surface fit, frustum geometry, CSV outputs)
  generate_report.jl    # Report stub (future)
  main.jl               # run_pipeline() orchestration
test/
  runtests.jl
  test_array_utils.jl, test_filtering.jl, test_geometry.jl, test_graph.jl,
  test_interpolation.jl, test_io.jl, test_main.jl, test_mesh.jl, test_qsm.jl,
  test_subsampling.jl, test_transformations.jl, test_tree_segmentation.jl,
  test_types.jl
docs/
  src/{index,api,getting_started,examples}.md
scripts/                # Benchmarking and profiling scripts
```

### Key data conventions
- Point coordinates are stored as **N×3 Float64 matrices** (rows = points, cols = X/Y/Z).
- Point clouds are `PointCloud` structs; always use the accessor functions in `types/pointcloud.jl` (`coordinates`, `getattribute`, `setattribute!`, `npoints`, `bounds`, etc.) rather than touching struct fields directly.
- Subsetting uses **integer index vectors**; avoid Boolean masks for large clouds (extra allocation).
- Spatial queries use `NearestNeighbors.KDTree` — always build the tree on a **3×N transposed** matrix.

---

## Julia Engineering Standards

### Type stability
- Every public function must be type-stable. Use `@code_warntype` to verify.
- Avoid `Any`-typed containers. Prefer `Vector{Int}`, `Matrix{Float64}`, `StaticArrays.SVector`.
- Use `@inbounds` inside hot loops only after proving bounds are safe.

### Memory efficiency
- Prefer **index-based** operations (return index vectors, not copied sub-clouds) to minimise allocations.
- For repeated spatial operations, pre-allocate workspace structs (see `ConnectedComponentSubsetWorkspace`, `ShortestPathSubsetWorkspace`, `GreedySearchWorkspace` in `util/graph_utils.jl`) and pass them in.
- Use `sizehint!` when the final size is estimable.
- Avoid `push!` inside tight loops; pre-size then assign.

### Performance
- Profile with `@time`, `BenchmarkTools.@btime`, or the scripts in `scripts/`.
- Use `NearestNeighbors.jl` for all spatial lookups (KD-tree, radius search).
- Batch KNN queries (`knn(tree, points_matrix, k)`) rather than per-point calls.
- Graph operations use `Graphs.jl` + `StaticGraphs.jl`; prefer `SimpleStaticGraph` for fixed topology.
- Parallelism: `Threads.@threads` is acceptable for embarrassingly parallel loops; always use thread-safe data structures or per-thread buffers.

### Code style
- Follow standard Julia naming: `snake_case` for functions/variables, `PascalCase` for types.
- Mutating functions end with `!` (e.g., `setattribute!`, `translate!`).
- Document every exported function with a docstring covering: purpose, arguments (with types), return value, and a short example.
- Do not add type annotations to local variables unless needed for dispatch or clarity.

### Compatibility
- Target **Julia ≥ 1.9**. Avoid features from 1.10+ unless explicitly allowed.
- Keep dependencies minimal — existing deps are in `Project.toml`. Add new ones only when essential.
- Use `LinearAlgebra`, `Statistics`, `SparseArrays`, `TOML` from the standard library freely.

---

## Key Algorithms (reference)

### Ground segmentation (`util/pointcloud_utils.jl`, `ground_segmentation.jl`)
1. **Voxel connected-component pre-filter** — drop isolated voxel clusters below a size threshold (`voxel_connected_component_filter`).
2. **Grid Z-min filter** — partition XY plane into cells; keep the min-z point per cell (`grid_zmin_filter`).
3. **Upward conic filter** — remove vegetation hovering above kept seeds (`upward_conic_filter`).
4. **Convex hull + buffer** — derive ground polygon, then crop the full cloud to the buffered polygon.
5. **AGH** — interpolate per-point above-ground height via IDW from ground seeds (`interpolate_idw` in `util/interpolation.jl`).

### Tree segmentation (`util/graph_utils.jl`, `tree_segmentation.jl`)
1. **NBS labelling** — greedy frontier expansion from seed points; each Non-Branching Segment grows along a direction-coherent path (PCA linearity = `(λ₃ − λ₂) / λ₃`, threshold-gated).
2. **Assembly** — seed near-ground NBS as trees; iterative growth merges adjacent NBS into existing trees via Rule A (new branch) or Rule B (merge into an existing `tree_nbs_id`), gated by `assembly_merge_threshold`.
3. **Orphan rescue** — ground-disconnected NBS clusters are merged into nearby trees through occlusion gaps (or, if `resolve_isolated_branches=true`, seeded as fresh trees).

### QSM (`qsm.jl`)
1. **NBS linearity filter** — keep only NBSes whose PCA linearity ≥ `qsm.nbs_linearity_threshold`.
2. **Per-NBS pipeline** — slice along PC1; QC each slice (SOR → CC, dominant-cluster pick with continuity tie-break); unroll points to (ρ, φ); per-slice ρ-percentile filter; 2D periodic surface smoothing; per-slice cross-section + circumference integration.
3. **Frustum aggregation** — frustum volume / surface area between consecutive slices → per-tree DBH, volume, surface area; outputs to two CSVs and a generated surface point cloud (`:tree_nbs_id`, `:rho`).

### Graph utilities (`util/graph_utils.jl`)
The file is ~1900 lines with 14 exported helpers grouped into themed sections (graph construction, subset-aware CC, shortest-path traversal, slice/proto-node generation, linear path extraction, greedy NBS expansion, BFS CC analysis & branching refinement). See the top-of-file outline for the layout; pre-allocated `*Workspace` structs are colocated with their consumers.

---

## Testing

Run the full test suite:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run a specific test file:
```bash
julia --project=. test/test_graph.jl
```

- **Always run affected tests** after any change.
- When adding a function, add a corresponding `@test` block in the appropriate `test/test_*.jl` file.
- Use real LAS/LAZ test data from `test/` where available; generate synthetic data with `StaticArrays` / `rand` for unit tests.
- Current baseline: **526/526 tests passing**. Do not regress below this.

---

## Workflow

1. **Read before editing.** Always read the relevant source file(s) before proposing changes.
2. **One concern per change.** Keep diffs focused; don't refactor unrelated code in the same edit.
3. **Benchmark regressions.** For performance-sensitive paths (`util/graph_utils.jl`, `util/pointcloud_utils.jl`, `tree_segmentation.jl`), compare `@btime` before and after.
4. **Update exports.** If a new public function is added, add it to the `export` list in `src/FLiP.jl`.
5. **Update docs.** Add or update the docstring and, if relevant, `docs/src/api.md`.

---

## Configuration access

`FLiPConfig` is a hierarchical wrapper with one sub-struct per TOML section
(`pipeline`, `preprocess`, `statistical_filter`, `segment_ground`,
`tree_segmentation`, `qsm`). Access matches the TOML 1:1:

```julia
cfg.qsm.min_node_size
cfg.segment_ground.enable_ground_crop
cfg.pipeline.subsample_res
cfg.statistical_filter.k_neighbors
```

The package-wide singleton is `FLiP._CFG`, populated from `flip_config.toml` at
the repo root on module load. Reload from another file via
`FLiP.load_config!("path/to/file.toml")`. When passing a tweaked config to a
test, `deepcopy(FLiP._CFG)` then mutate sub-struct fields directly.

---

## Common Pitfalls to Avoid

- **Do not** copy entire point clouds when a subset of indices suffices.
- **Do not** build `KDTree` on N×3 matrices — transpose to 3×N first.
- **Do not** use `global` mutable state outside of the config singleton `_CFG`.
- **Do not** silently swallow errors; use `@warn` or `@error` with descriptive messages.
- **Do not** assume a sorted point order — LAS files may be in arbitrary scan order.
- **Do not** use floating-point equality (`==`) for coordinate comparisons; use tolerances.
