"""
Cross-stage data types for the tree domain: the per-node geometric model (`QSMNode`) and the
NBS-refinement models/report rows. Defined before the tree-stage files that name them.
"""

"""Per-QSM node biometric results (one per z-slice per NBS)."""
mutable struct QSMNode
    qsm_node_id::Int
    nbs_id::Int32
    tree_id::Int32
    tree_nbs_id::Int32
    agh::Float64
    height::Float64
    completeness::Float64
    n_points::Int
    center_x::Float64
    center_y::Float64
    center_z::Float64
    direction_x::Float64
    direction_y::Float64
    direction_z::Float64
    cross_area::Float64
    circumference::Float64
    radius_area::Float64
    radius_circ::Float64
end

# ═══════════════════════════════════════════════════════════════════════════════
# ★ Public entry point: qsm
# ═══════════════════════════════════════════════════════════════════════════════
#
# High-level pipeline:
#   Stage 1 — _filter_linear_nbs   : keep NBSes that are linear enough to QSM
#   Stage 2 — _process_single_nbs! : per-NBS slicing, fitting, surface estimation
#               2a slicing & centerline   2d surface point cloud generation
#               2b unroll & rho stats     2e frustum geometry
#               2c 2D surface smoothing
#   Stage 3 — _build_node_table, _build_tree_table, _write_csv : aggregation & I/O
# ═══════════════════════════════════════════════════════════════════════════════


# ── NBS-refinement models + report rows ──

"""Per-node cylinder model (one trial-QSM slice). `trial_node_id` links to per-point `:trial_node_id`."""
struct NodeModel
    trial_node_id::Int
    seg_id::Int32                 # owning nbs_id (original, pre-refine)
    cyl::Cyl
    aabb::NTuple{6,Float64}
    vol_vox::Float64             # deterministic per-node self-volume (global lattice)
    n_points::Int
    completeness::Float64
    agh::Float64
end

"""Per-NBS focal anchor: union of the NBS's node cylinders + aggregates."""
struct SegModel
    seg_id::Int32
    cyls::Vector{Cyl}
    aabb::NTuple{6,Float64}       # union of node AABBs
    vol_vox::Float64             # Σ node vol_vox (size for ordering & donor/receiver)
    n_points::Int                # Σ node n_points (per-focal min_points_gate)
    completeness::Float64        # mean node completeness
end

"""One whole-NBS merge for the Rule-B report (a donor NBS folded into a receiver NBS)."""
struct RuleBMove
    donor_nbs::Int32
    receiver_nbs::Int32
    donor_vol::Float64
    receiver_vol::Float64
    frac_connected::Float64
    n_donor_skel_nodes::Int
end

"""One node reassignment for the volume-merge report (a node moved into another NBS)."""
struct NodeMove
    trial_node_id::Int
    from_nbs::Int32
    to_nbs::Int32
    overlap_ratio::Float64
    n_points::Int
    completeness::Float64
    agh::Float64
end

