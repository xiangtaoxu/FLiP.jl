#!/usr/bin/env julia

"""
Quick demo of FLiP.jl functionality

This script demonstrates the core features of FLiP.jl:
- Reading LAS/LAZ files
- Subsampling
- Filtering
- Transformations
- Saving results
"""

using FLiP

println("=" ^ 70)
println("FLiP.jl Demo - Forest Lidar Processing in Julia")
println("=" ^ 70)
println()

# Check if test file exists
test_file = "MLS-Corson-test.laz"
if !isfile(test_file)
    println("⚠️  Test file '$test_file' not found")
    println("Please run this script from the Flip_julia directory")
    exit(1)
end

# 1. Read point cloud
println("📂 Reading LAZ file...")
pc = read_las(test_file)
println("   ✓ Loaded $(length(pc)) points")
bbox = bounds(pc)
println("   ✓ Bounds: X[$(round(bbox[1], digits=2)), $(round(bbox[2], digits=2))]")
println("             Y[$(round(bbox[3], digits=2)), $(round(bbox[4], digits=2))]")
println("             Z[$(round(bbox[5], digits=2)), $(round(bbox[6], digits=2))]")
println()

# 2. Minimum distance subsampling
println("🔽 Subsampling with minimum distance (10cm)...")
pc_sub = distance_subsample(pc, 0.1)
reduction = round((1 - length(pc_sub)/length(pc)) * 100, digits=1)
println("   ✓ Subsampled to $(length(pc_sub)) points ($(reduction)% reduction)")
println()

# 3. Statistical filtering  
println("🧹 Removing outliers with statistical filter...")
pc_clean = statistical_filter(pc_sub, 10, 2.0)  # k_neighbors=10, n_sigma=2.0
outliers = length(pc_sub) - length(pc_clean)
println("   ✓ Removed $outliers outliers")
println("   ✓ Clean point cloud: $(length(pc_clean)) points")
println()

# 4. Ground filtering
println("🌲 Separating ground and vegetation...")
coords = coordinates(pc_clean)
seed_idx = grid_zmin_filter_indices(coords, 1.0)
ground_local = upward_conic_filter_indices(coords[seed_idx, :], 45.0)
ground_idx = seed_idx[ground_local]
vegetation_idx = sort(setdiff(1:length(pc_clean), ground_idx))
ground = pc_clean[ground_idx]
vegetation = pc_clean[vegetation_idx]
println("   ✓ Ground points: $(length(ground))")
println("   ✓ Vegetation points: $(length(vegetation))")
println()

# 5. Transformation
println("🔄 Applying transformations...")
pc_centered = center_at_origin(vegetation)
centroid = center(pc_centered)
println("   ✓ Centered at origin: ($(round(centroid[1], digits=6)), $(round(centroid[2], digits=6)), $(round(centroid[3], digits=6)))")

pc_translated = translate(pc_centered, 100.0, 200.0, 0.0)
new_center = center(pc_translated)
println("   ✓ Translated: ($(round(new_center[1], digits=2)), $(round(new_center[2], digits=2)), $(round(new_center[3], digits=2)))")
println()

# 6. Height filtering
println("📏 Filtering by height...")
pc_subset = height_filter(vegetation, 0.5, 10.0)
println("   ✓ Kept points between 0.5m and 10.0m: $(length(pc_subset)) points")
println()

# 7. Summary statistics
println("📊 Summary:")
println("   Original points:     $(length(pc))")
println("   After subsampling:   $(length(pc_sub))")
println("   After filtering:     $(length(pc_clean))")
println("   Ground:              $(length(ground))")
println("   Vegetation:          $(length(vegetation))")
println("   Height filtered:     $(length(pc_subset))")
println()

println("=" ^ 70)
println("✨ Demo completed successfully!")
println("=" ^ 70)
