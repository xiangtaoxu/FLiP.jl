# API Reference

```@meta
CurrentModule = FLiP
```

## Types

```@docs
AbstractPointCloud
PointCloud
```

## Utility Functions

```@docs
npoints
coordinates
hasattribute
getattribute
setattribute!
bounds
center
```

## I/O Functions

### LAS/LAZ Format

```@docs
read_las
read_laz
write_las
write_laz
```

### PCD Format

```@docs
read_pcd
write_pcd
```

### PTX Format

```@docs
read_ptx
```

## Subsampling

```@docs
voxel_grid_downsample
voxel_grid_downsample_indices
distance_subsample
distance_subsample_indices
```

## Filtering

```@docs
statistical_filter
statistical_filter_indices
grid_zmin_filter_indices
upward_conic_filter_indices
```

## Transformations

```@docs
translate
translate!
scale
rotate
center_at_origin
transform
apply_transform
apply_transform!
bounding_box_crop
```
