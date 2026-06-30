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

## Subsampling

```@docs
distance_subsample
```

## Filtering

```@docs
statistical_filter
grid_zmin_filter
upward_conic_filter
voxel_connected_component_filter
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
