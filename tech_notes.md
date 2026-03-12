1. Ground filtering

* Previous FLIP uses Cloth Simulation Filter, which performs poorly near tree base

* In this version, we combine a local minimum z filter and a global upward conic filter to attain efficient and customizable ground segmentation. -- TODO - add a schematic figure in 2D

* This will allow for better results for fine-scale topographic features such as tree base

* The final segment_ground will first conduct voxel_connected_component_filter (to remove noise 'underground' points), and optionally statistical_filter (for MLS or ULS data that is super noisy), second conduct grid_zmin_filter -> get nearground poinds, third, conduct conic filtering -> remove stump points that remain after grid_zmin_filter 