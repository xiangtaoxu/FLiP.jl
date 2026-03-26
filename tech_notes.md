0. TODO list

* performance optimization for ground filtering (for v1.0)

* implement parallelization for all modules (for v2.0)


1. Ground filtering

* Previous FLIP uses Cloth Simulation Filter, which performs poorly near tree base

* In this version, we combine a local minimum z filter and a global upward conic filter to attain efficient and customizable ground segmentation. -- TODO - add a schematic figure in 2D

* This will allow for better results for fine-scale topographic features such as tree base

* The final segment_ground will first conduct voxel_connected_component_filter (to remove noise 'underground' points), and optionally statistical_filter (for MLS or ULS data that is super noisy), second conduct grid_zmin_filter -> get nearground poinds, third, conduct conic filtering -> remove stump points that remain after grid_zmin_filter.

2. Calculate Aboveground Height

* We use IDW interpolation to generate a dense evenly spaced ground point cloud. Then for each point find the closest ground point in XY plane and calculate Z distance difference. This is a good enough approximation for aboveground height and is much faster than getting accurate cloud to mesh height

3. Tree segmentation
* [TODO] add near ground point clusters as seed points. These can help to capture big diameter stumps. When all seed clusters are labeled, use lowest point as seed

* The overall philosophy is similar to the python version but split it into two steps - **Step 1**: find all non-branching segments through a greedy graph labeling algorithm that iteratively expands the largest connected `frontier` from a seed vertex. The underlying theory is that any (major) branching in the graph will lead to connectivity break during neighborhood expansion (TODO: add a schematic figure); This process will also generate proto node ids for step 2. **Step 2**: find linear connected segments (LCS) within the skeleton cloud generated from NBS nodes. This section helps to (a) prepare for QSM later, which is based on linear slicing and (b) rectify points in branching nodes.

* Step 1 is facilitated by a highly efficient and customized graph trasversal and labeling algorithm

Add _refine_branching in step 1 to correct the branching node after deciding which branch to proceed in frontier expansion. [add figures]

* [thought] refine prepare_seed to be used in each iteration?

-> Current status

* refine LCS along their major axis

* assemble LCS to trees, combine LCS if they connected by the end [need to further think]

4. QSM

* for each LCS, find center point along major axis (reuse results from step 3?)

* unwrap the LCS along the nonlinear center line

* conduct extrapolation

* estimate circumfirence and area for each center line point/node

* export the results as csv

* export results as primitives in CloudCompare?

5. Visualization

* visualize stand map

* visualize DBH and the profile

* visualize selected tree