"""
Input/Output functions for LAS/LAZ point cloud formats.
"""

"""
    read_las(path::AbstractString) -> PointCloud

Read a LAS or LAZ file and return a `PointClouds.LAS` object.
"""
function read_las(path::AbstractString)
    return PointClouds.LAS(path)
end

const read_laz = read_las

"""
    write_las(path::AbstractString, pc::PointCloud)

Write a point cloud to LAS/LAZ.

Note: PointClouds.jl currently supports writing `.las` (uncompressed).
"""
function write_las(path::AbstractString, pc::PointCloud)
    PointClouds.write(path, pc)
    return nothing
end

const write_laz = write_las
