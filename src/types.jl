"""
Core data structures and utilities for FLiP.jl.

Uses a lightweight PointCloudData struct instead of PointClouds.LAS
for minimal memory overhead and O(1) attribute operations.
"""

# ── Point cloud struct ─────────────────────────────────────────────

"""
    PointCloudData{T<:AbstractFloat}

Lightweight point cloud container storing coordinates as an N×3 matrix
and attributes as a `Dict{Symbol,Vector}`. Parametrized by coordinate
precision `T` (typically `Float32` or `Float64`).

# Fields
- `coords::Matrix{T}` — N×3 matrix of XYZ coordinates
- `attrs::Dict{Symbol,Vector}` — named attribute vectors (e.g., `:intensity`, `:classification`)
"""
mutable struct PointCloudData{T<:AbstractFloat}
    coords::Matrix{T}            # N×3
    attrs::Dict{Symbol,Vector}
end

const AbstractPointCloud = PointCloudData
const PointCloud = PointCloudData

# ── Type mappings for LAS extra byte dimensions ───────────────────

const _EXTRA_TYPE_TO_CODE = Dict{DataType,Int}(
    UInt8 => 1, Int8 => 2, UInt16 => 3, Int16 => 4,
    UInt32 => 5, Int32 => 6, UInt64 => 7, Int64 => 8,
    Float32 => 9, Float64 => 10,
)

const _LAS_EXTRA_SCALAR_TYPES = Dict(
    1 => UInt8, 2 => Int8, 3 => UInt16, 4 => Int16,
    5 => UInt32, 6 => Int32, 7 => UInt64, 8 => Int64,
    9 => Float32, 10 => Float64,
)

const _LAS_EXTRA_SCALAR_BYTES = Dict(
    1 => 1, 2 => 1, 3 => 2, 4 => 2,
    5 => 4, 6 => 4, 7 => 8, 8 => 8,
    9 => 4, 10 => 8,
)

# ── Accessors ──────────────────────────────────────────────────────

"""Return the number of points in the cloud."""
npoints(pc::PointCloud) = size(pc.coords, 1)

Base.length(pc::PointCloud) = npoints(pc)

"""Return coordinates as an N×3 matrix (element type matches coordinate precision)."""
coordinates(pc::PointCloud) = pc.coords

"""Return a copy of all scalar attributes."""
_all_attributes(pc::PointCloud) = copy(pc.attrs)

"""Check whether an attribute exists."""
hasattribute(pc::PointCloud, attr::Symbol) = haskey(pc.attrs, attr)

"""Get an attribute vector by name."""
getattribute(pc::PointCloud, attr::Symbol) = pc.attrs[attr]

"""Add or replace a scalar attribute and return a new point cloud."""
function addattribute(pc::PointCloud, name::Symbol, values::AbstractVector)
    length(values) == npoints(pc) || throw(ArgumentError("attribute :$name length $(length(values)) does not match point count $(npoints(pc))"))
    new_attrs = copy(pc.attrs)
    new_attrs[name] = collect(values)
    return PointCloudData(pc.coords, new_attrs)
end

"""Delete an attribute and return a new point cloud."""
function deleteattribute(pc::PointCloud, name::Symbol)
    haskey(pc.attrs, name) || return pc
    new_attrs = copy(pc.attrs)
    delete!(new_attrs, name)
    return PointCloudData(pc.coords, new_attrs)
end

"""Add or replace a scalar attribute. Returns a new point cloud (original is not mutated)."""
function setattribute!(pc::PointCloud, attr::Symbol, values::Vector)
    return addattribute(pc, attr, values)
end

# ── Subsetting ─────────────────────────────────────────────────────

function Base.getindex(pc::PointCloud, inds::AbstractVector{<:Integer})
    new_attrs = Dict{Symbol,Vector}(k => v[inds] for (k, v) in pc.attrs)
    return PointCloudData(pc.coords[inds, :], new_attrs)
end

# ── Coordinate replacement ─────────────────────────────────────────

function _replace_coordinates(pc::PointCloud, new_coords::AbstractMatrix{<:Real})
    n = size(new_coords, 1)
    n == npoints(pc) || throw(ArgumentError("new_coords row count must match number of points"))
    size(new_coords, 2) == 3 || throw(ArgumentError("new_coords must be N×3"))
    T = eltype(pc.coords)
    return PointCloudData(T.(new_coords), copy(pc.attrs))
end

# ── Summary functions ──────────────────────────────────────────────

function bounds(pc::PointCloud)
    c = pc.coords
    return (
        minimum(c[:, 1]), maximum(c[:, 1]),
        minimum(c[:, 2]), maximum(c[:, 2]),
        minimum(c[:, 3]), maximum(c[:, 3]),
    )
end

center(pc::PointCloud) = vec(mean(pc.coords, dims=1))

# ── Display ────────────────────────────────────────────────────────

function Base.show(io::IO, pc::PointCloudData)
    print(io, "PointCloudData($(npoints(pc)) points, $(length(pc.attrs)) attributes)")
end

function Base.show(io::IO, ::MIME"text/plain", pc::PointCloudData)
    println(io, "PointCloudData")
    println(io, "  Points: $(npoints(pc))")
    attr_names = sort(collect(keys(pc.attrs)), by=string)
    print(io, "  Attributes: $(join(attr_names, ", "))")
end

# ── Point cloud metadata struct ───────────────────────────────────

"""
    PointCloudMetadata

Lightweight container for point cloud file metadata, read without loading
point data. Covers both LAS/LAZ and E57 formats.

# Fields
- `path::String` — file path
- `format::String` — `"LAS"`, `"LAZ"`, or `"E57"`
- `point_count::Int` — number of points
- `bounds_min::Vector{Float64}` — `[xmin, ymin, zmin]`
- `bounds_max::Vector{Float64}` — `[xmax, ymax, zmax]`
- `scan_count::Int` — 1 for LAS/LAZ, N for E57
- `scan_index::Int` — 0-based scan index, -1 when representing all scans
- `version::String` — LAS version string, `""` for E57
- `point_format::Int` — LAS point format ID, -1 for E57
- `scales::Vector{Float64}` — LAS `[sx, sy, sz]`, empty for E57
- `offsets::Vector{Float64}` — LAS `[ox, oy, oz]`, empty for E57
- `translation::Vector{Float64}` — E57 per-scan `[tx, ty, tz]`, empty for LAS
- `rotation::Matrix{Float64}` — E57 per-scan 3×3 rotation, identity for LAS
- `extra::Dict{String,Any}` — format-specific extras
"""
mutable struct PointCloudMetadata
    path::String
    format::String
    point_count::Int
    bounds_min::Vector{Float64}
    bounds_max::Vector{Float64}
    scan_count::Int
    scan_index::Int
    version::String
    point_format::Int
    scales::Vector{Float64}
    offsets::Vector{Float64}
    translation::Vector{Float64}
    rotation::Matrix{Float64}
    extra::Dict{String,Any}
end

function Base.show(io::IO, m::PointCloudMetadata)
    print(io, "PointCloudMetadata($(m.format), $(m.point_count) points)")
end

function Base.show(io::IO, ::MIME"text/plain", m::PointCloudMetadata)
    println(io, "PointCloudMetadata")
    println(io, "  Path: $(m.path)")
    println(io, "  Format: $(m.format)")
    println(io, "  Points: $(m.point_count)")
    println(io, "  Bounds: $(m.bounds_min) — $(m.bounds_max)")
    if m.format == "E57"
        println(io, "  Scan: $(m.scan_index) of $(m.scan_count)")
        if !isempty(m.translation)
            print(io, "  Translation: $(m.translation)")
        end
    else
        print(io, "  Version: $(m.version), Point Format: $(m.point_format)")
    end
end
