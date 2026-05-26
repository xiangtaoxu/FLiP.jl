"""
Core data structures and utilities for FLiP.jl.

Uses a lightweight `PointCloud` struct instead of `PointClouds.LAS` for
minimal memory overhead and O(1) attribute operations.
"""

# ── Point cloud struct ─────────────────────────────────────────────

"""
    PointCloud{T<:AbstractFloat}

Lightweight point cloud container storing coordinates as an N×3 matrix
and attributes as a `Dict{Symbol,Vector}`. Parametrized by coordinate
precision `T` (typically `Float32` or `Float64`).

# Fields
- `coords::Matrix{T}` — N×3 matrix of XYZ coordinates
- `attrs::Dict{Symbol,Vector}` — named attribute vectors (e.g., `:intensity`, `:classification`)
"""
mutable struct PointCloud{T<:AbstractFloat}
    coords::Matrix{T}            # N×3
    attrs::Dict{Symbol,Vector}
end

# ── Accessors ──────────────────────────────────────────────────────

"""Return the number of points in the cloud."""
npoints(pc::PointCloud) = size(pc.coords, 1)

Base.length(pc::PointCloud) = npoints(pc)

"""Return coordinates as an N×3 matrix (element type matches coordinate precision)."""
coordinates(pc::PointCloud) = pc.coords

"""
Return a shallow copy of all scalar attributes — the outer `Dict` is fresh,
but the inner `Vector`s are shared with `pc`. Safe for read-only uses and
for downstream `vcat`-style merging (which allocates new vectors); not safe
if a caller mutates the returned vectors in place.
"""
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
    return PointCloud(pc.coords, new_attrs)
end

"""Delete an attribute and return a new point cloud."""
function deleteattribute(pc::PointCloud, name::Symbol)
    haskey(pc.attrs, name) || return pc
    new_attrs = copy(pc.attrs)
    delete!(new_attrs, name)
    return PointCloud(pc.coords, new_attrs)
end

"""Add or replace a scalar attribute in place. Returns the (mutated) point cloud."""
function setattribute!(pc::PointCloud, attr::Symbol, values::AbstractVector)
    length(values) == npoints(pc) ||
        throw(ArgumentError("attribute :$attr length $(length(values)) does not match point count $(npoints(pc))"))
    pc.attrs[attr] = collect(values)
    return pc
end

# ── Subsetting ─────────────────────────────────────────────────────

function Base.getindex(pc::PointCloud, inds::AbstractVector{<:Integer})
    new_attrs = Dict{Symbol,Vector}(k => v[inds] for (k, v) in pc.attrs)
    return PointCloud(pc.coords[inds, :], new_attrs)
end

# ── Merging ────────────────────────────────────────────────────────

"""
    merge_pointclouds(coords_list, attrs_list) -> PointCloud

Concatenate a list of N×3 coordinate matrices and merge attribute dicts by
key intersection (attributes present in *all* clouds are kept; others are
dropped). With a single-element list, returns a PointCloud built from that
element without any concatenation.
"""
function merge_pointclouds(coords_list::AbstractVector{<:AbstractMatrix},
                           attrs_list::AbstractVector{<:Dict{Symbol,<:Vector}})
    length(coords_list) == length(attrs_list) ||
        throw(ArgumentError("coords_list and attrs_list must have the same length"))
    isempty(coords_list) && throw(ArgumentError("cannot merge an empty list"))

    if length(coords_list) == 1
        return PointCloud(coords_list[1], Dict{Symbol,Vector}(attrs_list[1]))
    end

    common_keys = Set(keys(attrs_list[1]))
    all_keys = Set(keys(attrs_list[1]))
    for a in @view attrs_list[2:end]
        intersect!(common_keys, keys(a))
        union!(all_keys, keys(a))
    end
    dropped = setdiff(all_keys, common_keys)
    isempty(dropped) ||
        @info "merge_pointclouds: dropping attributes missing from some scans: $(sort(collect(dropped); by=string))"
    merged_attrs = Dict{Symbol,Vector}()
    for k in common_keys
        merged_attrs[k] = vcat((a[k] for a in attrs_list)...)
    end
    merged_coords = vcat(coords_list...)
    return PointCloud(merged_coords, merged_attrs)
end

# ── Coordinate replacement ─────────────────────────────────────────

function _replace_coordinates(pc::PointCloud, new_coords::AbstractMatrix{<:Real})
    n = size(new_coords, 1)
    n == npoints(pc) || throw(ArgumentError("new_coords row count must match number of points"))
    size(new_coords, 2) == 3 || throw(ArgumentError("new_coords must be N×3"))
    T = eltype(pc.coords)
    return PointCloud(T.(new_coords), copy(pc.attrs))
end

# ── Summary functions ──────────────────────────────────────────────

function bounds(pc::PointCloud)
    c = pc.coords
    n = size(c, 1)
    n > 0 || throw(ArgumentError("bounds: empty point cloud"))
    xmin = xmax = c[1, 1]
    ymin = ymax = c[1, 2]
    zmin = zmax = c[1, 3]
    @inbounds for i in 2:n
        x = c[i, 1]; y = c[i, 2]; z = c[i, 3]
        x < xmin && (xmin = x); x > xmax && (xmax = x)
        y < ymin && (ymin = y); y > ymax && (ymax = y)
        z < zmin && (zmin = z); z > zmax && (zmax = z)
    end
    return (xmin, xmax, ymin, ymax, zmin, zmax)
end

center(pc::PointCloud) = vec(mean(pc.coords, dims=1))

# ── Display ────────────────────────────────────────────────────────

function Base.show(io::IO, pc::PointCloud)
    print(io, "PointCloud($(npoints(pc)) points, $(length(pc.attrs)) attributes)")
end

function Base.show(io::IO, ::MIME"text/plain", pc::PointCloud)
    println(io, "PointCloud")
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
