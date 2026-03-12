"""
Core data structures and utilities built on PointClouds.jl.
"""

const AbstractPointCloud = PointClouds.LAS
const PointCloud = PointClouds.LAS

const _STD_ACCESSORS = Dict{Symbol,Symbol}(
    :intensity => :intensity,
    :classification => :classification,
    :return_number => :return_number,
    :number_of_returns => :return_count,
    :scan_angle => :scan_angle,
    :user_data => :user_data,
    :source_id => :source_id,
    :scanner_channel => :scanner_channel,
    :gps_time => :gps_time,
    :is_key_point => :is_key_point,
    :is_overlap => :is_overlap,
    :is_synthetic => :is_synthetic,
    :is_withheld => :is_withheld,
    :is_left_to_right => :is_left_to_right,
    :is_right_to_left => :is_right_to_left,
    :is_edge_of_line => :is_edge_of_line,
)

const _STD_POINT_FIELDS = Dict{Symbol,Symbol}(
    :intensity => :intensity,
    :classification => :classification,
    :return_number => :return_number,
    :number_of_returns => :return_count,
    :scan_angle => :scan_angle,
    :user_data => :user_data,
    :source_id => :source_id,
    :scanner_channel => :scanner_channel,
    :gps_time => :gps_time,
    :is_key_point => :is_key_point,
    :is_overlap => :is_overlap,
    :is_synthetic => :is_synthetic,
    :is_withheld => :is_withheld,
    :is_left_to_right => :is_left_to_right,
    :is_right_to_left => :is_right_to_left,
    :is_edge_of_line => :is_edge_of_line,
)

const _EXTRA_TYPE_TO_CODE = Dict{DataType,Int}(
    UInt8 => 1,
    Int8 => 2,
    UInt16 => 3,
    Int16 => 4,
    UInt32 => 5,
    Int32 => 6,
    UInt64 => 7,
    Int64 => 8,
    Float32 => 9,
    Float64 => 10,
)

const _LAS_EXTRA_SCALAR_TYPES = Dict(
    1 => UInt8,
    2 => Int8,
    3 => UInt16,
    4 => Int16,
    5 => UInt32,
    6 => Int32,
    7 => UInt64,
    8 => Int64,
    9 => Float32,
    10 => Float64,
)

const _LAS_EXTRA_SCALAR_BYTES = Dict(
    1 => 1,
    2 => 1,
    3 => 2,
    4 => 2,
    5 => 4,
    6 => 4,
    7 => 8,
    8 => 8,
    9 => 4,
    10 => 8,
)

function _las_extra_nbytes(data_type::Int, options::Int)
    if 1 <= data_type <= 10
        return _LAS_EXTRA_SCALAR_BYTES[data_type]
    elseif 11 <= data_type <= 20
        return 2 * _LAS_EXTRA_SCALAR_BYTES[data_type - 10]
    elseif 21 <= data_type <= 30
        return 3 * _LAS_EXTRA_SCALAR_BYTES[data_type - 20]
    elseif data_type == 0
        return options
    else
        return 0
    end
end

function _las_extra_scalar_descriptors(las::PointCloud)
    descriptors = NamedTuple{(:name, :data_type, :byte_offset),Tuple{Symbol,Int,Int}}[]
    byte_offset = 1

    for vlr in las.vlrs
        if vlr.user_id == "LASF_Spec" && vlr.record_id == 4
            data = vlr.data
            desc_len = 192
            n_desc = length(data) ÷ desc_len

            for i in 0:(n_desc - 1)
                base = i * desc_len + 1
                data_type = Int(data[base + 2])
                options = Int(data[base + 3])
                raw_name = data[(base + 4):(base + 35)]

                name_chars = UInt8[]
                for b in raw_name
                    b == 0x00 && break
                    push!(name_chars, b)
                end

                nbytes = _las_extra_nbytes(data_type, options)
                if !isempty(name_chars) && haskey(_LAS_EXTRA_SCALAR_TYPES, data_type)
                    name = Symbol(String(name_chars))
                    push!(descriptors, (name=name, data_type=data_type, byte_offset=byte_offset))
                end

                byte_offset += nbytes
            end
        end
    end

    return descriptors
end

function _decode_extra_scalar_dimension(extra::AbstractVector, byte_offset::Int, data_type::Int)
    T = _LAS_EXTRA_SCALAR_TYPES[data_type]
    n = length(extra)
    values = Vector{T}(undef, n)
    nbytes = _LAS_EXTRA_SCALAR_BYTES[data_type]

    @inbounds for i in 1:n
        bytes = extra[i]
        raw = UInt8[bytes[byte_offset + j - 1] for j in 1:nbytes]
        values[i] = read(IOBuffer(raw), T)
    end

    return values
end

"""Return coordinates as an N×3 Float64 matrix."""
function coordinates(pc::PointCloud)
    xyz = PointClouds.coordinates(pc, :)
    n = length(xyz)
    out = Matrix{Float64}(undef, n, 3)
    @inbounds for i in 1:n
        x, y, z = xyz[i]
        out[i, 1] = x
        out[i, 2] = y
        out[i, 3] = z
    end
    return out
end

"""Return all scalar attributes available from LAS standard fields and extra dimensions."""
function _all_attributes(pc::PointCloud)
    n = length(pc)
    attrs = Dict{Symbol,Vector}()

    for (name, accessor) in _STD_ACCESSORS
        try
            vals = getproperty(PointClouds, accessor)(pc, :)
            if length(vals) == n
                vec_vals = collect(vals)
                # PointClouds can return Vector{Missing} for unavailable optional fields.
                if !(eltype(vec_vals) <: Missing)
                    attrs[name] = vec_vals
                end
            end
        catch
        end
    end

    try
        desc = _las_extra_scalar_descriptors(pc)
        if !isempty(desc)
            extra = PointClouds.extra_bytes(pc, :)
            for d in desc
                vals = _decode_extra_scalar_dimension(extra, d.byte_offset, d.data_type)
                length(vals) == n && (attrs[d.name] = vals)
            end
        end
    catch
    end

    return attrs
end

hasattribute(pc::PointCloud, attr::Symbol) = haskey(_all_attributes(pc), attr)
getattribute(pc::PointCloud, attr::Symbol) = _all_attributes(pc)[attr]

function _build_extra_bytes_vlr(custom_desc)
    data = UInt8[]
    for d in custom_desc
        buf = fill(UInt8(0), 192)
        buf[3] = UInt8(d.data_type)
        name_bytes = codeunits(String(d.name))
        ncopy = min(length(name_bytes), 32)
        for i in 1:ncopy
            buf[4 + i] = name_bytes[i]
        end
        append!(data, buf)
    end
    return PointClouds.IO.VariableLengthRecord("LASF_Spec", UInt16(4), data, "")
end

function _pack_extra_bytes(custom_desc, attrs, n::Int)
    total_bytes = isempty(custom_desc) ? 0 : sum(_LAS_EXTRA_SCALAR_BYTES[d.data_type] for d in custom_desc)
    total_bytes == 0 && return nothing

    packed = Vector{NTuple{total_bytes,UInt8}}(undef, n)
    @inbounds for i in 1:n
        row = UInt8[]
        for d in custom_desc
            T = _LAS_EXTRA_SCALAR_TYPES[d.data_type]
            v = convert(T, attrs[d.name][i])
            io = IOBuffer()
            write(io, v)
            append!(row, take!(io))
        end
        packed[i] = ntuple(j -> row[j], total_bytes)
    end
    return packed
end

function _build_las_with_attributes(template::PointCloud, coords::AbstractMatrix{<:Real}, attrs::Dict{Symbol,Vector})
    n = size(coords, 1)
    size(coords, 2) == 3 || throw(ArgumentError("coords must be N×3"))

    fields = Dict{Symbol,Any}(
        :x => Float64.(coords[:, 1]),
        :y => Float64.(coords[:, 2]),
        :z => Float64.(coords[:, 3]),
    )

    for (name, point_field) in _STD_POINT_FIELDS
        if haskey(attrs, name)
            vals = attrs[name]
            length(vals) == n || throw(ArgumentError("attribute :$name has wrong length"))
            # Skip unavailable optional metadata vectors represented as missing.
            if !(eltype(vals) <: Missing)
                fields[point_field] = vals
            end
        end
    end

    custom_names = sort([k for k in keys(attrs) if !haskey(_STD_POINT_FIELDS, k)], by=string)
    custom_desc = NamedTuple{(:name, :data_type),Tuple{Symbol,Int}}[]
    for name in custom_names
        vals = attrs[name]
        length(vals) == n || throw(ArgumentError("attribute :$name has wrong length"))
        T = eltype(vals)
        haskey(_EXTRA_TYPE_TO_CODE, T) || throw(ArgumentError("attribute :$name type $T is unsupported for LAS extra scalar fields"))
        push!(custom_desc, (name=name, data_type=_EXTRA_TYPE_TO_CODE[T]))
    end

    extra_bytes = _pack_extra_bytes(custom_desc, attrs, n)
    isnothing(extra_bytes) || (fields[:extra_bytes] = extra_bytes)

    points_nt = NamedTuple{Tuple(keys(fields))}(values(fields))
    new_las = PointClouds.LAS(points_nt; coord_scale=template.coord_scale, coord_offset=template.coord_offset)

    base_vlrs = [v for v in template.vlrs if !(v.user_id == "LASF_Spec" && v.record_id == 4)]
    if !isempty(custom_desc)
        push!(base_vlrs, _build_extra_bytes_vlr(custom_desc))
    end

    return PointClouds.LAS(
        new_las.points,
        base_vlrs,
        template.extra_data,
        new_las.coord_scale,
        new_las.coord_offset,
        new_las.coord_min,
        new_las.coord_max,
        new_las.return_counts,
        new_las.version,
        template.source_id,
        template.project_id,
        template.system_id,
        template.software_id,
        template.creation_date,
        template.has_adjusted_standard_gps_time,
        template.has_internal_waveform,
        template.has_external_waveform,
        template.has_synthetic_return_numbers,
        template.has_well_known_text,
    )
end

"""Add or replace a scalar attribute and return a new point cloud."""
function addattribute(pc::PointCloud, name::Symbol, values::AbstractVector)
    length(values) == length(pc) || throw(ArgumentError("attribute length does not match point count"))
    attrs = _all_attributes(pc)
    attrs[name] = collect(values)
    return _build_las_with_attributes(pc, coordinates(pc), attrs)
end

"""Delete an attribute and return a new point cloud."""
function deleteattribute(pc::PointCloud, name::Symbol)
    haskey(_STD_POINT_FIELDS, name) && throw(ArgumentError("cannot delete core LAS standard field :$name"))
    attrs = _all_attributes(pc)
    haskey(attrs, name) || return pc
    delete!(attrs, name)
    return _build_las_with_attributes(pc, coordinates(pc), attrs)
end

function setattribute!(pc::PointCloud, attr::Symbol, values::Vector)
    return addattribute(pc, attr, values)
end

npoints(pc::PointCloud) = length(pc)

function bounds(pc::PointCloud)
    c = coordinates(pc)
    return (
        minimum(c[:, 1]), maximum(c[:, 1]),
        minimum(c[:, 2]), maximum(c[:, 2]),
        minimum(c[:, 3]), maximum(c[:, 3]),
    )
end

center(pc::PointCloud) = vec(mean(coordinates(pc), dims=1))

function _subset_las(pc::PointCloud, inds)
    points = pc.points[inds]
    return PointClouds.LAS(
        points,
        pc.vlrs,
        pc.extra_data,
        pc.coord_scale,
        pc.coord_offset,
        pc.coord_min,
        pc.coord_max,
        pc.return_counts,
        pc.version,
        pc.source_id,
        pc.project_id,
        pc.system_id,
        pc.software_id,
        pc.creation_date,
        pc.has_adjusted_standard_gps_time,
        pc.has_internal_waveform,
        pc.has_external_waveform,
        pc.has_synthetic_return_numbers,
        pc.has_well_known_text,
    )
end

Base.getindex(pc::PointCloud, inds::AbstractVector{<:Integer}) = _subset_las(pc, inds)

function _replace_coordinates(pc::PointCloud, new_coords::AbstractMatrix{<:Real})
    n = size(new_coords, 1)
    n == length(pc) || throw(ArgumentError("new_coords row count must match number of points"))
    size(new_coords, 2) == 3 || throw(ArgumentError("new_coords must be N×3"))

    fields = Dict{Symbol,Any}(
        :x => Float64.(new_coords[:, 1]),
        :y => Float64.(new_coords[:, 2]),
        :z => Float64.(new_coords[:, 3]),
    )

    for (name, accessor) in _STD_ACCESSORS
        try
            vals = getproperty(PointClouds, accessor)(pc, :)
            fields[name == :number_of_returns ? :return_count : name] = vals
        catch
        end
    end

    try
        fields[:extra_bytes] = PointClouds.extra_bytes(pc, :)
    catch
    end

    points_nt = NamedTuple{Tuple(keys(fields))}(values(fields))
    pt_type = getfield(PointClouds.IO, :point_record_type)(eltype(pc.points))
    new_las = PointClouds.LAS(pt_type, points_nt; coord_scale=pc.coord_scale, coord_offset=pc.coord_offset)

    return PointClouds.LAS(
        new_las.points,
        pc.vlrs,
        pc.extra_data,
        pc.coord_scale,
        pc.coord_offset,
        pc.coord_min,
        pc.coord_max,
        pc.return_counts,
        pc.version,
        pc.source_id,
        pc.project_id,
        pc.system_id,
        pc.software_id,
        pc.creation_date,
        pc.has_adjusted_standard_gps_time,
        pc.has_internal_waveform,
        pc.has_external_waveform,
        pc.has_synthetic_return_numbers,
        pc.has_well_known_text,
    )
end
