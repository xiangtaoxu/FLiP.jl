"""
Input/Output functions for LAS/LAZ/E57 point cloud formats.

LAS/LAZ: via laspy (Python, called through PythonCall.jl)
E57:     via pye57 (Python, called through PythonCall.jl)
"""

using PythonCall

# ── Lazy Python module references ──────────────────────────────────

const _laspy = PythonCall.pynew()
const _pye57 = PythonCall.pynew()
const _np    = PythonCall.pynew()

function _ensure_np()
    if PythonCall.pyisnull(_np)
        PythonCall.pycopy!(_np, pyimport("numpy"))
    end
end

function _ensure_laspy()
    if PythonCall.pyisnull(_laspy)
        PythonCall.pycopy!(_laspy, pyimport("laspy"))
        _ensure_np()
    end
end

function _ensure_pye57()
    if PythonCall.pyisnull(_pye57)
        PythonCall.pycopy!(_pye57, pyimport("pye57"))
        _ensure_np()
    end
end

# ── Numpy ↔ Julia conversion helpers ──────────────────────────────

const _JULIA_TO_NUMPY_DTYPE = Dict{DataType,String}(
    Float64 => "float64", Float32 => "float32",
    UInt8 => "uint8", Int8 => "int8",
    UInt16 => "uint16", Int16 => "int16",
    UInt32 => "uint32", Int32 => "int32",
    UInt64 => "uint64", Int64 => "int64",
)

"""Convert a Python numpy array to a Julia Vector{T} via direct buffer copy."""
function _numpy_to_jl(py_arr, ::Type{T}) where T
    dtype_str = get(_JULIA_TO_NUMPY_DTYPE, T, "float64")
    arr = _np.ascontiguousarray(py_arr, dtype=_np.dtype(pystr(dtype_str)))
    n = pyconvert(Int, pybuiltins.len(arr))
    out = Vector{T}(undef, n)
    buf = pyconvert(Vector{UInt8}, arr.tobytes())
    copyto!(reinterpret(UInt8, out), buf)
    return out
end

"""Convert a Julia Vector to a numpy array."""
function _jl_to_numpy(vec::Vector{T}) where T
    return _np.array(Py(vec), copy=true)
end

# ── LAS field name mappings ────────────────────────────────────────

# FLiP attr name → (laspy field name, Julia type)
const _LAS_STD_FIELDS = Dict{Symbol,Tuple{String,DataType}}(
    :intensity          => ("intensity",          UInt16),
    :classification     => ("classification",     UInt8),
    :return_number      => ("return_number",      UInt8),
    :number_of_returns  => ("number_of_returns",  UInt8),
    :gps_time           => ("gps_time",           Float64),
    :scan_angle         => ("scan_angle",         Int16),
    :user_data          => ("user_data",          UInt8),
    :source_id          => ("point_source_id",    UInt16),
)

# Reverse: laspy field name → FLiP attr name
const _LASPY_TO_FLIP = Dict{String,Symbol}(v[1] => k for (k, v) in _LAS_STD_FIELDS)

# ── LAS/LAZ (via laspy) ───────────────────────────────────────────

"""
    read_las(path::AbstractString) -> PointCloud

Read a LAS or LAZ file and return a `PointCloud` object.
"""
function read_las(path::AbstractString)
    isfile(path) || error("LAS/LAZ file not found: $path")
    _ensure_laspy()

    las = _laspy.read(path)

    # Coordinates
    x = _numpy_to_jl(las.x, Float64)
    y = _numpy_to_jl(las.y, Float64)
    z = _numpy_to_jl(las.z, Float64)
    n = length(x)
    coords = Matrix{Float64}(undef, n, 3)
    @inbounds for i in 1:n
        coords[i, 1] = x[i]
        coords[i, 2] = y[i]
        coords[i, 3] = z[i]
    end

    attrs = Dict{Symbol,Vector}()

    # Standard LAS fields
    dim_names = pyconvert(Vector{String}, las.point_format.dimension_names)
    for (flip_name, (laspy_name, T)) in _LAS_STD_FIELDS
        if laspy_name in dim_names
            try
                arr = pygetattr(las, laspy_name)
                attrs[flip_name] = _numpy_to_jl(arr, T)
            catch
            end
        end
    end

    # Extra dimensions
    try
        extra_names = pyconvert(Vector{String}, las.point_format.extra_dimension_names)
        for name in extra_names
            arr = pygetattr(las, name)
            # Detect type from numpy dtype
            dtype_name = pyconvert(String, arr.dtype.name)
            T = get(_NUMPY_DTYPE_TO_JULIA, dtype_name, Float64)
            attrs[Symbol(name)] = _numpy_to_jl(arr, T)
        end
    catch
    end

    return PointCloudData(coords, attrs)
end

const read_laz = read_las

const _NUMPY_DTYPE_TO_JULIA = Dict{String,DataType}(
    "float64" => Float64, "float32" => Float32,
    "uint8" => UInt8, "int8" => Int8,
    "uint16" => UInt16, "int16" => Int16,
    "uint32" => UInt32, "int32" => Int32,
    "uint64" => UInt64, "int64" => Int64,
)

"""
    write_las(path::AbstractString, pc::PointCloud)

Write a point cloud to a LAS or LAZ file (LAZ requires lazrs backend).
"""
function write_las(path::AbstractString, pc::PointCloud)
    n = npoints(pc)
    _ensure_laspy()

    header = _laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = _np.array(Py([0.0, 0.0, 0.0]))
    header.scales = _np.array(Py([1e-6, 1e-6, 1e-6]))

    las = _laspy.LasData(header)

    # Coordinates
    coords = coordinates(pc)
    las.x = _jl_to_numpy(Vector{Float64}(coords[:, 1]))
    las.y = _jl_to_numpy(Vector{Float64}(coords[:, 2]))
    las.z = _jl_to_numpy(Vector{Float64}(coords[:, 3]))

    # Standard LAS fields
    std_laspy_names = Set(v[1] for v in values(_LAS_STD_FIELDS))
    for (flip_name, (laspy_name, T)) in _LAS_STD_FIELDS
        if haskey(pc.attrs, flip_name)
            vals = pc.attrs[flip_name]
            pysetattr(las, laspy_name, _jl_to_numpy(Vector{T}(vals)))
        end
    end

    # Extra dimensions (non-standard attributes)
    for (name, vals) in pc.attrs
        haskey(_LAS_STD_FIELDS, name) && continue
        T = eltype(vals)
        dtype_str = get(_JULIA_TO_NUMPY_DTYPE, T, nothing)
        isnothing(dtype_str) && continue
        np_dtype = _np.dtype(pystr(dtype_str))
        las.add_extra_dim(_laspy.ExtraBytesParams(name=string(name), type=np_dtype))
        pysetattr(las, string(name), _jl_to_numpy(Vector{T}(vals)))
    end

    las.write(path)
    return nothing
end

const write_laz = write_las

# ── E57 (via pye57) ───────────────────────────────────────────────

"""Read a single scan from an open pye57.E57 object. Returns (coords, attrs)."""
function _read_e57_scan(e57_obj, scan_index::Int)
    data = e57_obj.read_scan(scan_index, ignore_missing_fields=true)

    # Coordinates (always present)
    x = _numpy_to_jl(data["cartesianX"], Float64)
    y = _numpy_to_jl(data["cartesianY"], Float64)
    z = _numpy_to_jl(data["cartesianZ"], Float64)
    n = length(x)
    coords = Matrix{Float64}(undef, n, 3)
    @inbounds for i in 1:n
        coords[i, 1] = x[i]
        coords[i, 2] = y[i]
        coords[i, 3] = z[i]
    end

    attrs = Dict{Symbol,Vector}()

    # Intensity: E57 float → LAS UInt16 [0, 65535]
    has_intensity = pyconvert(Bool, pycontains(data, "intensity"))
    if has_intensity
        int_f = _numpy_to_jl(data["intensity"], Float64)
        imax = maximum(int_f)
        imin = minimum(int_f)
        if imax > 1.0
            scale = imax > imin ? 65535.0 / (imax - imin) : 1.0
            attrs[:intensity] = round.(UInt16, clamp.((int_f .- imin) .* scale, 0.0, 65535.0))
        else
            attrs[:intensity] = round.(UInt16, clamp.(int_f .* 65535.0, 0.0, 65535.0))
        end
    end

    # RGB color
    has_red   = pyconvert(Bool, pycontains(data, "colorRed"))
    has_green = pyconvert(Bool, pycontains(data, "colorGreen"))
    has_blue  = pyconvert(Bool, pycontains(data, "colorBlue"))
    if has_red && has_green && has_blue
        attrs[:color_red]   = _numpy_to_jl(data["colorRed"],   UInt16)
        attrs[:color_green] = _numpy_to_jl(data["colorGreen"], UInt16)
        attrs[:color_blue]  = _numpy_to_jl(data["colorBlue"],  UInt16)
    end

    return coords, attrs
end

"""Merge attribute dictionaries from multiple scans by concatenation."""
function _merge_scan_attrs(all_attrs::Vector{Dict{Symbol,Vector}})
    isempty(all_attrs) && return Dict{Symbol,Vector}()
    length(all_attrs) == 1 && return all_attrs[1]

    common_keys = Set(keys(all_attrs[1]))
    for a in all_attrs[2:end]
        intersect!(common_keys, keys(a))
    end

    merged = Dict{Symbol,Vector}()
    for k in common_keys
        merged[k] = vcat([a[k] for a in all_attrs]...)
    end
    return merged
end

"""
    _read_e57_to_raw(path::AbstractString; scan_index::Int=-1) -> (Matrix{Float64}, Dict{Symbol,Vector})

Read an E57 file and return raw (coords, attrs) without constructing a PointCloud.
Useful for subsampling before building the PointCloud object.
"""
function _read_e57_to_raw(path::AbstractString; scan_index::Int=-1)
    isfile(path) || error("E57 file not found: $path")
    _ensure_pye57()

    e57_obj = _pye57.E57(path)
    try
        n_scans = pyconvert(Int, e57_obj.scan_count)

        if scan_index >= 0
            scan_index < n_scans || error("scan_index $scan_index out of range (file has $n_scans scans)")
            return _read_e57_scan(e57_obj, scan_index)
        else
            n_scans > 0 || error("E57 file contains no 3D scans: $path")
            all_coords = Vector{Matrix{Float64}}(undef, n_scans)
            all_attrs  = Vector{Dict{Symbol,Vector}}(undef, n_scans)
            for si in 0:(n_scans - 1)
                all_coords[si + 1], all_attrs[si + 1] = _read_e57_scan(e57_obj, si)
            end
            coords = vcat(all_coords...)
            attrs  = _merge_scan_attrs(all_attrs)
            return (coords, attrs)
        end
    finally
    end
end

"""
    read_e57(path::AbstractString; scan_index::Int=-1) -> PointCloud

Read an E57 file and return a `PointCloud` object.

By default all scans in the file are merged into a single point cloud.
Set `scan_index` (0-based) to read only a specific scan.

# Field Mapping
- XYZ coordinates are read as Float64.
- Intensity is scaled to LAS UInt16 [0, 65535].
- RGB colors are stored as attributes `:color_red`, `:color_green`, `:color_blue`.

# Example
```julia
pc = read_e57("scan.e57")
coords = coordinates(pc)
```
"""
function read_e57(path::AbstractString; scan_index::Int=-1)
    coords, attrs = _read_e57_to_raw(path; scan_index=scan_index)
    return PointCloudData(coords, attrs)
end

"""
    write_e57(path::AbstractString, pc::PointCloud)

Write a point cloud to an E57 file as a single scan.

# Field Mapping
- XYZ coordinates are written as Float64.
- `:intensity` (UInt16) is scaled to float [0, 1].
- `:color_red`, `:color_green`, `:color_blue` (UInt16) are written as RGB.

# Example
```julia
write_e57("output.e57", pc)
```
"""
function write_e57(path::AbstractString, pc::PointCloud)
    n = npoints(pc)
    n > 0 || error("Cannot write empty point cloud to E57")
    _ensure_pye57()

    coords = coordinates(pc)
    attrs = _all_attributes(pc)

    data = pydict()
    data["cartesianX"] = _jl_to_numpy(Vector{Float64}(coords[:, 1]))
    data["cartesianY"] = _jl_to_numpy(Vector{Float64}(coords[:, 2]))
    data["cartesianZ"] = _jl_to_numpy(Vector{Float64}(coords[:, 3]))

    if haskey(attrs, :intensity)
        int_f = Float64.(attrs[:intensity]) ./ 65535.0
        data["intensity"] = _jl_to_numpy(int_f)
    end

    if haskey(attrs, :color_red) && haskey(attrs, :color_green) && haskey(attrs, :color_blue)
        data["colorRed"]   = _jl_to_numpy(UInt16.(attrs[:color_red]))
        data["colorGreen"] = _jl_to_numpy(UInt16.(attrs[:color_green]))
        data["colorBlue"]  = _jl_to_numpy(UInt16.(attrs[:color_blue]))
    end

    e57_obj = _pye57.E57(path, mode="w")
    try
        e57_obj.write_scan_raw(data)
    finally
        e57_obj.close()
    end
    return nothing
end

# ── Master dispatch ────────────────────────────────────────────────

"""
    read_pc(path::AbstractString) -> PointCloud

Read a point cloud file, dispatching by file extension.
Supported: `.las`, `.laz`, `.e57`.
"""
function read_pc(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    if ext in (".las", ".laz")
        return read_las(path)
    elseif ext == ".e57"
        return read_e57(path)
    else
        error("Unsupported point cloud format: $ext (supported: .las, .laz, .e57)")
    end
end

"""
    write_pc(path::AbstractString, pc::PointCloud)

Write a point cloud file, dispatching by file extension.
Supported: `.las`, `.laz`, `.e57`.
"""
function write_pc(path::AbstractString, pc::PointCloud)
    ext = lowercase(splitext(path)[2])
    if ext in (".las", ".laz")
        return write_las(path, pc)
    elseif ext == ".e57"
        return write_e57(path, pc)
    else
        error("Unsupported point cloud write format: $ext (supported: .las, .laz, .e57)")
    end
end
