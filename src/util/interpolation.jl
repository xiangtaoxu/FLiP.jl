"""
2D spatial interpolation helpers.

Functions:
- `interpolate_idw(known_xy, known_z, query_xy; k, power, max_distance, exact_tol)`
        — pointwise inverse-distance-weighted interpolation
"""

"""
    interpolate_idw(known_xy::AbstractMatrix{<:Real},
                    known_z::AbstractVector{<:Real},
                    query_xy::AbstractMatrix{<:Real};
                    k::Int=8, power::Real=2.0,
                    max_distance::Real=Inf,
                    exact_tol::Real=eps(Float64)) -> Vector{Float64}

Inverse-distance-weighted interpolation of `known_z` (length M) at each row of
`query_xy` (N×2). Builds a KDTree on `known_xy` (M×2, transposed internally to
2×M as required by NearestNeighbors.jl) and, for each query, takes the k
nearest known points with weight ∝ 1/d^power.

Behavior:
- When a query is within `exact_tol` of a known point, returns the known z.
- When the nearest known point is farther than `max_distance`, returns NaN.
- When a query coordinate is non-finite, returns NaN.

# Arguments
- `known_xy`: M×2 matrix of known sample XY positions
- `known_z`: length-M vector of known sample values
- `query_xy`: N×2 matrix of query XY positions
- `k`: number of nearest neighbors per query (clamped to M; must be ≥ 1)
- `power`: IDW exponent (must be > 0)
- `max_distance`: NaN-out queries whose nearest known sample is farther
- `exact_tol`: distance below which a query is treated as coincident
"""
function interpolate_idw(known_xy::AbstractMatrix{<:Real},
                         known_z::AbstractVector{<:Real},
                         query_xy::AbstractMatrix{<:Real};
                         k::Int=8, power::Real=2.0,
                         max_distance::Real=Inf,
                         exact_tol::Real=eps(Float64))
    size(known_xy, 2) == 2 || throw(ArgumentError("known_xy must be M×2"))
    size(query_xy, 2) == 2 || throw(ArgumentError("query_xy must be N×2"))
    length(known_z) == size(known_xy, 1) ||
        throw(ArgumentError("known_z length must match known_xy rows"))
    k >= 1 || throw(ArgumentError("k must be >= 1"))
    power > 0 || throw(ArgumentError("power must be > 0"))
    max_distance > 0 || throw(ArgumentError("max_distance must be > 0"))

    m = size(known_xy, 1)
    n = size(query_xy, 1)
    n == 0 && return Float64[]
    m == 0 && throw(ArgumentError("known_xy must contain at least one point"))

    # NearestNeighbors expects D×M layout
    known_xy_t = Matrix{Float64}(undef, 2, m)
    @inbounds for i in 1:m
        known_xy_t[1, i] = float(known_xy[i, 1])
        known_xy_t[2, i] = float(known_xy[i, 2])
    end
    tree = KDTree(known_xy_t)
    k_use = min(k, m)
    p = float(power)
    max_d = float(max_distance)

    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        xq = float(query_xy[i, 1])
        yq = float(query_xy[i, 2])
        if !(isfinite(xq) && isfinite(yq))
            out[i] = NaN
            continue
        end
        idxs, dists = knn(tree, SVector(xq, yq), k_use, true)
        if dists[1] > max_d
            out[i] = NaN
            continue
        end
        if dists[1] <= exact_tol
            out[i] = float(known_z[idxs[1]])
            continue
        end
        wsum = 0.0
        zwsum = 0.0
        for j in eachindex(idxs)
            w = 1.0 / (dists[j]^p)
            wsum += w
            zwsum += w * float(known_z[idxs[j]])
        end
        out[i] = wsum > 0 ? (zwsum / wsum) : NaN
    end
    return out
end
