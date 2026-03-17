"""
    qsm(; kwargs...) -> NamedTuple

Placeholder for quantitative structural modeling stage.
"""
function qsm(; kwargs...)
    return (status=:placeholder, kwargs=NamedTuple(kwargs))
end
