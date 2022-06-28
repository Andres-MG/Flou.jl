"""
    AbstractEquation{NV}

Must contain a field `operators` containing the operators that it uses (iterable).
"""
abstract type AbstractEquation{NV} end

@inline nvariables(::AbstractEquation{NV}) where {NV} = NV
@inline eachvariable(e::AbstractEquation) = Base.Base.OneTo(nvariables(e))

function variablenames end

"""
    rhs!(dQ, Q, p::Tuple{<:AbstractEquation,<:AbstractSpatialDiscretization}, time)

Evaluate the right-hand side (spatial term) of the ODE.
"""
function rhs! end
