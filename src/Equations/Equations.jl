"""
    AbstractEquation{NV}

Must contain a field 'operators' containing the operators that it uses (iterable).
"""
abstract type AbstractEquation{NV} end

nvariables(::AbstractEquation{NV}) where {NV} = NV
eachvariable(e::AbstractEquation) = Base.Base.OneTo(nvariables(e))

function variablenames end

"""
    rhs!(dQ, Q, p::Tuple{<:AbstractEquation,<:AbstractSpatialDiscretization}, time)

Evaluate the right-hand side (spatial term) of the ODE.
"""
function rhs! end

#==========================================================================================#
#                                   Hyperbolic equations                                   #

abstract type HyperbolicEquation{NV} <: AbstractEquation{NV} end

requires_subgrid(e::HyperbolicEquation) = requires_subgrid(e.div_operator)

include("Hyperbolic/Hyperbolic.jl")
include("Hyperbolic/LinearAdvection.jl")
include("Hyperbolic/Burgers.jl")
include("Hyperbolic/Euler.jl")
include("Hyperbolic/KPP.jl")
