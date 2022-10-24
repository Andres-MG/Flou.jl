include("StdRegions.jl")
include("DofHandler.jl")
include("PhysicalRegions.jl")
include("Containers.jl")

abstract type AbstractSpatialDiscretization{RT<:Real} end

Base.eltype(::AbstractSpatialDiscretization{RT}) where {RT} = RT

abstract type AbstractSpatialDiscretizationCache{RT<:Real} end

"""
    construct_cache(disctype::Symbol, realtype, dofhandler, equation)

Allocates the required global storage for the specified spatial discretization and equation.
"""
function construct_cache end

abstract type AbstractBC end

"""
    GenericBC(Qext::Function)

Generic boundary condition where `Qext = Qext(Q, x, frame, time, equation)`.
"""
struct GenericBC{QF} <: AbstractBC
    Qext::QF
end

function (bc::GenericBC)(Qin, x, frame, time, eq)
    return bc.Qext(Qin, x, frame, time, eq)
end

include("DG/DG.jl")
