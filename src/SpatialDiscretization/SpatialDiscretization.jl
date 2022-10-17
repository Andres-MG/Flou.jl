include("StdRegions.jl")
include("DofHandler.jl")
include("PhysicalRegions.jl")
include("Containers.jl")

abstract type AbstractSpatialDiscretization{RT<:Real} end

Base.eltype(::AbstractSpatialDiscretization{RT}) where {RT} = RT

abstract type AbstractBC end

"""
    DirichletBC(Qext::Function)

Dirichlet boundary condition where `Qext = Qext(Q, x, n, t, b, time, equation)`.
"""
struct DirichletBC{QF} <: AbstractBC
    Qext::QF     # Qext(Qin, x, frame, time, eq)
end

function stateBC(Qin, x, frame, time, eq, bc::DirichletBC)
    return bc.Qext(Qin, x, frame, time, eq)
end

include("DG/DG.jl")
