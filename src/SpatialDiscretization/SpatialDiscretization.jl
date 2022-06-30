include("Mesh.jl")
include("StdRegions.jl")
include("PhysicalRegions.jl")

abstract type AbstractSpatialDiscretization{EQ<:AbstractEquation,RT<:Real} end

abstract type AbstractDofHandler end

function ndofs end

abstract type AbstractBC end

function stateBC! end

"""
    DirichletBC(Qext::Function)

Dirichlet boundary condition where `Qext = Qext(Q, x, n, t, b, time, equation)`.
"""
struct DirichletBC{QF} <: AbstractBC
    Qext::QF     # Qext(Qin, x, n, t, b, time, eq)
end

function stateBC(Qin, x, n, t, b, time, eq, bc::DirichletBC)
    return bc.Qext(Qin, x, n, t, b, time, eq)
end

include("DG/DG.jl")
