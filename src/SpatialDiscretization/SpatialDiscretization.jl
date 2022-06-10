include("Mesh.jl")
include("StdRegions.jl")
include("PhysicalRegions.jl")

abstract type AbstractSpatialDiscretization{EQ,RT} end

abstract type AbstractDofHandler end

function ndofs end

abstract type AbstractBC end

function stateBC! end

struct DirichletBC{QF} <: AbstractBC
    Q!::QF     # Q!(Q, x, n, t, b, time, eq)  in/out
end

function stateBC!(Q, x, n, t, b, time, eq, bc::DirichletBC)
    bc.Q!(Q, x, n, t, b, time, eq)
    return nothing
end

include("DG/DG.jl")
