include("Mesh.jl")
include("StdRegions.jl")
include("PhysicalRegions.jl")

abstract type AbstractDofHandler end

abstract type AbstractSpatialDiscretization{EQ,RT} end

abstract type AbstractBC end

"""
    stateBC!(Q, x, n, t, b, time, eq, bc)
"""
function state_BC! end

include("DG/DG.jl")
