# Copyright (C) 2023 Andrés Mateo Gabín
#
# This file is part of Flou.jl.
#
# Flou.jl is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# Flou.jl is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Flou.jl. If
# not, see <https://www.gnu.org/licenses/>.

module FlouCommon

using StaticArrays: StaticArrays, SVector, @SVector
using LinearAlgebra: LinearAlgebra, Transpose, Diagonal, dot, cross, mul!
using Gmsh: Gmsh, gmsh

export gaussian_bump, logarithmic_mean, @flouthreads
export datatype

export AbstractMesh, CartesianMesh, StepMesh, UnstructuredMesh
export spatialdim, eachdim, datatype
export nelements, nboundaries, nfaces, nintfaces, nbdfaces, nperiodic, nvertices, nregions
export intfaces, bdfaces, periodic, regions, intface, bdface, region
export eachelement, eachboundary, eachface, eachintface, eachbdface
export eachvertex, eachmapping, eachregion
export apply_periodicBCs!
export phys_coords, face_phys_coords, map_basis, map_dual_basis, map_jacobian

export AbstractEquation
export nvariables, eachvariable, variablenames

export HyperbolicEquation, LinearAdvection, BurgersEquation, KPPEquation, EulerEquation
export volumeflux, get_max_dt, pressure, kinetic_energy, energy, entropy, math_entropy 
export soundvelocity_sqr, soundvelocity, vars_cons2prim, vars_prim2cons
export vars_cons2entropy, vars_entropy2prim, vars_entropy2cons
export normal_shockwave

export GradientEquation

export AbstractSpatialDiscretization, EquationConfig
export rhs!

export list_monitors, get_monitor

# Basic utilities
include("Utilities.jl")

# Mesh
include("Mesh.jl")

# Equations interface and definitions
include("Equations.jl")

# Discretization of the spatial terms of the equations
abstract type AbstractSpatialDiscretization{ND,RT} end

"""
    spatialdim(discretization)

Return the spatial dimension of `discretization`.
"""
function spatialdim(::AbstractSpatialDiscretization{ND}) where {ND}
    return ND
end

"""
    eachdim(discretization)

Return a range that covers the spatial dimensions of `discretization`.
"""
function eachdim(disc::AbstractSpatialDiscretization)
    return Base.OneTo(spatialdim(disc))
end

"""
    datatype(discretization)

Return the type of float used internally by `discretization`.
"""
function datatype(::AbstractSpatialDiscretization{ND,RT}) where {ND,RT}
    return RT
end

struct EquationConfig{D<:AbstractSpatialDiscretization,E<:AbstractEquation}
    disc::D
    equation::E
end

"""
    rhs!(dQ, Q, config, time)

Evaluate the right-hand side (spatial part) of the ODE.
"""
function rhs!(::AbstractArray, ::AbstractArray, ::EquationConfig, ::Real) end

# Monitors
"""
    list_monitors(disc, equation)

List the monitors available for the given discretization and equation.
"""
function list_monitors(::AbstractSpatialDiscretization, ::AbstractEquation) end

"""
    get_monitor(disc, equation, name, params=nothing)

Get the monitor with the given name for the given discretization and equation.
"""
function get_monitor(
    ::AbstractSpatialDiscretization,
    ::AbstractEquation,
    ::Symbol,
    ::Any,
) end

end # FlouCommon