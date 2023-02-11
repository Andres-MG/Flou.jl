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

module FlouSpatial

using ..FlouCommon
using ..FlouBiz
using StaticArrays: StaticArrays, SVector, MVector, SMatrix, SDiagonal
using LinearAlgebra: LinearAlgebra, Transpose, Diagonal
using LinearAlgebra: dot, cross, mul!, lmul!, diag, factorize, normalize, norm
using BlockDiagonals: BlockDiagonals, BlockDiagonal, blocks
using SparseArrays: SparseArrays, SparseMatrixCSC, sparse, mul!
using Polynomials: Polynomials
using SpecialPolynomials: SpecialPolynomials, Legendre
using Polyester: Polyester, @batch
using HDF5: HDF5, File, write, create_group, attributes

# Nodal distributions and related functionality
include("ApproximationBases.jl")
using .ApproximationBases

export StateVector, BlockVector

export GaussNodes, GaussChebyshevNodes, GaussLobattoNodes
export GL, GCL, GLL

export dofsize, lineardofs, cartesiandofs, ndofs, eachdof
export ndirections, eachdirection
export equisize, nequispaced, nodetype
export is_tensor_product, project2equispaced!
export nfacedofs, get_dofid
export integrate, contravariant

export StdAverageNumericalFlux, LxFNumericalFlux
export ChandrasekharAverage, ScalarDissipation, MatrixDissipation

export GenericBC
export EulerInflowBC, EulerOutflowBC, EulerSlipBC

export FR
export VCJH, reconstruction_name
export FRStdPoint, FRStdSegment, FRStdQuad, FRStdHex

export StrongDivOperator, SplitDivOperator, SSFVDivOperator
export StrongGradOperator

FlouCommon.nelements(disc::AbstractSpatialDiscretization) = nelements(disc.dh)
ndofs(disc::AbstractSpatialDiscretization) = ndofs(disc.dh)

# Code common to all spatial discretizations
include("StdRegions.jl")
include("DofHandler.jl")
include("PhysicalRegions.jl")
include("Containers.jl")
include("IO.jl")
include("Interfaces.jl")
include("GlobalContainers.jl")

abstract type AbstractOperator end

abstract type AbstractCache end

"""
    construct_cache(disctype::Symbol, realtype, dofhandler, equation)

Allocates the required global storage for the specified spatial discretization and equation.
"""
function construct_cache(::Symbol, ::Type, ::DofHandler, ::AbstractEquation) end

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

# Spatial discretizations
include("FR/FR.jl")
# include("StaggeredFR/StaggeredFR.jl")

end # FlouSpatial