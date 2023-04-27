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
using LinearAlgebra: dot, cross, mul!, ldiv!, diag, diagm, factorize, normalize, norm
using BlockDiagonals: BlockDiagonals, BlockDiagonal, blocks
using Polynomials: Polynomials, Polynomial
using SpecialPolynomials: SpecialPolynomials, Legendre
using FastGaussQuadrature: FastGaussQuadrature, gausslegendre, gausschebyshev, gausslobatto
using Polyester: Polyester, @batch
using HDF5: HDF5

export LagrangeBasis, RBFpolyBasis
export hasboundaries, nnodes, basisname, nodesname
export interp_matrix, derivative_matrix

export StateVector, BlockVector
export GlobalStateVector, GlobalBlockVector, FaceStateVector, FaceBlockVector

export dofsize, lineardofs, cartesiandofs, ndofs, eachdof
export ndirections, eachdirection
export equisize, nequispaced, basis
export is_tensor_product, project2equispaced!
export hasboundaries, nnodes
export nfacedofs, eachfacedof, dofid, facedofid
export integrate, contravariant

export StdAverage, LxF
export ChandrasekharAverage, ScalarDissipation, MatrixDissipation

export GenericBC
export EulerInflowBC, EulerOutflowBC, EulerSlipBC

export MultielementDisc
export NodalRec, VCJHrec, DGSEMrec, reconstruction, reconstruction_name
export StdPoint, StdSegment, StdQuad, StdHex

export StrongDivOperator, SplitDivOperator, HybridDivOperator
export StrongGradOperator

#==========================================================================================#
#                    Definitions for multi-element discontinuous methods                   #

include("Containers.jl")
include("StdRegions/StdRegions.jl")
include("DofHandler.jl")
include("PhysicalRegions.jl")
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

include("MultielementDiscontinuous.jl")

end # FlouSpatial
