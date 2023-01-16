module FlouSpatial

using ..FlouCommon
using ..FlouBiz
using StaticArrays: StaticArrays, SVector, MVector, SMatrix, SDiagonal
using LinearAlgebra: LinearAlgebra, Transpose, Diagonal
using LinearAlgebra: dot, cross, mul!, ldiv!, diag, factorize, normalize, norm
using BlockDiagonals: BlockDiagonals, BlockDiagonal, blocks
using SparseArrays: SparseArrays, SparseMatrixCSC, sparse, mul!
using FastGaussQuadrature: FastGaussQuadrature, gausslegendre, gausslobatto
using Polynomials: Polynomials, Polynomial, derivative, fit
using Polyester: Polyester, @batch
using HDF5: HDF5, File, write, create_group, attributes

export StateVector, BlockVector

export dofsize, lineardofs, cartesiandofs, ndofs, eachdof
export ndirections, eachdirection
export is_tensor_product, project2equispaced!
export nfacedofs, get_dofid
export integrate, contravariant

export StdAverageNumericalFlux, LxFNumericalFlux
export ChandrasekharAverage, ScalarDissipation, MatrixDissipation

export GenericBC
export EulerInflowBC, EulerOutflowBC, EulerSlipBC

export DGSEM
export GaussQuadrature, GaussLobattoQuadrature, GL, GLL
export StdPoint, StdSegment, StdQuad, StdHex
export quadrature

export WeakDivOperator, StrongDivOperator, SplitDivOperator, SSFVDivOperator
export WeakGradOperator, StrongGradOperator

# Abstract spatial discretization for high-order multi-element methods
abstract type HighOrderElements{ND,RT} <: AbstractSpatialDiscretization{ND,RT} end

FlouCommon.nelements(disc::HighOrderElements) = nelements(disc.dh)
ndofs(disc::HighOrderElements) = ndofs(disc.dh)

# Code common to all spatial discretizations
include("StdRegions.jl")
include("DofHandler.jl")
include("PhysicalRegions.jl")
include("Containers.jl")
include("IO.jl")
include("Interfaces.jl")

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
include("DGSEM/DGSEM.jl")
include("SD/SD.jl")

end # FlouSpatial