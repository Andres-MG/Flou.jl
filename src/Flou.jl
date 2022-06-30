module Flou

using LinearAlgebra: LinearAlgebra, dot, mul!, Diagonal, factorize, ldiv!
using SparseArrays: SparseArrays, sparse, mul!
using StaticArrays: StaticArrays, SVector, MVector, SMatrix, MMatrix, MArray, SDiagonal
using StructArrays: StructArrays, StructVector, LazyRows, LazyRow
using FastGaussQuadrature: FastGaussQuadrature, gausslegendre, gausslobatto
using Polynomials: Polynomials, Polynomial, derivative
using SpecialPolynomials: SpecialPolynomials, Lagrange
using Printf: Printf, @printf, @sprintf
using HDF5: HDF5
using OrdinaryDiffEq: OrdinaryDiffEq, ODEProblem, solve
using DiffEqCallbacks: DiffEqCallbacks, PresetTimeCallback

# Globals
export FlouVersion, print_flou_header

const FlouVersion = v"0.1.0"

function print_flou_header(io::IO=stdout)
    @nospecialize
    # Taken from https://onlineasciitools.com/convert-text-to-ascii-art
    header =
        raw"_______________"               * "\n" *
        raw"___  ____/__  /_________  __"  * "\n" *
        raw"__  /_   __  /_  __ \  / / /"  * "\n" *
        raw"_  __/   _  / / /_/ / /_/ /"   * "\n" *
        raw"/_/      /_/  \____/\__,_/ v." * string(FlouVersion)

    println(io, "")
    println(io, header)
    println(io, "")
end

const AbstractVecOrTuple = Union{AbstractVector,Tuple}

# Containers.jl
export StateVector, MortarStateVector
export nregions, nvariables, eachregion, eachvariable

# FlouBiz.jl
export save

# EquationsInterface.jl
export nvariables, eachvariable, variablenames

# SpatialDiscretization.jl
export CartesianMesh, StepMesh
export apply_periodicBCs!
export spatialdim, coords
export nelements, nboundaries, nfaces, nintfaces, nbdfaces, nperiodic, nvertices
export elements, faces, intfaces, bdfaces, periodic, vertices
export element, face, intface, bdface, vertex
export eachelement, eachboundary, eachface, eachintface, eachbdface, eachvertex

export GaussQuadrature, GL, GaussLobattoQuadrature, GLL
export StdSegment, StdQuad
export ndofs, is_tensor_product, ndirections, eachdirection, ndofs, quadratures, quadrature

export DirichletBC

export DGSEM
export StdAverageNumericalFlux, LxFNumericalFlux
export rotate2face!, rotate2phys!

export nregions, nelements
export eachregion, eachelement

# Operators.jl
export WeakDivOperator, StrongDivOperator, SplitDivOperator, SSFVDivOperator

# Equations.jl
export LinearAdvection
export BurgersEquation
export EulerEquation, pressure, energy, soundvelocity
export ChandrasekharAverage, MatrixDissipation
export EulerInflowBC, EulerOutflowBC, EulerSlipBC
export KPPEquation, initial_whirl_KPP!

# TimeDiscretization.jl
export integrate, get_save_callback

# Basic utilities
include("Utilities.jl")
include("Containers.jl")

# Equations interface
include("Equations/EquationsInterface.jl")

# Spatial discretizations
include("SpatialDiscretization/SpatialDiscretization.jl")

# Spatial operators
include("Operators/Operators.jl")

# Equation implementations
include("Equations/Equations.jl")

# Time discretizations
include("TimeDiscretization/TimeDiscretization.jl")

# Visualization
include("FlouBiz.jl")

end # Flou
