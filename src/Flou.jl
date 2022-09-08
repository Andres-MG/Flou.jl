module Flou

using LinearAlgebra: LinearAlgebra, norm, normalize, dot, cross, mul!, factorize, ldiv!
using LinearAlgebra: transpose, Transpose, diag, Diagonal
using SparseArrays: SparseArrays, SparseMatrixCSC, sparse, mul!
using StaticArrays: StaticArrays, SVector, MVector, SMatrix, MMatrix, MArray, SDiagonal
using StructArrays: StructArrays, StructVector, LazyRows, LazyRow
using FastGaussQuadrature: FastGaussQuadrature, gausslegendre, gausslobatto
using Polynomials: Polynomials, Polynomial, derivative
using SpecialPolynomials: SpecialPolynomials, Lagrange
using Printf: Printf, @printf, @sprintf
using HDF5: HDF5
using OrdinaryDiffEq: OrdinaryDiffEq, ODEProblem, solve, DiscreteCallback, CallbackSet
using DiffEqCallbacks: DiffEqCallbacks, SavingCallback, SavedValues
using Gmsh: gmsh

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

# EquationsInterface.jl
export nvariables, eachvariable, variablenames

# SpatialDiscretization.jl
export CartesianMesh, StepMesh, UnstructuredMesh
export apply_periodicBCs!
export get_spatialdim, phys_coords
export nelements, nboundaries, nfaces, nintfaces, nbdfaces, nperiodic, nvertices, nregions
export get_elements, get_faces, get_intfaces, get_bdfaces, get_periodic, get_vertices
export get_element, get_face, get_intface, get_bdface, get_vertex, get_region
export eachelement, eachboundary, eachface, eachintface, eachbdface, eachvertex, eachregion

export GaussQuadrature, GL, GaussLobattoQuadrature, GLL
export StdSegment, StdQuad, StdHex
export ndofs, is_tensor_product, ndirections, eachdirection, ndofs
export get_quadrature, get_edges, get_edge

export DirichletBC

export DGSEM
export StdAverageNumericalFlux, LxFNumericalFlux

export nregions, nelements
export eachregion, eachelement

# Operators.jl
export WeakDivOperator, StrongDivOperator, SplitDivOperator, SSFVDivOperator

# Equations.jl
export LinearAdvection
export BurgersEquation
export EulerEquation, pressure, energy, math_entropy, soundvelocity
export ChandrasekharAverage, ScalarDissipation, MatrixDissipation
export EulerInflowBC, EulerOutflowBC, EulerSlipBC
export KPPEquation

# TimeDiscretization.jl
export integrate, make_callback_list, get_save_callback, get_monitors_callback

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
