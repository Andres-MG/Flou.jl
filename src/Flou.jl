module Flou

using LinearAlgebra: LinearAlgebra, norm, normalize, dot, cross, mul!, factorize, ldiv!
using LinearAlgebra: transpose, Transpose, diag, Diagonal
using SparseArrays: SparseArrays, SparseMatrixCSC, sparse, mul!
using StaticArrays: StaticArrays, SVector, MVector, SMatrix, SDiagonal, @SVector
using FastGaussQuadrature: FastGaussQuadrature, gausslegendre, gausslobatto
using Polynomials: Polynomials, Polynomial, derivative
using SpecialPolynomials: SpecialPolynomials, Lagrange
using Polyester: Polyester, @batch
using Printf: Printf, @printf, @sprintf
using HDF5: HDF5
using OrdinaryDiffEq: OrdinaryDiffEq, ODEProblem, solve, DiscreteCallback, CallbackSet
using Gmsh: gmsh

# Globals
export FLOUVERSION, print_flou_header

const FLOUVERSION = v"0.1.0"

function print_flou_header(io::IO=stdout)
    @nospecialize
    # Taken from https://onlineasciitools.com/convert-text-to-ascii-art
    header =
        raw"_______________"               * "\n" *
        raw"___  ____/__  /_________  __"  * "\n" *
        raw"__  /_   __  /_  __ \  / / /"  * "\n" *
        raw"_  __/   _  / / /_/ / /_/ /"   * "\n" *
        raw"/_/      /_/  \____/\__,_/ v." * string(FLOUVERSION)

    println(io, "")
    println(io, header)
    println(io, "")
end

const AbstractVecOrTuple = Union{AbstractVector,Tuple}

# EquationsInterface.jl
export nvariables, eachdim, eachvariable, variablenames

# Mesh.jl
export CartesianMesh, StepMesh, UnstructuredMesh
export apply_periodicBCs!
export get_spatialdim, phys_coords, face_phys_coords
export nelements, nboundaries, nfaces, nintfaces, nbdfaces, nperiodic, nvertices, nregions
export get_intfaces, get_bdfaces, get_periodic
export get_intface, get_bdface, get_region
export eachelement, eachboundary, eachface, eachintface, eachbdface, eachvertex, eachregion

# SpatialDiscretization.jl
export GenericBC

## ./StdRegions.jl
export GaussQuadrature, GL, GaussLobattoQuadrature, GLL
export StdSegment, StdQuad, StdHex
export is_tensor_product, ndirections, ndofs, eachdirection, eachdof
export get_quadrature

## ./Containers.jl
export StateVector, BlockVector, FaceStateVector, FaceBlockVector

## ./DG.jl
export StdAverageNumericalFlux, LxFNumericalFlux
export integrate

### ././DGSEM.jl
export DGSEM

export WeakDivOperator, StrongDivOperator, SplitDivOperator, SSFVDivOperator
export WeakGradOperator, StrongGradOperator

# Equations.jl
export LinearAdvection

export BurgersEquation

export EulerEquation
export EulerInflowBC, EulerOutflowBC, EulerSlipBC
export ChandrasekharAverage, ScalarDissipation, MatrixDissipation

export KPPEquation

export GradientEquation

# Monitors.jl
export list_monitors

# TimeDiscretization.jl
export timeintegrate, make_callback_list
export get_save_callback, get_cfl_callback, get_monitor_callback

# FlouBiz.jl
export open_for_write, close_file!
export add_fielddata!, add_celldata!, add_pointdata!, add_solution!

# Basic utilities
include("Utilities/Utilities.jl")

# Equations interface
include("Equations/EquationsInterface.jl")

# Mesh
include("Mesh/Mesh.jl")

# Spatial discretizations
include("SpatialDiscretization/SpatialDiscretization.jl")

# Equation implementations
include("Equations/Equations.jl")

# Monitors
include("Monitors/Monitors.jl")

# Time discretizations
include("TimeDiscretization/TimeDiscretization.jl")

# Visualization
include("FlouBiz.jl")

end # Flou
