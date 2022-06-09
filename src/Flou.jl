module Flou

using LinearAlgebra: LinearAlgebra, dot, mul!, Diagonal, diagm, factorize, ldiv!
using StaticArrays: StaticArrays, SVector, MVector, SMatrix, SDiagonal
using StructArrays: StructArrays, StructVector, LazyRows, LazyRow
using FastGaussQuadrature:FastGaussQuadrature, gausslegendre, gausslobatto
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
        raw"_______________"*"\n"*
        raw"___  ____/__  /_________  __"*"\n"*
        raw"__  /_   __  /_  __ \  / / /"*"\n"*
        raw"_  __/   _  / / /_/ / /_/ /"*"\n"*
        raw"/_/      /_/  \____/\__,_/ v."*
        string(FlouVersion)
    println(io, "")
    println(io, header)
    println(io, "")
end

# Utilities.jl
export gaussian_bump

# Mesh.jl
export CartesianMesh
export apply_periodicBCs!
export spatialdim, coords
export nelements, nboundaries, nfaces, nintfaces, nbdfaces, nperiodic, nvertices
export elements, faces, intfaces, bdfaces, periodic, vertices
export element, face, intface, bdface, vertex
export eachelement, eachboundary, eachface, eachintface, eachbdface, eachvertex

# StdRegions.jl
export GaussQuadrature, GL, GaussLobattoQuadrature, GLL
export ndofs, eachdirection
export StdSegment, StdQuad

# Containers.jl
export StateVector, MortarStateVector
export nregions, nvariables, eachregion, eachvariable

# SpatialDiscretizations.jl
export StdAverageNumericalFlux, LxFNumericalFlux
export DGSEM
export WeakDivOperator, StrongDivOperator, SplitDivOperator, SSFVDivOperator

# Equations.jl
export DirichletBC
export evaluate!, variablenames

export LinearAdvection
export BurgersEquation
export EulerEquation, pressure, soundvelocity
export vars_cons2prim, vars_prim2cons, vars_cons2entropy
export ChandrasekharAverage, MatrixDissipation
export EulerInflowBC, EulerOutflowBC, EulerSlipBC
export KPPEquation, initial_whirl_KPP!

# FlouBiz.jl
export save

# TimeIntegration.jl
export integrate, get_save_callback

include("Utilities.jl")
include("Mesh/Mesh.jl")
include("StdRegions.jl")
include("DofHandler.jl")
include("Containers.jl")
include("SpatialDiscretizations/PhysicalRegions.jl")
include("SpatialDiscretizations/SpatialDiscretizations.jl")
include("Equations/Equations.jl")
include("FlouBiz.jl")
include("TimeIntegration.jl")

end # Flou
