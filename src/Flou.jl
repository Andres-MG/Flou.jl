module Flou

# Globals
export FLOUVERSION, print_flou_header

using TOML: parsefile
const FLOUVERSION = let
    file = parsefile(joinpath(pkgdir(@__MODULE__), "Project.toml"))
    VersionNumber(file["version"])
end

function print_flou_header(io::IO=stdout)
    @nospecialize
    # Taken from https://onlineasciitools.com/convert-text-to-ascii-art
    header =
        raw"_______________             "  * "\n" *
        raw"___  ____/__  /_________  __"  * "\n" *
        raw"__  /_   __  /_  __ \  / / /"  * "\n" *
        raw"_  __/   _  / / /_/ / /_/ / "  * "\n" *
        raw"/_/      /_/  \____/\__,_/ v." * string(FLOUVERSION)
    println(io, "")
    println(io, header)
    println(io, "")
end

# Code common to all submodules
include("FlouCommon/FlouCommon.jl")
using .FlouCommon

export CartesianMesh, StepMesh, UnstructuredMesh
export spatialdim, eachdim
export nelements, nboundaries, nfaces, nintfaces, nbdfaces, nperiodic, nvertices, nregions
export intfaces, bdfaces, periodic, regions, intface, bdface, region
export eachelement, eachboundary, eachface, eachintface, eachbdface
export eachvertex, eachmapping, eachregion
export apply_periodicBCs!

export nvariables, eachvariable
export LinearAdvection, BurgersEquation, KPPEquation, EulerEquation, GradientEquation
export EquationConfig
export rhs!

export list_monitors

# Visualization
include("FlouBiz/FlouBiz.jl")
using .FlouBiz

# Spatial discretizations
include("FlouSpatial/FlouSpatial.jl")
using .FlouSpatial

export StateVector, BlockVector

export dofsize, ndofs, eachdof
export integrate

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

# Time discretizations
include("FlouTime/FlouTime.jl")
using .FlouTime

export timeintegrate, make_callback_list
export get_save_callback, get_cfl_callback, get_monitor_callback

end # Flou
