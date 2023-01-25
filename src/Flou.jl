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
export nequispaced, nodetype
export integrate

export GaussNodes, GaussChebyshevNodes, GaussLobattoNodes
export GL, GCL, GLL

export StdAverageNumericalFlux, LxFNumericalFlux
export ChandrasekharAverage, ScalarDissipation, MatrixDissipation

export GenericBC
export EulerInflowBC, EulerOutflowBC, EulerSlipBC

export FR
export VCJH, reconstruction_name
export FRStdPoint, FRStdSegment, FRStdQuad, FRStdHex

export StrongDivOperator, SplitDivOperator, SSFVDivOperator
export StrongGradOperator

# export SD
# export SDStdPoint, SDStdSegment, SDStdQuad, SDStdHex
# export nodetype_solution, nodetype_flux

# Time discretizations
include("FlouTime/FlouTime.jl")
using .FlouTime

export timeintegrate, make_callback_list
export get_save_callback, get_cfl_callback, get_monitor_callback

end # Flou
