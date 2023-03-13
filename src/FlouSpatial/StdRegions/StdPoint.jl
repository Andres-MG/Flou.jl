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

struct StdPoint{RT} <: AbstractStdRegion{0}
    ω::Vector{RT}
    ωf::Vector{RT}
    ξ::Vector{SVector{1,RT}}
    ξf::Vector{SVector{1,RT}}
    ξe::Vector{SVector{1,RT}}
    ξc::Vector{SVector{1,RT}}
end

function StdPoint(ftype=Float64)
    return StdPoint(
        [one(ftype)],
        [one(ftype)],
        [SVector(zero(ftype))],
        [SVector(zero(ftype))],
        [SVector(zero(ftype))],
        [SVector(zero(ftype))],
    )
end

ndirections(::StdPoint) = 1
nvertices(::StdPoint) = 1

function tpdofs(::StdPoint, _)
    return (Colon(),)
end

function slave2master(i::Integer, _, ::StdPoint)
    return i
end

function master2slave(i::Integer, _, ::StdPoint)
    return i
end
