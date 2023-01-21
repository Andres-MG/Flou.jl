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

abstract type AbstractGradOperator <: AbstractOperator end

function surface_contribution!(
    G,
    _,
    Fn,
    ielem,
    std::AbstractStdRegion,
    fr::FR,
    equation::AbstractEquation,
    ::AbstractGradOperator,
)
    # Unpack
    (; mesh) = fr

    rt = datatype(G)
    iface = mesh.elements[ielem].faceinds
    facepos = mesh.elements[ielem].facepos

    @inbounds for (s, (face, pos)) in enumerate(zip(iface, facepos))
        mul!(G, std.∂g[s], Fn.face[face][pos], one(rt), one(rt))
    end
    return nothing
end

#==========================================================================================#
#                                 Strong gradient operator                                 #

struct StrongGradOperator{F<:AbstractNumericalFlux} <: AbstractGradOperator
    numflux::F
end

function volume_contribution!(
    G,
    Q,
    ielem,
    std::AbstractStdRegion,
    fr::FR,
    ::AbstractEquation,
    ::StrongGradOperator,
)
    # Unpack
    (; geometry) = fr

    # Weak gradient operator
    d = std.cache.state[Threads.threadid()][1]
    Ja = geometry.elements[ielem].Ja
    @inbounds for dir in eachdirection(std)
        mul!(d, std.Ds[dir], Q)
        for i in eachdof(std), innerdir in eachdirection(std)
            G[i, innerdir] += d[i] * Ja[i][innerdir, dir]
        end
    end
    return nothing
end
