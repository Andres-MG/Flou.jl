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

struct KPPEquation <: HyperbolicEquation{2,1} end

function Base.show(io::IO, ::MIME"text/plain", eq::KPPEquation)
    @nospecialize
    print(io, "2D KPP equation")
end

function variablenames(::KPPEquation; unicode=false)
    return if unicode
        ("u",)
    else
        ("u",)
    end
end

function volumeflux(Q, ::KPPEquation)
    return SVector{1}(
        SVector{2}(sin(Q[1]), cos(Q[1]))
    )
end

# TODO
# function initial_whirl_KPP!(Q, disc)
#     (; mesh, std, elemgeom) = disc
#     for ie in eachelement(mesh)
#         xy = coords(elemgeom, ie)
#         for I in eachindex(std)
#             x, y = xy[I]
#             Q[I, 1, ie] = (x^2 + y^2) <= 1 ? 7π/2 : π/4
#         end
#     end
# end
