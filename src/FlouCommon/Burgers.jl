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

struct BurgersEquation <: HyperbolicEquation{1,1} end

function Base.show(io::IO, ::MIME"text/plain", eq::BurgersEquation)
    @nospecialize
    print(io, "1D Burgers equation")
end

function variablenames(::BurgersEquation; unicode=false)
    return if unicode
        ("u",)
    else
        ("u",)
    end
end

function volumeflux(Q, ::BurgersEquation)
    return (SVector{1}(Q[1]^2 / 2),)
end

function get_max_dt(Q, Δx::Real, cfl::Real, ::BurgersEquation)
    return cfl * Δx / Q[1]
end
