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

function Base.show(io::IO, ::MIME"text/plain", eq::GradientEquation)
    @nospecialize
    nd = ndims(eq)
    nvars = nvariables(eq)
    vstr = (nvars == 1) ? " variable" : " variables"
    print(io, nd, "D Gradient equation with ", nvars, vstr)
    return nothing
end

function variablenames(::GradientEquation{ND,NV}; unicode=false) where {ND,NV}
    names = if unicode
        ["∂u$(i)/∂x$(j)" for i in 1:NV, j in 1:ND]
    else
        ["u_$(i)$(j)" for i in 1:NV, j in 1:ND]
    end
    return names |> vec |> Tuple
end
