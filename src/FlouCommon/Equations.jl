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

abstract type AbstractEquation{ND,NV} end

"""
    spatialdim(equation)

Return the spatial dimension of `equation`.
"""
function spatialdim(::AbstractEquation{ND}) where {ND}
    return ND
end

"""
    eachdim(equation)

Return a range that covers the spatial dimensions of `equation`.
"""
function eachdim(eq::AbstractEquation)
    return Base.OneTo(spatialdim(eq))
end

"""
    nvariables(equation)

Return the number of variables of `equation`.

See also [`eachvariable`](@ref), [`variablenames`](@ref).
"""
function nvariables(::AbstractEquation{ND,NV}) where {ND,NV}
    return NV
end

"""
    eachvariable(equation)

Return a range that covers all the variables of `equation`.

See also [`nvariables`](@ref), [`variablenames`](@ref).
"""
function eachvariable(e::AbstractEquation)
    return Base.OneTo(nvariables(e))
end

"""
    variablenames(equation, unicode=false)

Return the names of the variables in `equation`. If `unicode=true` the names may contain
unicode symbols.

See also [`nvariables`](@ref), [`eachvariable`](@ref).
"""
function variablenames(::AbstractEquation, ::Bool=false) end

#==========================================================================================#
#                                     Implementations                                      #

abstract type HyperbolicEquation{ND,NV} <: AbstractEquation{ND,NV} end
include("LinearAdvection.jl")
include("Burgers.jl")
include("KPP.jl")
include("Euler.jl")

abstract type GradientEquation{ND,NV} <: AbstractEquation{ND,NV} end
include("Gradient.jl")
