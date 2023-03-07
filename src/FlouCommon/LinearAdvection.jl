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

struct LinearAdvection{ND,NV,RT} <: HyperbolicEquation{ND,NV}
    a::SVector{ND,RT}
end

function LinearAdvection(velocity::RT...) where {RT}
    ndim = length(velocity)
    1 <= ndim <= 3 || throw(ArgumentError(
        "Linear advection is implemented in 1D, 2D and 3D."
    ))
    LinearAdvection{ndim,1,RT}(SVector{ndim,RT}(velocity...))
end

function Base.show(io::IO, ::MIME"text/plain", eq::LinearAdvection{ND}) where {ND}
    @nospecialize
    println(io, ND, "D linear advection equation:")
    print(io, " Advection velocity: ", eq.a)
end

function variablenames(::LinearAdvection; unicode=false)
    return if unicode
        ("u",)
    else
        ("u",)
    end
end

function volumeflux(Q, eq::LinearAdvection{ND}) where {ND}
    return ntuple(d -> SVector{1}(eq.a[d] * Q[1]), ND)
end

function get_max_dt(_, Δx::Real, cfl::Real, eq::LinearAdvection)
    return cfl * Δx / norm(eq.a)
end
