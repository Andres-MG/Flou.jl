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

function gaussian_bump(x, y, z, x0, y0, z0, sx, sy, sz, h)
    return h * exp(-(x-x0)^2/(2sx^2) - (y-y0)^2/(2sy^2) - (z-z0)^2/(2sz^2))
end

function gaussian_bump(x, y, x0, y0, sx, sy, h)
    rt = typeof(x)
    v0 = zero(rt)
    v1 = one(rt)
    return gaussian_bump(x, y, v0, x0, y0, v0, sx, sy, v1, h)
end

function gaussian_bump(x, x0, sx, h)
    rt = typeof(x)
    v0 = zero(rt)
    v1 = one(rt)
    return gaussian_bump(x, v0, v0, x0, v0, v0, sx, v1, v1, h)
end

function logarithmic_mean(al, ar)
    ξ = al / ar
    f = (ξ - 1) / (ξ + 1)
    u = f^2
    F = if u < convert(typeof(u), 0.01)
        1 + u/3 + u^2/5 + u^3/7
    else
        log(ξ) / 2f
    end
    return (al + ar) / 2F
end

macro flouthreads(expr)
    # esc(quote
    #     return if Threads.nthreads() == 1
    #         $(expr)
    #     else
    #         Threads.@threads $(expr)
    #     end
    # end)
    return esc(:(@batch $(expr)))
end

#==========================================================================================#
#                                        Lazy vector                                       #

struct LazyVector{T} <: AbstractVector{T}
    data::Vector{T}
    indices::Vector{Int}
    function LazyVector(data::AbstractVector{T}, indices::Vector{<:Integer}) where {T}
        minimum(indices) >= 1 || throw(
            ArgumentError("Indices must be positive")
        )
        maximum(indices) <= length(data) || throw(
            ArgumentError("Indices must be less than length(data)")
        )
        return new{T}(data, indices)
    end
end

function LazyVector(data::AbstractVector{T}) where {T}
    return LazyVector(data, collect(1:length(data)))
end

Base.size(v::LazyVector) = size(v.indices)
Base.IndexStyle(::Type{<:LazyVector}) = IndexLinear()

@inline function Base.getindex(v::LazyVector, i)
    @boundscheck checkbounds(v, i)
    return @inbounds v.data[v.indices[i]]
end

@inline function Base.setindex!(v::LazyVector, x, i)
    @boundscheck checkbounds(v, i)
    @inbounds v.data[v.indices[i]] = x
end
