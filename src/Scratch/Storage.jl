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

#==========================================================================================#
#                                       Hybrid array                                       #

struct HybridArray{
    NI,RT,N,S<:AbstractArray{SVector{NI,RT},N},F<:AbstractArray{RT}
} <: AbstractArray{SVector{NI,RT},N}
    svec::S
    flat::F
    @inline function HybridArray(
        svec::AbstractArray{SVector{NI,RT}},
        flat::AbstractArray{RT},
    ) where {
        NI,
        RT
    }
        @boundscheck begin
            size(flat) == (NI, size(svec)...) || throw(DimensionMismatch(
                "The sizes of `flat` and `svec` do not match"
            ))
        end
        new{NI,RT,ndims(svec),typeof(svec),typeof(flat)}(svec, flat)
    end
end

const HybridVector{NI,RT,S,F} = HybridArray{NI,RT,1,S,F}
const HybridMatrix{NI,RT,S,F} = HybridArray{NI,RT,2,S,F}

@inline function HybridArray{NI}(flat::DenseArray{RT}) where {NI,RT<:Number}
    @boundscheck size(flat, 1) == NI || throw(DimensionMismatch("size(flat, 1) != $NI"))
    sshape = size(flat) |> Base.tail
    svec = reinterpret(SVector{NI,RT}, flat)
    svec = unsafe_wrap(Array, pointer(svec), sshape)
    return HybridArray(svec, flat)
end

HybridVector{NI}(flat) where {NI} = HybridArray{NI}(flat)
HybridMatrix{NI}(flat) where {NI} = HybridArray{NI}(flat)

@inline function HybridArray(svec::DenseArray{SVector{NI,RT}}) where {NI,RT<:Number}
    fshape = (NI, size(svec)...)
    flat = reinterpret(RT, svec)
    flat = unsafe_wrap(Array, pointer(flat), fshape)
    return HybridArray(svec, flat)
end

HybridVector(svec) = HybridArray(svec)
HybridMatrix(svec) = HybridArray(svec)

@inline function HybridArray{NI,RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    dims::Integer...,
) where {
    NI,
    RT
}
    flat = Array{RT,length(dims) + 1}(value, NI, dims...)
    return HybridArray{NI}(flat)
end

HybridVector{NI,RT}(val, dim) where {NI,RT} = HybridArray{NI,RT}(val, dim)
HybridMatrix{NI,RT}(val, dim1, dim2) where {NI,RT} = HybridArray{NI,RT}(val, dim1, dim2)

Base.IndexStyle(::Type{<:HybridArray}) = IndexLinear()
Base.BroadcastStyle(::Type{<:HybridArray}) = Broadcast.ArrayStyle{HybridArray}()
Base.size(ha::HybridArray) = size(ha.svec)

function Base.fill!(ha::HybridArray, v::SVector)
    fill!(ha.svec, v)
    return ha
end
function Base.fill!(ha::HybridArray{NI}, v) where {NI}
    fill!(ha, @SVector fill(v, NI))
    return ha
end

function Base.similar(::HybridArray{NI}, ::Type{<:SVector{NI,S}}, dims::Dims) where {NI,S}
    return HybridArray{NI,S}(undef, dims...)
end

function Base.similar(::HybridArray{NI}, ::Type{S}, dims::Dims) where {NI,S}
    return HybridArray{NI,S}(undef, dims...)
end

@inline function Base.getindex(ha::HybridArray{NI}, i::Integer) where {NI}
    @boundscheck checkbounds(ha, i)
    return @inbounds ha.svec[i]
end

@inline function Base.setindex!(ha::HybridArray{NI}, value, i::Integer) where {NI}
    @boundscheck checkbounds(ha, i)
    @inbounds ha.svec[i] = value
end

@inline function Base.setindex!(ha::HybridArray{NI}, value::Number, i::Integer) where {NI}
    @boundscheck checkbounds(ha, i)
    @inbounds ha.svec[i] = @SVector fill(value, NI)
end

@inline function Base.view(ha::HybridArray{NI,RT,N}, I::Vararg{Any,N}) where {NI,RT,N}
    @boundscheck checkbounds(ha, I...)
    @inbounds begin
        svec = view(ha.svec, I...)
        flat = view(ha.flat, 1:NI, I...)
        return HybridArray(svec, flat)
    end
end

@inline function Base.reshape(
    ha::HybridArray{NI,RT},
    dims::Tuple{Vararg{Int,N}},
) where {
    NI,
    RT,
    N
}
    @boundscheck prod(dims) == length(ha) || throw(DimensionMismatch())
    @inbounds begin
        svec = reshape(ha.svec, dims)
        flat = reshape(ha.flat, (NI, dims...))
        return HybridArray(svec, flat)
    end
end

datatype(a::AbstractArray) = eltype(a)
datatype(::HybridArray{NI,RT}) where {NI,RT} = RT
innerdim(::HybridArray{NI}) where {NI} = NI

const HybridVecOrMat = Union{HybridVector,HybridMatrix}

Base.@propagate_inbounds function LinearAlgebra.mul!(
    C::HybridVecOrMat,
    A::AbstractVecOrMat,
    B::HybridVecOrMat,
    α::Number,
    β::Number,
)
    return mul!(C.svec, A, B.svec, α, β)
end

Base.@propagate_inbounds function LinearAlgebra.mul!(
    C::HybridVecOrMat,
    A::Transpose{<:Any,<:AbstractVecOrMat},
    B::HybridVecOrMat,
    α::Number,
    β::Number,
)
    return mul!(C.svec, A, B.svec, α, β)
end

Base.@propagate_inbounds function LinearAlgebra.mul!(
    C::HybridVecOrMat,
    A::Diagonal,
    B::HybridVecOrMat,
    α::Number,
    β::Number,
)
    return mul!(C.svec, A, B.svec, α, β)
    println("heeey")
end

# TODO: dot in DifferentialEquations.jl will not work without this
Base.@propagate_inbounds LinearAlgebra.dot(a::Number, b::SVector) = a * b

# TODO: Required for mul!
Base.@propagate_inbounds Base.:+(s::SVector, n::Number) = s .+ n
