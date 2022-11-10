#==========================================================================================#
#                                       Hybrid array                                       #

struct HybridArray{NI,RT,N,V<:AbstractArray{SVector{NI,RT}}} <:
        AbstractArray{SVector{NI,RT},N}
    svec::V
    @inline function HybridArray(svec::AbstractArray{SVector{NI,RT}}) where {NI, RT}
        nd = ndims(svec)
        return new{NI,RT,nd,typeof(svec)}(svec)
    end
end

const HybridVector{NI,RT,V} = HybridArray{NI,RT,1,V}
const HybridMatrix{NI,RT,V} = HybridArray{NI,RT,2,V}

HybridVector(svec::AbstractVector{SVector{NI,RT}}) where {NI,RT} = HybridArray(svec)
HybridMatrix(svec::AbstractMatrix{SVector{NI,RT}}) where {NI,RT} = HybridArray(svec)

function HybridArray{NI,RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    dims::Integer...,
) where {
    NI,
    RT
}
    svec = Array{SVector{NI,RT},length(dims)}(value, dims...)
    return HybridArray(svec)
end

HybridVector{NI,RT}(val, dim) where {NI,RT} = HybridArray{NI,RT}(val, dim)
HybridMatrix{NI,RT}(val, dim1, dim2) where {NI,RT} = HybridArray{NI,RT}(val, dim1, dim2)

function HybridArray{NI}(flat::AbstractArray{RT}) where {NI,RT<:Number}
    svec = reshape(reinterpret(SVector{NI,RT}, flat), size(flat)[2:end])
    return HybridArray(svec)
end

HybridVector{NI}(flat) where {NI} = HybridArray{NI}(flat)
HybridMatrix{NI}(flat) where {NI} = HybridArray{NI}(flat)

Base.IndexStyle(::Type{<:HybridArray}) = IndexLinear()
Base.BroadcastStyle(::Type{<:HybridArray}) = Broadcast.ArrayStyle{HybridArray}()
Base.size(ha::HybridArray) = size(ha.svec)
Base.fill!(ha::HybridArray, v::SVector) = fill!(ha.svec, v)
Base.fill!(ha::HybridArray{NI}, v) where {NI} = fill!(ha, @SVector fill(v, NI))

function Base.similar(ha::HybridArray{NI,RT}) where {NI,RT}
    return HybridArray{NI,RT}(undef, size(ha)...)
end

@inline function Base.getindex(ha::HybridArray{NI}, i::Integer) where {NI}
    @boundscheck checkbounds(ha, i)
    return @inbounds ha.svec[i]
end

@inline function Base.setindex!(ha::HybridArray{NI}, value, i::Integer) where {NI}
    @boundscheck checkbounds(ha, i)
    @inbounds ha.svec[i] = value
    return nothing
end

@inline function Base.setindex!(ha::HybridArray{NI}, value::Number, i::Integer) where {NI}
    @boundscheck checkbounds(ha, i)
    @inbounds ha.svec[i] = @SVector fill(value, NI)
    return nothing
end

@inline function Base.view(ha::HybridArray{NI,RT,N}, I::Vararg{Any,N}) where {NI,RT,N}
    @boundscheck checkbounds(ha, I...)
    return @inbounds HybridArray(view(ha.svec, I...))
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
        svec = reshape(ha.svec, dims...)
        return HybridArray(svec)
    end
end

@inline function flatten(ha::HybridArray{NI,RT}) where {NI,RT}
    return reshape(reinterpret(RT, ha.svec), NI, size(ha)...)
end

@inline function as_mut(ha::HybridArray{NI,RT,N}, I::Vararg{Integer,N}) where {NI,RT,N}
    @boundscheck checkbounds(ha, I...)
    return @inbounds view(flatten(ha), 1:NI, I...)
end

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
    return LinearAlgebra.mul!(C.svec, A, B.svec, α, β)
end
Base.@propagate_inbounds function LinearAlgebra.mul!(
    C::HybridVecOrMat,
    A::Transpose{<:Any,<:AbstractVecOrMat},
    B::HybridVecOrMat,
    α::Number,
    β::Number,
)
    return LinearAlgebra.mul!(C.svec, A, B.svec, α, β)
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
    return @inbounds v.data[v.indices[i]] = x
end

