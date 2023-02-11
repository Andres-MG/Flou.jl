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
#                                       State vector                                       #

struct StateVector{NV,M}
    data::M
    function StateVector{NV}(data) where {NV}
        return new{NV,typeof(data)}(data)
    end
end

function StateVector(data::AbstractMatrix)
    nv = size(data, 2)
    return StateVector{nv}(data)
end

function StateVector{NV,RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    ndofs::Int,
) where {
    NV,
    RT
}
    data = Matrix{RT}(value, ndofs, NV)
    return StateVector{NV}(data)
end

function Base.similar(s::StateVector, ::Type{T}, dims::Dims) where {T}
    data = similar(s.data, T, dims)
    return StateVector(data)
end

function Base.show(io::IO, ::MIME"text/plain", s::StateVector)
    @nospecialize
    rt = datatype(s)
    ndof = ndofs(s)
    nvars = nvariables(s)
    vstr = (nvars == 1) ? " variable" : " variables"
    print(io, ndof, "-DOF StateVector{", rt, "} with ", nvars, vstr)
    return nothing
end

function Base.fill!(s::StateVector, v)
    fill!(s.data, v)
    return s
end

FlouCommon.datatype(s::StateVector) = eltype(s.data)

FlouCommon.nvariables(::StateVector{NV}) where {NV} = NV
Base.@propagate_inbounds ndofs(s::StateVector) = size(s.data, 1)

FlouCommon.eachvariable(s::StateVector) = Base.OneTo(nvariables(s))
Base.@propagate_inbounds eachdof(s::StateVector) = Base.OneTo(ndofs(s))

struct SVDofs{NV,RT} <: AbstractVector{SVector{NV,RT}}
    s::StateVector{NV,RT}
end
@inline function Base.size(d::SVDofs)
    return (ndofs(d.s),)
end
function Base.getindex(d::SVDofs{NV}, i) where {NV}
    @boundscheck checkbounds(d, i)
    return SVector{NV}(ntuple(iv -> @inbounds(d.s.data[i, iv]), NV))
end
@inline function Base.setindex!(d::SVDofs{NV}, val, i) where {NV}
    @boundscheck checkbounds(d, i); checkbounds(val, 1:NV)
    @inbounds begin
        for iv in 1:NV
            d.s.data[i, iv] = val[iv]
        end
        return view(d.s.data, i, :)
    end
end

struct SVDofsMut{NV,RT} <: AbstractVector{SVector{NV,RT}}
    s::StateVector{NV,RT}
end
@inline function Base.size(d::SVDofsMut)
    return (ndofs(d.s),)
end
function Base.getindex(d::SVDofsMut{NV}, i) where {NV}
    @boundscheck checkbounds(d, i)
    return @inbounds view(d.s.data, i, :)
end
@inline function Base.setindex!(d::SVDofsMut{NV}, val, i) where {NV}
    @boundscheck checkbounds(d, i); checkbounds(val, 1:NV)
    @inbounds begin
        for iv in 1:NV
            d.s.data[i, iv] = val[iv]
        end
        return view(d.s.data, i, :)
    end
end

const SVVarsView{RT} =
    SubArray{RT,1,Matrix{RT},Tuple{Base.Slice{Base.OneTo{Int64}},Int64},true}

struct SVVars{NV,RT} <: AbstractVector{SVVarsView{RT}}
    s::StateVector{NV,RT}
end
@inline function Base.size(::SVVars{NV}) where {NV}
    return (NV,)
end
@inline function Base.getindex(v::SVVars, i)
    @boundscheck checkbounds(v, i)
    return @inbounds view(v.s.data, :, i)
end
@inline function Base.setindex!(v::SVVars, val, i)
    @boundscheck checkbounds(v, i); checkbounds(val, 1:ndofs(v.s))
    return @inbounds v.s.data[:, i] = val
end

Base.propertynames(::StateVector) = (:dofs, :dofsmut, :vars, fieldnames(StateVector)...)
function Base.getproperty(sv::StateVector, s::Symbol)
    return if s == :dofs
        SVDofs(sv)
    elseif s == :dofsmut
        SVDofsMut(sv)
    elseif s == :vars
        SVVars(sv)
    else
        getfield(sv, s)
    end
end

#==========================================================================================#
#                                       Block vector                                       #

struct BlockVector{NV,A}
    data::A
    function BlockVector{NV}(data::AbstractArray{RT,3}) where {NV,RT}
        return new{NV,typeof(data)}(data)
    end
end

function BlockVector(data::AbstractArray{RT,3}) where {RT}
    nv = size(data, 2)
    return BlockVector{nv}(data)
end

function BlockVector{NV,RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    ndofs::Int,
    ndims::Int,
) where {
    NV,
    RT
}
    data = Array{RT,3}(value, ndofs, NV, ndims)
    return BlockVector{NV}(data)
end

function Base.similar(b::BlockVector, ::Type{T}, dims::Dims) where {T}
    data = similar(b.data, T, dims)
    return BlockVector(data)
end

function Base.show(io::IO, ::MIME"text/plain", b::BlockVector)
    @nospecialize
    rt = datatype(b)
    ndof = ndofs(b)
    nvars = nvariables(b)
    dim = spatialdim(b)
    vstr = (nvars == 1) ? " variable" : " variables"
    print(io, ndof, "-DOF, ", dim, "D BlockVector{", rt, "} with ", nvars, vstr)
    return nothing
end

function Base.fill!(b::BlockVector, v)
    fill!(b.data, v)
    return b
end

FlouCommon.datatype(b::BlockVector) = eltype(b.data)

FlouCommon.nvariables(::BlockVector{NV}) where {NV} = NV
ndofs(b::BlockVector) = size(b.data, 1)

FlouCommon.eachvariable(b::BlockVector) = Base.OneTo(nvariables(b))
eachdof(b::BlockVector) = Base.OneTo(ndofs(b))

FlouCommon.spatialdim(b::BlockVector) = size(b.data, 3)
FlouCommon.eachdim(b::BlockVector) = Base.OneTo(spatialdim(b))

struct BVDofs{NV,RT} <: AbstractMatrix{SVector{NV,RT}}
    b::BlockVector{NV,RT}
end
@inline function Base.size(d::BVDofs{NV}) where {NV}
    return (ndofs(d.b), spatialdim(d.b))
end
@inline function Base.getindex(d::BVDofs{NV}, i, j) where {NV}
    @boundscheck checkbounds(d, i, j)
    return SVector(ntuple(iv -> @inbounds(d.b.data[i, iv, j]), NV))
end
@inline function Base.setindex!(d::BVDofs{NV}, val, i, j) where {NV}
    @boundscheck checkbounds(d, i, j); checkbounds(val, 1:NV)
    @inbounds begin
        for iv in 1:NV
            d.b.data[i, iv, j] = val[iv]
        end
        return view(d.b.data, i, :, j)
    end
end

struct BVDofsMut{NV,RT} <: AbstractMatrix{SVector{NV,RT}}
    b::BlockVector{NV,RT}
end
@inline function Base.size(d::BVDofsMut{NV}) where {NV}
    return (ndofs(d.b), spatialdim(d.b))
end
@inline function Base.getindex(d::BVDofsMut{NV}, i, j) where {NV}
    @boundscheck checkbounds(d, i, j)
    return @inbounds view(d.b.data, i, :, j)
end
@inline function Base.setindex!(d::BVDofsMut{NV}, val, i, j) where {NV}
    @boundscheck checkbounds(d, i, j); checkbounds(val, 1:NV)
    @inbounds begin
        for iv in 1:NV
            d.b.data[i, iv, j] = val[iv]
        end
        return view(d.b.data, i, :, j)
    end
end

const BVVarsView{RT} =
    SubArray{RT,1,Array{RT,3},Tuple{Base.Slice{Base.OneTo{Int64}},Int64,Int64},true}

struct BVVars{NV,RT} <: AbstractMatrix{BVVarsView{RT}}
    b::BlockVector{NV,RT}
end
@inline function Base.size(v::BVVars{NV}) where {NV}
    return (NV, spatialdim(v.b))
end
@inline function Base.getindex(v::BVVars, i, j)
    @boundscheck checkbounds(v, i, j)
    return @inbounds view(v.b.data, :, i, j)
end
@inline function Base.setindex!(v::BVVars, val, i, j)
    @boundscheck checkbounds(v, i, j); checkbounds(val, 1:ndofs(v.b))
    return @inbounds v.b.data[:, i, j] = val
end

Base.propertynames(::BlockVector) = (:dofs, :dofsmut, :vars, fieldnames(BlockVector)...)
function Base.getproperty(bv::BlockVector, s::Symbol)
    return if s == :dofs
        BVDofs(bv)
    elseif s == :dofsmut
        BVDofsMut(bv)
    elseif s == :vars
        BVVars(bv)
    else
        getfield(bv, s)
    end
end
