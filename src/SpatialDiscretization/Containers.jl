abstract type DGContainer{NV,RT<:Real,T} <: AbstractVector{T} end

datatype(::DGContainer{NV,RT}) where {NV,RT} = RT
Base.ndims(::DGContainer{NV}) where {NV} = NV
eachdim(dg::DGContainer) = Base.OneTo(ndims(dg))

#==========================================================================================#
#                                       State vector                                       #

struct StateVector{NV,RT<:Real,D<:HybridVector{NV,RT},T} <: DGContainer{NV,RT,T}
    data::D
    dh::DofHandler
    function StateVector(data::HybridVector, dh::DofHandler)
        length(data) == ndofs(dh) || throw(DimensionMismatch(
            "The length of `data` must be equal to the number of degrees of freedom"
        ))
        nv = innerdim(data)
        rt = datatype(data)
        etype = view(data, 1:1) |> typeof
        return new{nv,rt,typeof(data),etype}(data, dh)
    end
end

function StateVector{NV}(vec::AbstractVector{<:SVector}, dh::DofHandler) where {NV}
    return StateVector(HybridVector(vec), dh)
end

function StateVector{NV}(mat::AbstractMatrix, dh::DofHandler) where {NV}
    return StateVector(HybridVector{NV}(mat), dh)
end

function StateVector{NV,RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    dh::DofHandler,
) where {
    NV,
    RT
}
    data = HybridVector{NV,RT}(value, ndofs(dh))
    return StateVector(data, dh)
end

function Base.similar(s::StateVector)
    data = similar(s.data)
    return StateVector(data, s.dh)
end

function Base.show(io::IO, ::MIME"text/plain", s::StateVector)
    @nospecialize
    rt = datatype(s)
    nvars = nvariables(s)
    nelem = nelements(s)
    vstr = (nvars == 1) ? " variable" : " variables"
    print(io, nelem, "-element StateVector{", rt, "} with ", nvars, vstr)
    return nothing
end

Base.size(s::StateVector) = size(s.data)
Base.fill!(s::StateVector, v) = fill!(s.data, v)

@inline function Base.getindex(s::StateVector, i::Integer)
    @boundscheck checkbounds(s, i)
    return @inbounds view(s.data, (s.dh.elem_offsets[i] + 1):s.dh.elem_offsets[i + 1])
end

nelements(s::StateVector) = nelements(s.dh)
nvariables(s::StateVector) = innerdim(s.data)
ndofs(s::StateVector) = length(s.data)

eachelement(s::StateVector) = Base.OneTo(nelements(s))
eachvariable(s::StateVector) = Base.OneTo(nvariables(s))
eachdof(s::StateVector) = Base.OneTo(ndofs(s))

#==========================================================================================#
#                                       Block vector                                       #

struct BlockVector{NV,RT<:Real,D<:HybridMatrix{NV,RT},T} <: DGContainer{NV,RT,T}
    data::D
    dh::DofHandler
    function BlockVector(data::HybridMatrix, dh::DofHandler)
        size(data, 1) == ndofs(dh) || throw(DimensionMismatch(
            "The first dimension of `data` must equal the number of dofs in dh"
        ))
        nv = innerdim(data)
        rt = datatype(data)
        etype = view(data, 1:1, :) |> typeof
        return new{nv,rt,typeof(data),etype}(data, dh)
    end
end

function BlockVector{NV}(mat::AbstractMatrix{<:SVector}, dh::DofHandler) where {NV}
    return BlockVector(HybridMatrix(mat), dh)
end

function BlockVector{NV}(array::AbstractArray{RT,3}, dh::DofHandler) where {NV,RT}
    return BlockVector(HybridMatrix{NV}(array), dh)
end

function BlockVector{NV,RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    ndims::Integer,
    dh::DofHandler,
) where {
    NV,
    RT
}
    data = HybridMatrix{NV,RT}(value, ndofs(dh), ndims)
    return BlockVector(data, dh)
end

function Base.similar(b::BlockVector)
    data = similar(b.data)
    return BlockVector(data, b.dh)
end

function Base.show(io::IO, ::MIME"text/plain", b::BlockVector)
    @nospecialize
    rt = datatype(b)
    nvars = nvariables(b)
    nelem = nelements(b)
    dim = ndims(b)
    vstr = (nvars == 1) ? " variable" : " variables"
    print(io, nelem, "-element ", dim, "D BlockVector{", rt, "} with ", nvars, vstr)
    return nothing
end

Base.size(b::BlockVector) = size(b.data)
Base.fill!(b::BlockVector, v) = fill!(b.data, v)

@inline function Base.getindex(b::BlockVector, i::Integer)
    @boundscheck checkbounds(b, i)
    return @inbounds view(
        b.data,
        (b.dh.elem_offsets[i] + 1):b.dh.elem_offsets[i + 1],
        1:ndims(b),
    )
end

nelements(b::BlockVector) = nelements(b.dh)
nvariables(b::BlockVector) = innerdim(b.data)
Base.ndims(b::BlockVector) = size(b.data, 2)
ndofs(b::BlockVector) = size(b.data, 1)

eachelement(b::BlockVector) = Base.OneTo(nelements(b))
eachvariable(b::BlockVector) = Base.OneTo(nvariables(b))
eachdim(b::BlockVector) = Base.OneTo(ndims(b))
eachdof(b::BlockVector) = Base.OneTo(ndofs(b))

#==========================================================================================#
#                                     Face state vector                                    #

struct FaceStateVector{NV,RT<:Real,D<:HybridVector{NV,RT},T} <: DGContainer{NV,RT,T}
    data::D
    dh::DofHandler
    function FaceStateVector(data::HybridVector, dh::DofHandler)
        length(data) == nfacedofs(dh) || throw(DimensionMismatch(
            "The length of `data` must equal the number of face dofs in dh"
        ))
        nv = innerdim(data)
        rt = datatype(data)
        ftype = (view(data, 1:1), view(data, 1:1)) |> typeof
        return new{nv,rt,typeof(data),ftype}(data, dh)
    end
end

function FaceStateVector{NV,RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    dh::DofHandler,
) where {
    NV,
    RT
}
    data = HybridVector{NV,RT}(value, nfacedofs(dh))
    return FaceStateVector(data, dh)
end

function Base.similar(s::FaceStateVector)
    data = similar(s.data)
    return FaceStateVector(data, s.dh)
end

function Base.show(io::IO, ::MIME"text/plain", s::FaceStateVector)
    @nospecialize
    rt = datatype(s)
    nvars = nvariables(s)
    nface = nfaces(s)
    vstr = (nvars == 1) ? " variable" : " variables"
    print(io, nface, "-face FaceStateVector{", rt, "} with ", nvars, vstr)
    return nothing
end

Base.size(s::FaceStateVector) = size(s.data)
Base.fill!(s::FaceStateVector, v) = fill!(s.data, v)

@inline function Base.getindex(s::FaceStateVector, i::Integer)
    @boundscheck checkbounds(s, i)
    @inbounds begin
        i1 = 2 * s.dh.face_offsets[i] + 1
        i2 = (i1 - 1) + s.dh.face_offsets[i + 1] - s.dh.face_offsets[i]
        i3 = 2 * s.dh.face_offsets[i + 1]
        return (view(s.data, i1:i2), view(s.data, (i2 + 1):i3))
    end
end

nfaces(s::FaceStateVector) = nfaces(s.dh)
nvariables(s::FaceStateVector) = innerdim(s.data)

eachface(s::FaceStateVector) = Base.OneTo(nfaces(s))
eachvariable(s::FaceStateVector) = Base.OneTo(nvariables(s))

#==========================================================================================#
#                                     Face block vector                                    #

struct FaceBlockVector{NV,RT<:Real,D<:HybridMatrix{NV,RT},T} <: DGContainer{NV,RT,T}
    data::D
    dh::DofHandler
    function FaceBlockVector(data::HybridMatrix, dh::DofHandler)
        size(data, 1) == nfacedofs(dh) || throw(DimensionMismatch(
            "The first dimension of `data` must equal the number of face dofs in dh"
        ))
        nv = innerdim(data)
        rt = datatype(data)
        ftype = (view(data, 1:1, :), view(data, 1:1, :)) |> typeof
        return new{nv,rt,typeof(data),ftype}(data, dh)
    end
end

function FaceBlockVector{NV,RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    ndims::Integer,
    dh::DofHandler,
) where {
    NV,
    RT
}
    data = HybridMatrix{NV,RT}(value, nfacedofs(dh), ndims)
    return FaceBlockVector(data, dh)
end

function Base.similar(b::FaceBlockVector)
    data = similar(b.data)
    return BlockVector(data, b.dh)
end

function Base.show(io::IO, ::MIME"text/plain", s::FaceBlockVector)
    @nospecialize
    rt = datatype(s)
    nvars = nvariables(s)
    nface = nfaces(s)
    dim = ndims(s)
    vstr = (nvars == 1) ? " variable" : " variables"
    print(io, nface, "-face ", dim, "D FaceBlockVector{", rt, "} with ", nvars, vstr)
    return nothing
end

Base.size(b::FaceBlockVector) = size(b.data)
Base.fill!(b::FaceBlockVector, v) = fill!(b.data, v)

@inline function Base.getindex(b::FaceBlockVector, i::Integer)
    @boundscheck checkbounds(b, i)
    @inbounds begin
        i1 = 2 * b.dh.face_offsets[i] + 1
        i2 = (i1 - 1) + b.dh.face_offsets[i + 1] - b.dh.face_offsets[i]
        i3 = 2 * b.dh.face_offsets[i + 1]
        return (view(b.data, i1:i2, :), view(b.data, (i2 + 1):i3, :))
    end
end

nfaces(b::FaceBlockVector) = nfaces(b.dh)
nvariables(b::FaceBlockVector) = innerdim(b.data)
Base.ndims(b::FaceBlockVector) = size(b.data, 2)

eachface(b::FaceBlockVector) = Base.OneTo(nfaces(b))
eachvariable(b::FaceBlockVector) = Base.OneTo(nvariables(b))
eachdim(b::FaceBlockVector) = Base.OneTo(ndims(b))
