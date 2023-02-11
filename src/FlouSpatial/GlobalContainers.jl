#==========================================================================================#
#                                       State vector                                       #

struct GlobalStateVector{NV,RT}
    parent::StateVector{NV,Matrix{RT}}
    data::Matrix{RT}
    dh::DofHandler
    function GlobalStateVector{NV}(data::Matrix, dh::DofHandler) where {NV}
        rt = eltype(data)
        parent = StateVector{NV}(data)
        ndofs(parent) == ndofs(dh) || throw(DimensionMismatch())
        return new{NV,rt}(parent, data, dh)
    end
end

function GlobalStateVector(data::Matrix, dh::DofHandler)
    nv = size(data, 2)
    return GlobalStateVector{nv}(data, dh)
end

function GlobalStateVector{NV,RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    dh::DofHandler,
) where {
    NV,
    RT
}
    parent = StateVector{NV,RT}(value, ndofs(dh))
    return GlobalStateVector{NV}(parent.data, dh)
end

function Base.similar(s::GlobalStateVector, ::Type{T}, dims::Dims) where {T}
    parent = similar(s.parent, T, dims)
    return GlobalStateVector(parent.data, s.dh)
end

function Base.show(io::IO, ::MIME"text/plain", s::GlobalStateVector)
    @nospecialize
    rt = datatype(s)
    nvars = nvariables(s)
    nelem = nelements(s)
    vstr = (nvars == 1) ? " variable" : " variables"
    print(io, nelem, "-element GlobalStateVector{", rt, "} with ", nvars, vstr)
    return nothing
end

function Base.fill!(s::GlobalStateVector, v)
    fill!(s.data, v)
    return s
end

FlouCommon.datatype(s::GlobalStateVector) = eltype(s.data)

FlouCommon.nelements(s::GlobalStateVector) = nelements(s.dh)
FlouCommon.nvariables(::GlobalStateVector{NV}) where {NV} = NV
ndofs(s::GlobalStateVector) = ndofs(s.dh)
ndofs(s::GlobalStateVector, elem) = ndofs(s.dh, elem)

FlouCommon.eachelement(s::GlobalStateVector) = Base.OneTo(nelements(s))
FlouCommon.eachvariable(s::GlobalStateVector) = Base.OneTo(nvariables(s))
eachdof(s::GlobalStateVector) = Base.OneTo(ndofs(s))
eachdof(s::GlobalStateVector, elem) = Base.OneTo(ndofs(s, elem))

struct GSVElement{NV,RT}
    s::GlobalStateVector{NV,RT}
    elem::Int
end
Base.propertynames(::GSVElement) = (:dofs, :dofsmut, :vars, fieldnames(GSVElement)...)
function Base.getproperty(e::GSVElement{NV}, s::Symbol) where {NV}
    @inbounds return if s == :dofs
        sv = getfield(e, :s)
        elem = getfield(e, :elem)
        i1 = dofid(sv.dh, elem, 1)
        i2 = dofid(sv.dh, elem, ndofs(sv.dh, elem))
        StateVector{NV}(view(sv.data, i1:i2, :)).dofs
    elseif s == :dofsmut
        sv = getfield(e, :s)
        elem = getfield(e, :elem)
        i1 = dofid(sv.dh, elem, 1)
        i2 = dofid(sv.dh, elem, ndofs(sv.dh, elem))
        StateVector{NV}(view(sv.data, i1:i2, :)).dofsmut
    elseif s == :vars
        sv = getfield(e, :s)
        elem = getfield(e, :elem)
        i1 = dofid(sv.dh, elem, 1)
        i2 = dofid(sv.dh, elem, ndofs(sv.dh, elem))
        StateVector{NV}(view(sv.data, i1:i2, :)).vars
    else
        getfield(e, s)
    end
end

struct GSVElements{NV,RT} <: AbstractVector{GSVElement{NV,RT}}
    s::GlobalStateVector{NV,RT}
end
@inline function Base.size(e::GSVElements)
    return (nelements(e.s),)
end
@inline function Base.getindex(e::GSVElements, i::Int)
    @boundscheck checkbounds(e, i)
    return GSVElement(e.s, i)
end

Base.propertynames(::GlobalStateVector) = (
    :dofs, :dofsmut, :vars, :elements, fieldnames(GlobalStateVector)...
)
function Base.getproperty(sv::GlobalStateVector, s::Symbol)
    return if s == :dofs
        getfield(sv, :parent).dofs
    elseif s == :dofsmut
        getfield(sv, :parent).dofsmut
    elseif s == :vars
        getfield(sv, :parent).vars
    elseif s == :elements
        GSVElements(sv)
    else
        getfield(sv, s)
    end
end

#==========================================================================================#
#                                       Block vector                                       #

struct GlobalBlockVector{NV,RT}
    parent::BlockVector{NV,Array{RT,3}}
    data::Array{RT,3}
    dh::DofHandler
    function GlobalBlockVector{NV}(data::Array{RT,3}, dh::DofHandler) where {NV,RT}
        parent = BlockVector{NV}(data)
        ndofs(parent) == ndofs(dh) || throw(DimensionMismatch())
        return new{NV,RT}(parent, data, dh)
    end
end

function GlobalBlockVector(data::Array{<:Any,3}, dh::DofHandler)
    nv = size(data, 2)
    nd = size(data, 3)
    return GlobalBlockVector{nv,nd}(data, dh)
end

function GlobalBlockVector{NV,RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    dh::DofHandler,
    ndims::Int,
) where {
    NV,
    RT
}
    parent = BlockVector{NV,RT}(value, ndofs(dh), ndims)
    return GlobalBlockVector{NV}(parent.data, dh)
end

function Base.similar(b::GlobalBlockVector, ::Type{T}, dims::Dims) where {T}
    parent = similar(b.parent, T, dims)
    return GlobalBlockVector(parent.data, b.dh)
end

function Base.show(io::IO, ::MIME"text/plain", b::GlobalBlockVector)
    @nospecialize
    rt = datatype(b)
    nvars = nvariables(b)
    nelem = nelements(b)
    dim = spatialdim(b)
    vstr = (nvars == 1) ? " variable" : " variables"
    print(io, nelem, "-element ", dim, "D GlobalBlockVector{", rt, "} with ", nvars, vstr)
    return nothing
end

function Base.fill!(b::GlobalBlockVector, v)
    fill!(b.data, v)
    return b
end

FlouCommon.datatype(b::GlobalBlockVector) = eltype(b.data)

FlouCommon.nelements(b::GlobalBlockVector) = nelements(b.dh)
FlouCommon.nvariables(::GlobalBlockVector{NV}) where {NV} = NV
ndofs(b::GlobalBlockVector) = size(b.data, 1)
ndofs(b::GlobalBlockVector, elem) = ndofs(b.dh, elem)

FlouCommon.eachelement(b::GlobalBlockVector) = Base.OneTo(nelements(b))
FlouCommon.eachvariable(b::GlobalBlockVector) = Base.OneTo(nvariables(b))
eachdof(b::GlobalBlockVector) = Base.OneTo(ndofs(b))
eachdof(b::GlobalBlockVector, elem) = Base.OneTo(ndofs(b, elem))

FlouCommon.spatialdim(b::GlobalBlockVector) = spatialdim(b.parent)
FlouCommon.eachdim(b::GlobalBlockVector) = Base.OneTo(spatialdim(b))

struct GBVElement{NV,RT}
    b::GlobalBlockVector{NV,RT}
    elem::Int
end
Base.propertynames(::GBVElement) = (:dofs, :dofsmut, :vars, fieldnames(GBVElement)...)
function Base.getproperty(e::GBVElement{NV}, s::Symbol) where {NV}
    @inbounds return if s == :dofs
        bv = getfield(e, :b)
        elem = getfield(e, :elem)
        i1 = dofid(bv.dh, elem, 1)
        i2 = dofid(bv.dh, elem, ndofs(bv.dh, elem))
        BlockVector{NV}(view(bv.data, i1:i2, :, :)).dofs
    elseif s == :dofsmut
        bv = getfield(e, :b)
        elem = getfield(e, :elem)
        i1 = dofid(bv.dh, elem, 1)
        i2 = dofid(bv.dh, elem, ndofs(bv.dh, elem))
        BlockVector{NV}(view(bv.data, i1:i2, :, :)).dofsmut
    elseif s == :vars
        bv = getfield(e, :b)
        elem = getfield(e, :elem)
        i1 = dofid(bv.dh, elem, 1)
        i2 = dofid(bv.dh, elem, ndofs(bv.dh, elem))
        BlockVector{NV}(view(bv.data, i1:i2, :, :)).vars
    else
        getfield(e, s)
    end
end

struct GBVElements{NV,RT} <: AbstractVector{GBVElement{NV,RT}}
    b::GlobalBlockVector{NV,RT}
end
@inline function Base.size(e::GBVElements)
    return (nelements(e.b),)
end
@inline function Base.getindex(e::GBVElements, i::Int)
    @boundscheck checkbounds(e, i)
    return GBVElement(e.b, i)
end

Base.propertynames(::GlobalBlockVector) = (
    :dofs, :dofsmut, :vars, :elements, fieldnames(GlobalBlockVector)...
)
function Base.getproperty(bv::GlobalBlockVector, s::Symbol)
    return if s == :dofs
        getfield(bv, :parent).dofs
    elseif s == :dofsmut
        getfield(bv, :parent).dofsmut
    elseif s == :vars
        getfield(bv, :parent).vars
    elseif s == :elements
        GBVElements(bv)
    else
        getfield(bv, s)
    end
end

#==========================================================================================#
#                                     Face state vector                                    #

struct FaceStateVector{NV,RT}
    sides::NTuple{2,StateVector{NV,Matrix{RT}}}
    data::NTuple{2,Matrix{RT}}
    dh::DofHandler
    function FaceStateVector{NV}(data::NTuple{2,Matrix{RT}}, dh::DofHandler) where {NV,RT}
        sides = (StateVector{NV}(data[1]), StateVector{NV}(data[2]))
        ndofs(sides[1]) == nfacedofs(dh) || throw(DimensionMismatch())
        ndofs(sides[2]) == nfacedofs(dh) || throw(DimensionMismatch())
        return new{NV,RT}(sides, data, dh)
    end
end

function FaceStateVector(data::NTuple{2,Matrix{<:Any}}, dh::DofHandler)
    nv = size(data[1], 2)
    size(data[2], 2) == nv || throw(DimensionMismatch())
    return FaceStateVector{nv}(data, dh)
end

function FaceStateVector{NV,RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    dh::DofHandler,
) where {
    NV,
    RT
}
    sides = (
        StateVector{NV,RT}(value, nfacedofs(dh)),
        StateVector{NV,RT}(value, nfacedofs(dh)),
    )
    return FaceStateVector{NV}((sides[1].data, sides[2].data), dh)
end

function Base.similar(s::FaceStateVector, ::Type{T}, dims::Dims) where {T}
    sides = (similar(s.sides[1], T, dims), similar(s.sides[2], T, dims))
    return FaceStateVector((sides[1].data, sides[2].data), s.dh)
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

function Base.fill!(s::FaceStateVector, v)
    fill!(s.data[1], v)
    fill!(s.data[2], v)
    return s
end

FlouCommon.datatype(s::FaceStateVector) = eltype(s.data[1])

FlouCommon.nfaces(s::FaceStateVector) = nfaces(s.dh)
FlouCommon.nvariables(::FaceStateVector{NV}) where {NV} = NV
nfacedofs(s::FaceStateVector) = nfacedofs(s.dh)
nfacedofs(s::FaceStateVector, face::Int) = nfacedofs(s.dh, face)

FlouCommon.eachface(s::FaceStateVector) = Base.OneTo(nfaces(s))
FlouCommon.eachvariable(s::FaceStateVector) = Base.OneTo(nvariables(s))
eachfacedof(s::FaceStateVector) = Base.OneTo(nfacedofs(s))
eachfacedof(s::FaceStateVector, face::Int) = Base.OneTo(nfacedofs(s, face))

struct FSVFaceSide{NV,RT}
    s::FaceStateVector{NV,RT}
    face::Int
    side::Int
end
Base.propertynames(::FSVFaceSide) = (:dofs, :dofsmut, :vars, fieldnames(FSVFaceSide)...)
function Base.getproperty(f::FSVFaceSide{NV}, s::Symbol) where {NV}
    @inbounds return if s == :dofs
        sv = getfield(f, :s)
        face = getfield(f, :face)
        side = getfield(f, :side)
        i1 = facedofid(sv.dh, face, 1)
        i2 = facedofid(sv.dh, face, nfacedofs(sv.dh, face))
        StateVector{NV}(view(sv.data[side], i1:i2, :)).dofs
    elseif s == :dofsmut
        sv = getfield(f, :s)
        face = getfield(f, :face)
        side = getfield(f, :side)
        i1 = facedofid(sv.dh, face, 1)
        i2 = facedofid(sv.dh, face, nfacedofs(sv.dh, face))
        StateVector{NV}(view(sv.data[side], i1:i2, :)).dofsmut
    elseif s == :vars
        sv = getfield(f, :s)
        face = getfield(f, :face)
        side = getfield(f, :side)
        i1 = facedofid(sv.dh, face, 1)
        i2 = facedofid(sv.dh, face, nfacedofs(sv.dh, face))
        StateVector{NV}(view(sv.data[side], i1:i2, :)).vars
    else
        getfield(f, s)
    end
end

struct FSVFace{NV,RT}
    s::FaceStateVector{NV,RT}
    face::Int
end
Base.propertynames(::FSVFace) = (:sides, fieldnames(FSVFace)...)
function Base.getproperty(f::FSVFace, s::Symbol)
    return if s == :sides
        sv = getfield(f, :s)
        face = getfield(f, :face)
        (FSVFaceSide(sv, face, 1), FSVFaceSide(sv, face, 2))
    else
        getfield(f, s)
    end
end

struct FSVFaces{NV,RT} <: AbstractVector{FSVFace{NV,RT}}
    s::FaceStateVector{NV,RT}
end
@inline function Base.size(f::FSVFaces)
    return (nfaces(f.s),)
end
@inline function Base.getindex(f::FSVFaces{NV}, i::Int) where {NV}
    @boundscheck checkbounds(f, i)
    return FSVFace(f.s, i)
end

Base.propertynames(::FaceStateVector) = (:faces, fieldnames(FaceStateVector)...)
function Base.getproperty(sv::FaceStateVector, s::Symbol)
    return if s == :faces
        FSVFaces(sv)
    else
        getfield(sv, s)
    end
end

#==========================================================================================#
#                                     Face block vector                                    #

struct FaceBlockVector{NV,RT}
    sides::NTuple{2,BlockVector{NV,Array{RT,3}}}
    data::NTuple{2,Array{RT,3}}
    dh::DofHandler
    function FaceBlockVector{NV}(
        data::NTuple{2,Array{RT,3}},
        dh::DofHandler,
    ) where {
        NV,
        RT
    }
        sides = (BlockVector{NV}(data[1]), BlockVector{NV}(data[2]))
        ndofs(sides[1]) == nfacedofs(dh) || throw(DimensionMismatch())
        ndofs(sides[2]) == nfacedofs(dh) || throw(DimensionMismatch())
        return new{NV,RT}(sides, data, dh)
    end
end

function FaceBlockVector(data::NTuple{2,Array{<:Any,3}}, dh::DofHandler)
    nv = size(data[1], 2)
    nd = size(data, 3)
    size(data[2], 2) == nv || throw(DimensionMismatch())
    size(data[2], 3) == nd || throw(DimensionMismatch())
    return FaceBlockVector{nv,nd}(data, dh)
end

function FaceBlockVector{NV,RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    dh::DofHandler,
    ndims::Int,
) where {
    NV,
    RT
}
    sides = (
        BlockVector{NV,RT}(value, nfacedofs(dh), ndims),
        BlockVector{NV,RT}(value, nfacedofs(dh), ndims),
    )
    return FaceBlockVector{NV}((sides[1].data, sides[2].data), dh)
end

function Base.similar(b::FaceBlockVector, ::Type{T}, dims::Dims) where {T}
    sides = (similar(b.sides[1], T, dims), similar(b.sides[2], T, dims))
    return FaceBlockVector((sides[1].data, sides[2].data), b.dh)
end

function Base.show(io::IO, ::MIME"text/plain", b::FaceBlockVector)
    @nospecialize
    rt = datatype(b)
    nvars = nvariables(b)
    nface = nfaces(b)
    dim = spatialdim(b)
    vstr = (nvars == 1) ? " variable" : " variables"
    print(io, nface, "-face ", dim, "D FaceBlockVector{", rt, "} with ", nvars, vstr)
    return nothing
end

function Base.fill!(b::FaceBlockVector, v)
    fill!(b.data[1], v)
    fill!(b.data[2], v)
    return b
end

FlouCommon.datatype(b::FaceBlockVector) = eltype(b.data[1])

FlouCommon.nfaces(b::FaceBlockVector) = nfaces(b.dh)
FlouCommon.nvariables(::FaceBlockVector{NV}) where {NV} = NV
nfacedofs(b::FaceBlockVector) = nfacedofs(b.dh)
nfacedofs(b::FaceBlockVector, face::Int) = nfacedofs(b.dh, face)

FlouCommon.eachface(b::FaceBlockVector) = Base.OneTo(nfaces(b))
FlouCommon.eachvariable(b::FaceBlockVector) = Base.OneTo(nvariables(b))
eachfacedof(b::FaceBlockVector) = Base.OneTo(nfacedofs(b))
eachfacedof(b::FaceBlockVector, face::Int) = Base.OneTo(nfacedofs(b, face))

FlouCommon.spatialdim(b::FaceBlockVector) = spatialdim(b.sides[1])
FlouCommon.eachdim(b::FaceBlockVector) = Base.OneTo(spatialdim(b))

struct FBVFaceSide{NV,RT}
    b::FaceBlockVector{NV,RT}
    face::Int
    side::Int
end
Base.propertynames(::FBVFaceSide) = (:dofs, :dofsmut, :vars, fieldnames(FBVFaceSide)...)
function Base.getproperty(f::FBVFaceSide{NV}, s::Symbol) where {NV}
    @inbounds return if s == :dofs
        bv = getfield(f, :b)
        face = getfield(f, :face)
        side = getfield(f, :side)
        i1 = facedofid(bv.dh, face, 1)
        i2 = facedofid(bv.dh, face, nfacedofs(bv.dh, face))
        BlockVector{NV}(view(bv.data[side], i1:i2, :, :)).dofs
    elseif s == :dofsmut
        bv = getfield(f, :b)
        face = getfield(f, :face)
        side = getfield(f, :side)
        i1 = facedofid(bv.dh, face, 1)
        i2 = facedofid(bv.dh, face, nfacedofs(bv.dh, face))
        BlockVector{NV}(view(bv.data[side], i1:i2, :, :)).dofsmut
    elseif s == :vars
        bv = getfield(f, :b)
        face = getfield(f, :face)
        side = getfield(f, :side)
        i1 = facedofid(bv.dh, face, 1)
        i2 = facedofid(bv.dh, face, nfacedofs(bv.dh, face))
        BlockVector{NV}(view(bv.data[side], i1:i2, :, :)).vars
    else
        getfield(f, s)
    end
end

struct FBVFace{NV,RT}
    b::FaceBlockVector{NV,RT}
    face::Int
end
Base.propertynames(::FBVFace) = (:sides, fieldnames(FBVFace)...)
function Base.getproperty(f::FBVFace, s::Symbol)
    return if s == :sides
        bv = getfield(f, :b)
        face = getfield(f, :face)
        (FBVFaceSide(bv, face, 1), FBVFaceSide(bv, face, 2))
    else
        getfield(f, s)
    end
end

struct FBVFaces{NV,RT} <: AbstractVector{FBVFace{NV,RT}}
    b::FaceBlockVector{NV,RT}
end
@inline function Base.size(f::FBVFaces)
    return (nfaces(f.b),)
end
@inline function Base.getindex(f::FBVFaces, i::Int)
    @boundscheck checkbounds(f, i)
    return FBVFace(f.b, i)
end

Base.propertynames(::FaceBlockVector) = (:faces, fieldnames(FaceBlockVector)...)
function Base.getproperty(bv::FaceBlockVector, s::Symbol)
    return if s == :faces
        FBVFaces(bv)
    else
        getfield(bv, s)
    end
end
