#==========================================================================================#
#                                       State vector                                       #

struct StateVector{NV,RT<:Real,D<:HybridVector{NV,RT}} <: AbstractMatrix{RT}
    data::D
    dh::DofHandler
    function StateVector(data::HybridVector, dh::DofHandler)
        length(data) == ndofs(dh) || throw(DimensionMismatch(
            "The length of `data` must be equal to the number of degrees of freedom"
        ))
        nv = innerdim(data)
        rt = datatype(data)
        return new{nv,rt,typeof(data)}(data, dh)
    end
end

struct LazyStateVectorDof{S<:StateVector}
    s::S
end
@inline function Base.getindex(d::LazyStateVectorDof, i::Int)
    @boundscheck checkbounds(d.s.data,i)
    return @inbounds d.s.data[i]
end
@inline function Base.setindex!(d::LazyStateVectorDof, v, i::Int)
    @boundscheck checkbounds(d.s.data,i)
    @inbounds d.s.data[i] = v
end

struct LazyStateVectorElement{S<:StateVector}
    s::S
end
@inline function Base.getindex(e::LazyStateVectorElement, i::Int)
    @boundscheck checkindex(Bool, 1:nelements(e.s.dh), i) || throw(ErrorException(
        "Bounds Error: attempt to access $(nelements(e.s.dh))-element \
        StateVector at index [$i]"
    ))
    @inbounds begin
        i1 = e.s.dh.elem_offsets[i] + 1
        i2 = e.s.dh.elem_offsets[i + 1]
        return view(e.s.data, i1:i2)
    end
end
@inline function Base.setindex!(e::LazyStateVectorElement, v, i::Int)
    @boundscheck checkindex(Bool, 1:nelements(e.s.dh), i) || throw(ErrorException(
        "Bounds Error: attempt to access $(nelements(e.s.dh))-element \
        StateVector at index [$i]"
    ))
    @inbounds begin
        i1 = e.s.dh.elem_offsets[i] + 1
        i2 = e.s.dh.elem_offsets[i + 1]
        e.s.data[i1:i2] = v
    end
end

struct LazyStateVectorVar{S<:StateVector}
    s::S
end
@inline function Base.getindex(v::LazyStateVectorVar, i::Int)
    @boundscheck checkbounds(v.s.data.flat, i, :)
    return @inbounds view(v.s.data.flat, i, :)
end
@inline function Base.setindex!(sv::LazyStateVectorVar, v, i::Int)
    @boundscheck checkbounds(sv.s.data.flat, i, :)
    @inbounds sv.s.data.flat[i, :] = v
end

Base.propertynames(::StateVector) = (:data, :flat, :dof, :element, :var, :dh)
Base.@propagate_inbounds function Base.getproperty(sv::StateVector, s::Symbol)
    return if s == :flat
        sv.data.flat
    elseif s == :dof
        LazyStateVectorDof(sv)
    elseif s == :element
        LazyStateVectorElement(sv)
    elseif s == :var
        LazyStateVectorVar(sv)
    else
        getfield(sv, s)
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

function Base.similar(s::StateVector, ::Type{T}, dims::Dims) where {T}
    data = similar(s.data, T, Base.tail(dims))
    return StateVector(data, s.dh)
end

function Base.show(io::IO, ::MIME"text/plain", s::StateVector)
    @nospecialize
    rt = eltype(s)
    nvars = nvariables(s)
    nelem = nelements(s)
    vstr = (nvars == 1) ? " variable" : " variables"
    print(io, nelem, "-element StateVector{", rt, "} with ", nvars, vstr)
    return nothing
end

Base.IndexStyle(::Type{<:StateVector}) = IndexLinear()
Base.size(s::StateVector) = size(s.data.flat)

function Base.fill!(s::StateVector, v)
    fill!(s.data.flat, v)
    return s
end

@inline function Base.getindex(s::StateVector, i::Integer)
    @boundscheck checkbounds(s, i)
    return @inbounds s.data.flat[i]
end

@inline function Base.setindex!(s::StateVector, v, i::Integer)
    @boundscheck checkbounds(s, i)
    @inbounds s.data.flat[i] = v
end

FlouCommon.nelements(s::StateVector) = nelements(s.dh)
FlouCommon.nvariables(s::StateVector) = innerdim(s.data)
ndofs(s::StateVector) = length(s.data)

FlouCommon.eachelement(s::StateVector) = Base.OneTo(nelements(s))
FlouCommon.eachvariable(s::StateVector) = Base.OneTo(nvariables(s))
eachdof(s::StateVector) = Base.OneTo(ndofs(s))

#==========================================================================================#
#                                       Block vector                                       #

struct BlockVector{NV,RT<:Real,D<:HybridMatrix{NV,RT}} <: AbstractArray{RT,3}
    data::D
    dh::DofHandler
    function BlockVector(data::HybridMatrix, dh::DofHandler)
        size(data, 1) == ndofs(dh) || throw(DimensionMismatch(
            "The first dimension of `data` must equal the number of dofs in dh"
        ))
        nv = innerdim(data)
        rt = datatype(data)
        return new{nv,rt,typeof(data)}(data, dh)
    end
end

struct LazyBlockVectorDof{B<:BlockVector}
    b::B
end
@inline function Base.getindex(d::LazyBlockVectorDof, i::Int)
    @boundscheck checkbounds(d.b.data,i)
    return @inbounds view(d.b.data, i, :)
end
@inline function Base.setindex!(d::LazyBlockVectorDof, v, i::Int)
    @boundscheck checkbounds(d.b.data,i)
    @inbounds d.b.data[i, :] = v
end

struct LazyBlockVectorElement{B<:BlockVector}
    b::B
end
@inline function Base.getindex(e::LazyBlockVectorElement, i::Int)
    @boundscheck checkindex(Bool, 1:nelements(e.b.dh), i) || throw(ErrorException(
        "Bounds Error: attempt to access $(nelements(e.b.dh))-element \
        BlockVector at index [$i]"
    ))
    @inbounds begin
        i1 = e.b.dh.elem_offsets[i] + 1
        i2 = e.b.dh.elem_offsets[i + 1]
        return view(e.b.data, i1:i2, :)
    end
end
@inline function Base.setindex!(e::LazyBlockVectorElement, v, i::Int)
    @boundscheck checkindex(Bool, 1:nelements(e.b.dh), i) || throw(ErrorException(
        "Bounds Error: attempt to access $(nelements(e.b.dh))-element \
        BlockVector at index [$i]"
    ))
    @inbounds begin
        i1 = e.b.dh.elem_offsets[i] + 1
        i2 = e.b.dh.elem_offsets[i + 1]
        e.b.data[i1:i2, :] = v
    end
end

struct LazyBlockVectorVar{B<:BlockVector}
    b::B
end
@inline function Base.getindex(v::LazyBlockVectorVar, i::Int)
    @boundscheck checkbounds(v.b.data.flat, i, :, :)
    return @inbounds view(v.b.data.flat, i, :, :)
end
@inline function Base.setindex!(sv::LazyBlockVectorVar, v, i::Int)
    @boundscheck checkbounds(sv.b.data.flat, i, :, :)
    @inbounds sv.b.data.flat[i, :, :] = v
end

struct LazyBlockVectorDim{B<:BlockVector}
    b::B
end
@inline function Base.getindex(d::LazyBlockVectorDim, i::Int)
    @boundscheck checkbounds(d.b.data, :, i)
    return @inbounds view(d.b.data, :, i)
end
@inline function Base.setindex!(d::LazyBlockVectorDim, v, i::Int)
    @boundscheck checkbounds(d.b.data, :, i)
    @inbounds d.b.data[:, i] = v
end

Base.propertynames(::BlockVector) = (:data, :flat, :dof, :element, :var, :dim, :dh)
Base.@propagate_inbounds function Base.getproperty(b::BlockVector, s::Symbol)
    return if s == :flat
        b.data.flat
    elseif s == :dof
        LazyBlockVectorDof(b)
    elseif s == :element
        LazyBlockVectorElement(b)
    elseif s == :var
        LazyBlockVectorVar(b)
    elseif s == :dim
        LazyBlockVectorDim(b)
    else
        getfield(b, s)
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

function Base.similar(b::BlockVector, ::Type{T}, dims::Dims) where {T}
    data = similar(b.data, T, Base.tail(dims))
    return BlockVector(data, b.dh)
end

function Base.show(io::IO, ::MIME"text/plain", b::BlockVector)
    @nospecialize
    rt = eltype(b)
    nvars = nvariables(b)
    nelem = nelements(b)
    dim = spatialdim(b)
    vstr = (nvars == 1) ? " variable" : " variables"
    print(io, nelem, "-element ", dim, "D BlockVector{", rt, "} with ", nvars, vstr)
    return nothing
end

Base.IndexStyle(::Type{<:BlockVector}) = Base.IndexLinear()
Base.size(b::BlockVector) = size(b.data.flat)

function Base.fill!(b::BlockVector, v)
    fill!(b.data.flat, v)
    return b
end

@inline function Base.getindex(b::BlockVector, i::Integer)
    @boundscheck checkbounds(b, i)
    return @inbounds b.data.flat[i]
end

@inline function Base.setindex!(b::BlockVector, v, i::Integer)
    @boundscheck checkbounds(b, i)
    @inbounds b.data.flat[i] = v
end

FlouCommon.nelements(b::BlockVector) = nelements(b.dh)
FlouCommon.nvariables(b::BlockVector) = innerdim(b.data)
ndofs(b::BlockVector) = size(b.data, 1)

FlouCommon.eachelement(b::BlockVector) = Base.OneTo(nelements(b))
FlouCommon.eachvariable(b::BlockVector) = Base.OneTo(nvariables(b))
eachdof(b::BlockVector) = Base.OneTo(ndofs(b))

FlouCommon.spatialdim(b::BlockVector) = size(b.data, 2)
FlouCommon.eachdim(b::BlockVector) = Base.OneTo(spatialdim(b))

#==========================================================================================#
#                                     Face state vector                                    #

struct FaceStateVector{NV,RT<:Real,D<:HybridVector{NV,RT}} <: AbstractMatrix{RT}
    data::D
    dh::DofHandler
    function FaceStateVector(data::HybridVector, dh::DofHandler)
        length(data) == nfacedofs(dh) || throw(DimensionMismatch(
            "The length of `data` must equal the number of face dofs in dh"
        ))
        nv = innerdim(data)
        rt = datatype(data)
        return new{nv,rt,typeof(data)}(data, dh)
    end
end

struct LazyFaceStateVectorDof{S<:FaceStateVector}
    s::S
end
@inline function Base.getindex(d::LazyFaceStateVectorDof, i::Int)
    @boundscheck checkindex(Bool, 1:nfacedofs(d.s.dh), i) || throw(ErrorException(
        "Bounds Error: attempt to access $(nfacedofs(d.s.dh))-DOF \
        FaceStateVector at index [$i]"
    ))
    @inbounds begin
        i1 = 2 * d.s.dh.face_offsets[i] + 1
        i2 = i1 + d.s.dh.face_offsets[i + 1] - d.s.dh.face_offsets[i]
        return (d.s.data[i1], d.s.data[i2])
    end
end
@inline function Base.setindex!(d::LazyFaceStateVectorDof, v, i::Int)
    @boundscheck checkindex(Bool, 1:nfacedofs(d.s.dh), i) || throw(ErrorException(
        "Bounds Error: attempt to access $(nfacedofs(d.s.dh))-DOF \
        FaceStateVector at index [$i]"
    ))
    @inbounds begin
        i1 = 2 * d.s.dh.face_offsets[i] + 1
        i2 = i1 + d.s.dh.face_offsets[i + 1] - d.s.dh.face_offsets[i]
        d.s.data[i1] = v
        d.s.data[i2] = v
    end
end

struct LazyFaceStateVectorFace{S<:FaceStateVector}
    s::S
end
@inline function Base.getindex(f::LazyFaceStateVectorFace, i::Int)
    @boundscheck checkindex(Bool, 1:nfaces(f.s.dh), i) || throw(ErrorException(
        "Bounds Error: attempt to access $(nfaces(f.s.dh))-face \
        FaceStateVector at index [$i]"
    ))
    @inbounds begin
        i1 = 2 * f.s.dh.face_offsets[i] + 1
        i2 = (i1 - 1) + f.s.dh.face_offsets[i + 1] - f.s.dh.face_offsets[i]
        i3 = 2 * f.s.dh.face_offsets[i + 1]
        return (view(f.s.data, i1:i2), view(f.s.data, (i2 + 1):i3))
    end
end
@inline function Base.setindex!(f::LazyFaceStateVectorFace, v, i::Int)
    @boundscheck checkindex(Bool, 1:nfaces(f.s.dh), i) || throw(ErrorException(
        "Bounds Error: attempt to access $(nfaces(f.s.dh))-face \
        FaceStateVector at index [$i]"
    ))
    @inbounds begin
        i1 = 2 * f.s.dh.face_offsets[i] + 1
        i2 = (i1 - 1) + f.s.dh.face_offsets[i + 1] - f.s.dh.face_offsets[i]
        i3 = 2 * f.s.dh.face_offsets[i + 1]
        f.s.data[i1:i2] = v
        f.s.data[(i2 + 1):i3] = v
    end
end

struct LazyFaceStateVectorVar{S<:FaceStateVector}
    s::S
end
@inline function Base.getindex(v::LazyFaceStateVectorVar, i::Int)
    @boundscheck checkbounds(v.s.data.flat, i, :)
    return @inbounds view(v.s.data.flat, i, :)
end
@inline function Base.setindex!(sv::LazyFaceStateVectorVar, v, i::Int)
    @boundscheck checkbounds(sv.s.data.flat, i, :)
    @inbounds sv.s.data.flat[i, :] = v
end

Base.propertynames(::FaceStateVector) = (:data, :flat, :dof, :face, :var, :dh)
Base.@propagate_inbounds function Base.getproperty(fs::FaceStateVector, s::Symbol)
    return if s == :flat
        fs.data.flat
    elseif s == :dof
        LazyFaceStateVectorDof(fs)
    elseif s == :face
        LazyFaceStateVectorFace(fs)
    elseif s == :var
        LazyFaceStateVectorVar(fs)
    else
        getfield(fs, s)
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

function Base.similar(s::FaceStateVector, ::Type{T}, dims::Dims) where {T}
    data = similar(s.data, T, Base.tail(dims))
    return FaceStateVector(data, s.dh)
end

function Base.show(io::IO, ::MIME"text/plain", s::FaceStateVector)
    @nospecialize
    rt = eltype(s)
    nvars = nvariables(s)
    nface = nfaces(s)
    vstr = (nvars == 1) ? " variable" : " variables"
    print(io, nface, "-face FaceStateVector{", rt, "} with ", nvars, vstr)
    return nothing
end

Base.IndexStyle(::Type{<:FaceStateVector}) = IndexLinear()
Base.size(s::FaceStateVector) = size(s.data.flat)

function Base.fill!(s::FaceStateVector, v)
    fill!(s.data.flat, v)
    return s
end

@inline function Base.getindex(s::FaceStateVector, i::Integer)
    @boundscheck checkbounds(s, i)
    return @inbounds s.data.flat[i]
end

@inline function Base.setindex!(s::FaceStateVector, v, i::Integer)
    @boundscheck checkbounds(s, i)
    @inbounds s.data.flat[i] = v
end

FlouCommon.nfaces(s::FaceStateVector) = nfaces(s.dh)
FlouCommon.nvariables(s::FaceStateVector) = innerdim(s.data)

FlouCommon.eachface(s::FaceStateVector) = Base.OneTo(nfaces(s))
FlouCommon.eachvariable(s::FaceStateVector) = Base.OneTo(nvariables(s))

#==========================================================================================#
#                                     Face block vector                                    #

struct FaceBlockVector{NV,RT<:Real,D<:HybridMatrix{NV,RT}} <: AbstractArray{RT,3}
    data::D
    dh::DofHandler
    function FaceBlockVector(data::HybridMatrix, dh::DofHandler)
        size(data, 1) == nfacedofs(dh) || throw(DimensionMismatch(
            "The first dimension of `data` must equal the number of face dofs in dh"
        ))
        nv = innerdim(data)
        rt = datatype(data)
        return new{nv,rt,typeof(data)}(data, dh)
    end
end

struct LazyFaceBlockVectorDof{B<:FaceBlockVector}
    b::B
end
@inline function Base.getindex(d::LazyFaceBlockVectorDof, i::Int)
    @boundscheck checkindex(Bool, 1:nfacedofs(d.b.dh), i) || throw(ErrorException(
        "Bounds Error: attempt to access $(nfacedofs(d.b.dh))-DOF \
        FaceBlockVector at index [$i]"
    ))
    @inbounds begin
        i1 = 2 * d.b.dh.face_offsets[i] + 1
        i2 = i1 + d.b.dh.face_offsets[i + 1] - d.b.dh.face_offsets[i]
        return (view(d.b.data, i1, :), view(d.b.data, i2, :))
    end
end
@inline function Base.setindex!(d::LazyFaceBlockVectorDof, v, i::Int)
    @boundscheck checkindex(Bool, 1:nfacedofs(d.b.dh), i) || throw(ErrorException(
        "Bounds Error: attempt to access $(nfacedofs(d.b.dh))-DOF \
        FaceBlockVector at index [$i]"
    ))
    @inbounds begin
        i1 = 2 * d.b.dh.face_offsets[i] + 1
        i2 = i1 + d.b.dh.face_offsets[i + 1] - d.b.dh.face_offsets[i]
        d.b.data[i1, :] = v
        d.b.data[i2, :] = v
    end
end

struct LazyFaceBlockVectorElement{B<:FaceBlockVector}
    b::B
end
@inline function Base.getindex(e::LazyFaceBlockVectorElement, i::Int)
    @boundscheck checkindex(Bool, 1:nfaces(e.b.dh), i) || throw(ErrorException(
        "Bounds Error: attempt to access $(nfaces(e.b.dh))-face \
        FaceBlockVector at index [$i]"
    ))
    @inbounds begin
        i1 = 2 * e.b.dh.face_offsets[i] + 1
        i2 = (i1 - 1) + e.b.dh.face_offsets[i + 1] - e.b.dh.face_offsets[i]
        i3 = 2 * e.b.dh.face_offsets[i + 1]
        return (view(e.b.data, i1:i2, :), view(e.b.data, (i2 + 1):i3, :))
    end
end
@inline function Base.setindex!(e::LazyFaceBlockVectorElement, v, i::Int)
    @boundscheck checkindex(Bool, 1:nfaces(e.b.dh), i) || throw(ErrorException(
        "Bounds Error: attempt to access $(nfaces(e.b.dh))-face \
        FaceBlockVector at index [$i]"
    ))
    @inbounds begin
        i1 = 2 * e.b.dh.face_offsets[i] + 1
        i2 = (i1 - 1) + e.b.dh.face_offsets[i + 1] - e.b.dh.face_offsets[i]
        i3 = 2 * e.b.dh.face_offsets[i + 1]
        e.b.data[i1:i2, :] = v
        e.b.data[(i2 + 1):i3, :] = v
    end
end

struct LazyFaceBlockVectorVar{B<:FaceBlockVector}
    b::B
end
@inline function Base.getindex(v::LazyFaceBlockVectorVar, i)
    @boundscheck checkbounds(v.b.data.flat, i, :, :)
    @inbounds view(v.b.data.flat, i, :, :)
end
@inline function Base.setindex!(sv::LazyFaceBlockVectorVar, v, i)
    @boundscheck checkbounds(sv.b.data.flat, i, :, :)
    @inbounds sv.b.data.flat[i, :, :] = v
end

struct LazyFaceBlockVectorDim{B<:FaceBlockVector}
    b::B
end
@inline function Base.getindex(d::LazyFaceBlockVectorDim, i)
    @boundscheck checkbounds(d.b.data.flat, :, :, i)
    @inbounds view(d.b.data.flat, :, :, i)
end
@inline function Base.setindex!(d::LazyFaceBlockVectorDim, v, i)
    @boundscheck checkbounds(d.b.data.flat, :, :, i)
    @inbounds d.b.data.flat[:, :, i] = v
end

Base.propertynames(::FaceBlockVector) = (:data, :flat, :dof, :face, :var, :dim, :dh)
Base.@propagate_inbounds function Base.getproperty(b::FaceBlockVector, s::Symbol)
    return if s == :flat
        b.data.flat
    elseif s == :dof
        LazyFaceBlockVectorDof(b)
    elseif s == :face
        LazyFaceBlockVectorElement(b)
    elseif s == :var
        LazyFaceBlockVectorVar(b)
    elseif s == :dim
        LazyFaceBlockVectorDim(b)
    else
        getfield(b, s)
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

function Base.similar(b::FaceBlockVector, ::Type{T}, dims::Dims) where {T}
    data = similar(b.data, T, Base.tail(dims))
    return BlockVector(data, b.dh)
end

function Base.show(io::IO, ::MIME"text/plain", b::FaceBlockVector)
    @nospecialize
    rt = eltype(b)
    nvars = nvariables(b)
    nface = nfaces(b)
    dim = spatialdim(b)
    vstr = (nvars == 1) ? " variable" : " variables"
    print(io, nface, "-face ", dim, "D FaceBlockVector{", rt, "} with ", nvars, vstr)
    return nothing
end

Base.IndexStyle(::Type{<:FaceBlockVector}) = IndexLinear()
Base.size(b::FaceBlockVector) = size(b.data.flat)

function Base.fill!(b::FaceBlockVector, v)
    fill!(b.data.flat, v)
    return b
end

@inline function Base.getindex(b::FaceBlockVector, i::Integer)
    @boundscheck checkbounds(b, i)
    return @inbounds b.data.flat[i]
end

@inline function Base.setindex!(b::FaceBlockVector, v, i::Integer)
    @boundscheck checkbounds(b, i)
    @inbounds b.data.flat[i] = v
end

FlouCommon.nfaces(b::FaceBlockVector) = nfaces(b.dh)
FlouCommon.nvariables(b::FaceBlockVector) = innerdim(b.data)

FlouCommon.eachface(b::FaceBlockVector) = Base.OneTo(nfaces(b))
FlouCommon.eachvariable(b::FaceBlockVector) = Base.OneTo(nvariables(b))

FlouCommon.spatialdim(b::FaceBlockVector) = size(b.data, 2)
FlouCommon.eachdim(b::FaceBlockVector) = Base.OneTo(spatialdim(b))
