#==========================================================================================#
#                                       State vector                                       #

struct StateVector{RT<:Real,R<:AbstractMatrix{RT}} <: AbstractVector{RT}
    data::R
    dh::DofHandler
    function StateVector(data::AbstractMatrix, dh::DofHandler)
        size(data, 1) == ndofs(dh) || throw(DimensionMismatch(
            "The number of rows of data must equal the number of dofs in dh"
        ))
        r = typeof(data)
        rt = eltype(data)
        return new{rt,r}(data, dh)
    end
end

function Base.similar(s::StateVector)
    data = similar(s.data)
    return similar(data, s)
end

function Base.similar(data, s::StateVector)
    return if ndims(data) == 1
        StateVector(reshape(data, (:, 1)), s.dh)
    else
        StateVector(data, s.dh)
    end
end

function StateVector{RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    nvars::Integer,
    dh::DofHandler,
) where {
    RT,
}
    (; elem_offsets) = dh
    data = Matrix{RT}(value, last(elem_offsets), nvars)
    return StateVector(data, dh)
end

function Base.show(io::IO, ::MIME"text/plain", s::StateVector{RT}) where {RT}
    @nospecialize
    nvars = nvariables(s)
    nelem = nelements(s)
    print(io, "StateVector{", RT, "} with ", nelem, " elements and ", nvars, " variables")
    return nothing
end

Base.length(s::StateVector) = nelements(s.dh)
Base.size(s::StateVector) = (length(s),)
Base.fill!(s::StateVector, v) = fill!(s.data, v)

@inline function Base.getindex(s::StateVector, i::Integer)
    @boundscheck checkbounds(s, i)
    return @inbounds view(s.data, (s.dh.elem_offsets[i] + 1):s.dh.elem_offsets[i + 1], :)
end

nelements(s::StateVector) = nelements(s.dh)
nvariables(s::StateVector) = size(s.data, 2)
ndofs(s::StateVector) = ndofs(s.dh)

eachelement(s::StateVector) = Base.OneTo(nelements(s))
eachvariable(s::StateVector) = Base.OneTo(nvariables(s))
eachdof(s::StateVector) = Base.OneTo(ndofs(s))

#==========================================================================================#
#                                       Block vector                                       #

struct BlockVector{RT<:Real,R<:AbstractArray{RT,3}} <: AbstractVector{RT}
    data::R
    dh::DofHandler
    function BlockVector(data::AbstractArray, dh::DofHandler)
        size(data, 1) == ndofs(dh) || throw(DimensionMismatch(
            "The number of rows of data must equal the number of dofs in dh"
        ))
        r = typeof(data)
        rt = eltype(data)
        return new{rt,r}(data, dh)
    end
end

function Base.similar(b::BlockVector)
    data = similar(b.data)
    return similar(data, b)
end

function Base.similar(data, b::BlockVector)
    return if ndims(data) == 1
        BlockVector(reshape(data, (:, 1, 1)), b.dh)
    elseif ndims(data) == 2
        BlockVector(reshape(data, (:, size(data, 2), 1)), b.dh)
    else
        BlockVector(data, b.dh)
    end
end

function BlockVector{RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    nvars::Integer,
    ndims::Integer,
    dh::DofHandler,
) where {
    RT,
}
    (; elem_offsets) = dh
    data = Array{RT,3}(value, last(elem_offsets), nvars, ndims)
    return BlockVector(data, dh)
end

function Base.show(io::IO, ::MIME"text/plain", b::BlockVector{RT}) where {RT}
    @nospecialize
    nvars = nvariables(b)
    nelem = nelements(b)
    dim = ndims(b)
    print(
        io,
        dim, "D BlockVector{", RT, "} with ", nelem, " elements and ", nvars, " variables",
    )
    return nothing
end

Base.length(b::BlockVector) = nelements(b.dh)
Base.size(b::BlockVector) = (length(b),)
Base.fill!(b::BlockVector, v) = fill!(b.data, v)

@inline function Base.getindex(b::BlockVector, i::Integer)
    @boundscheck checkbounds(b, i)
    return @inbounds view(b.data, (b.dh.elem_offsets[i] + 1):b.dh.elem_offsets[i + 1], :, :)
end

nelements(b::BlockVector) = nelements(b.dh)
nvariables(b::BlockVector) = size(b.data, 2)
ndims(b::BlockVector) = size(b.data, 3)
ndofs(b::BlockVector) = ndofs(b.dh)

eachelement(b::BlockVector) = Base.OneTo(nelements(b))
eachvariable(b::BlockVector) = Base.OneTo(nvariables(b))
eachdim(b::BlockVector) = Base.OneTo(ndims(b))
eachdof(b::BlockVector) = Base.OneTo(ndofs(b))

#==========================================================================================#
#                                     Face state vector                                    #

struct FaceStateVector{RT<:Real,R<:AbstractMatrix{RT}} <: AbstractVector{RT}
    data::R
    dh::DofHandler
    function FaceStateVector(data::AbstractMatrix, dh::DofHandler)
        size(data, 1) == nfacedofs(dh) || throw(DimensionMismatch(
            "The number of rows of data must equal the number of face dofs in dh"
        ))
        r = typeof(data)
        rt = eltype(data)
        return new{rt,r}(data, dh)
    end
end

function Base.similar(s::FaceStateVector)
    data = similar(s.data)
    return similar(data, s)
end

function Base.similar(data, s::FaceStateVector)
    return if ndims(data) == 1
        FaceStateVector(reshape(data, (:, 1)), s.dh)
    else
        FaceStateVector(data, s.dh)
    end
end

function FaceStateVector{RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    nvars::Integer,
    dh::DofHandler,
) where {
    RT,
}
    data = Matrix{RT}(value, nfacedofs(dh), nvars)
    return FaceStateVector(data, dh)
end

function Base.show(io::IO, ::MIME"text/plain", s::FaceStateVector{RT}) where {RT}
    @nospecialize
    nvars = nvariables(s)
    nface = nfaces(s)
    print(io, "FaceStateVector{", RT, "} with ", nface, " faces and ", nvars, " variables")
    return nothing
end

Base.length(s::FaceStateVector) = nfaces(s.dh)
Base.size(s::FaceStateVector) = (length(s),)
Base.fill!(s::FaceStateVector, v) = fill!(s.data, v)

@inline function Base.getindex(s::FaceStateVector, i::Integer)
    @boundscheck checkbounds(s, i)
    @inbounds begin
        i1 = 2 * s.dh.face_offsets[i] + 1
        i2 = (i1 - 1) + s.dh.face_offsets[i + 1] - s.dh.face_offsets[i]
        i3 = 2 * s.dh.face_offsets[i + 1]
        return (
            view(s.data, i1:i2, :),
            view(s.data, (i2 + 1):i3, :),
        )
    end
end

nfaces(s::FaceStateVector) = nfaces(s.dh)
nvariables(s::FaceStateVector) = size(s.data, 2)

eachsace(s::FaceStateVector) = Base.OneTo(nfaces(s))
eachvariable(s::FaceStateVector) = Base.OneTo(nvariables(s))

#==========================================================================================#
#                                     Face block vector                                    #

struct FaceBlockVector{RT<:Real,R<:AbstractArray{RT,3}} <: AbstractVector{RT}
    data::R
    dh::DofHandler
    function FaceBlockVector(data::AbstractArray, dh::DofHandler)
        size(data, 1) == nfacedofs(dh) || throw(DimensionMismatch(
            "The number of rows of data must equal the number of face dofs in dh"
        ))
        r = typeof(data)
        rt = eltype(data)
        return new{rt,r}(data, dh)
    end
end

function Base.similar(b::FaceBlockVector)
    data = similar(b.data)
    return similar(data, b)
end

function Base.similar(data, b::FaceBlockVector)
    return if ndims(data) == 1
        FaceBlockVector(reshape(data, (:, 1, 1)), b.dh)
    elseif ndims(data) == 2
        FaceBlockVector(reshape(data, (:, size(data, 2), 1)), b.dh)
    else
        FaceBlockVector(data, b.dh)
    end
end

function FaceBlockVector{RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    nvars::Integer,
    ndims::Integer,
    dh::DofHandler,
) where {
    RT,
}
    data = Array{RT,3}(value, nfacedofs(dh), nvars, ndims)
    return FaceBlockVector(data, dh)
end

function Base.show(io::IO, ::MIME"text/plain", s::FaceBlockVector{RT}) where {RT}
    @nospecialize
    nvars = nvariables(s)
    nface = nfaces(s)
    dim = ndims(s)
    print(
        io,
        dim, "D FaceBlockVector{", RT, "} with ", nface, " faces and ", nvars, " variables",
    )
    return nothing
end

Base.length(b::FaceBlockVector) = nfaces(b.dh)
Base.size(b::FaceBlockVector) = (length(b),)
Base.fill!(b::FaceBlockVector, v) = fill!(b.data, v)

@inline function Base.getindex(b::FaceBlockVector, i::Integer)
    @boundscheck checkbounds(b, i)
    @inbounds begin
        i1 = 2 * b.dh.face_offsets[i] + 1
        i2 = (i1 - 1) + b.dh.face_offsets[i + 1] - b.dh.face_offsets[i]
        i3 = 2 * b.dh.face_offsets[i + 1]
        return (
            view(b.data, i1:i2, :, :),
            view(b.data, (i2 + 1):i3, :, :),
        )
    end
end

nfaces(b::FaceBlockVector) = nfaces(b.dh)
nvariables(b::FaceBlockVector) = size(b.data, 2)
ndims(b::FaceBlockVector) = size(b.data, 3)

eachsace(b::FaceBlockVector) = Base.OneTo(nfaces(b))
eachvariable(b::FaceBlockVector) = Base.OneTo(nvariables(b))
eachdim(b::FaceBlockVector) = Base.OneTo(ndims(b))
