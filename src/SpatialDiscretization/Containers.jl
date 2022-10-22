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
