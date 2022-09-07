#==========================================================================================#
#                                       State vector                                       #

struct StateVector{RT<:Real,R<:AbstractVector{RT},S} <: AbstractVector{RT}
    raw::R
    data::Vector{S}  # [ndofs, nvars, nelements] × nregions
end

function Base.similar(s::StateVector)
    raw = similar(s.raw)
    return similar(raw, s)
end

function Base.similar(raw, s::StateVector)
    data = similar(s.data)
    for i in eachindex(data, s.data)
        dims = s.data[i].dims
        inds = s.data[i].parent.indices
        data[i] = reshape(
            view(raw, inds...),
            dims,
        )
    end
    return StateVector(raw, data)
end

function StateVector{RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    dims::Vararg{<:Integer,4},
) where {
    RT,
}
    raw = Vector{RT}(value, prod(dims))
    return StateVector(raw, dims...)
end

function StateVector(raw, dims::Vararg{<:Integer,4})
    ndofs, nvars, nelems, nregions = dims
    return StateVector(
        raw,
        fill(ndofs, nregions),
        fill(nelems, nregions),
        nvars,
    )
end

function StateVector{RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    ndofs::AbstractVector{<:Integer},
    nelements::AbstractVector{<:Integer},
    nvars::Integer,
) where {
    RT,
}
    rawlen = sum(ndofs .* nvars .* nelements)
    raw = Vector{RT}(value, rawlen)
    return StateVector(raw, ndofs, nelements, nvars)
end

function StateVector(
    raw::AbstractVector,
    ndofs::AbstractVector{<:Integer},
    nelements::AbstractVector{<:Integer},
    nvars::Integer
)
    nregions = length(ndofs)
    blocklen = ndofs[1] * nvars * nelements[1]
    data = [
        reshape(
            view(raw, 1:blocklen),
            (ndofs[1], nvars, nelements[1]),
        )
    ]
    offset = blocklen
    for i in 2:nregions
        blocklen = ndofs[i] * nvars * nelements[i]
        lims = (offset + 1):(offset + blocklen)
        push!(
            data,
            reshape(
                view(raw, lims),
                (ndofs[i], nvars, nelements[i]),
            ),
        )
        offset += blocklen
    end
    return StateVector(raw, data)
end

function Base.show(io::IO, ::MIME"text/plain", s::StateVector{RT}) where {RT}
    @nospecialize
    nvars = nvariables(s)
    nelem = nelements(s)
    nreg = nregions(s)
    print(io, nelem, "-element state vector with ", nvars, " variable(s), ",
        nreg, " region(s) and eltype ", RT, ".")
    return nothing
end

Base.size(s::StateVector) = size(s.data)
Base.getindex(s::StateVector, i) = s.data[i]
Base.fill!(s::StateVector, v) = fill!(s.raw, v)

nregions(s::StateVector) = length(s.data)
nelements(s::StateVector) = sum(size.(s.data, 3))
nelements(s::StateVector, i) = size(s.data[i], 3)
nvariables(s::StateVector) = size(s.data[1], 2)

eachregion(s::StateVector) = Base.OneTo(nregions(s))
eachelement(s::StateVector) = Base.OneTo(nelements(s))
eachelement(s::StateVector, i) = Base.OneTo(nelements(s, i))
eachvariable(s::StateVector) = Base.OneTo(nvariables(s))

#==========================================================================================#
#                                    Mortar state vector                                   #

struct MortarStateVector{RT<:Real} <: AbstractVector{RT}
    data::Vector{NTuple{2,Matrix{RT}}}   # [[ndofs, nvar] × nsides] × nfaces
end

function Base.similar(m::MortarStateVector)
    data = similar(m.data)
    for i in eachindex(data, m.data)
        data[i] = (similar(m.data[i][1]), similar(m.data[i][2]))
    end
    return MortarStateVector(data)
end

function MortarStateVector{RT}(value, dims) where {RT}
    nfaces = length(dims)
    data = Vector{NTuple{2,Matrix{RT}}}(undef, nfaces)
    for i in 1:nfaces
        data[i] = (Matrix{RT}(value, dims[i][1]...), Matrix{RT}(value, dims[i][2]...))
    end
    return MortarStateVector(data)
end

function Base.show(io::IO, ::MIME"text/plain", m::MortarStateVector{RT}) where {RT}
    @nospecialize
    nface = length(m.data)
    nvars = size(m.data[1][1], 2)
    print(io, nface, " face mortar state vector with ", nvars,
        " variable(s) and eltype ", RT)
end
Base.size(m::MortarStateVector) = size(m.data)
Base.getindex(m::MortarStateVector, i::Int) = m.data[i]
Base.fill!(m::MortarStateVector, v) = fill!.(m.data, Ref(v))
