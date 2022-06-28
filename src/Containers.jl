#==========================================================================================#
#                                        DofHandler                                        #

struct DofHandler{ND,NDF}
    # Elements
    elemoffs::Vector{UInt}
    elemreg::Vector{UInt8}
    elemregsize::Vector{NTuple{ND,UInt8}}
    # Faces (no mortars, MPI...)
    faceoffs::Vector{UInt}
    facereg::Vector{UInt8}
    faceregsize::Vector{NTuple{NDF,UInt8}}
end

function DofHandler(
    elemregions, regionsizes,
    regionfacesizes, connectivities, sides, orientation,
)
    # Elements
    elemoffs = UInt[0]
    sizehint!(elemoffs, length(elemregions) + 1)
    for r in elemregions
        push!(elemoffs, last(elemoffs) + prod(regionsizes[r]))
    end
    _elemregions = UInt8.(elemregions)
    _regionsizes = [UInt8.(s) |> Tuple for s in regionsizes]

    # Faces
    faceoffs = UInt[0]
    facereg = UInt8[]
    faceregsize = Any[]
    sizehint!(faceoffs, length(connectivities) + 1)
    sizehint!(facereg, length(connectivities) + 1)
    for (i, c) in enumerate(connectivities)
        r = _elemregions[c[1]]
        s = regionfacesizes[r][sides[i][1]]
        push!(faceoffs, last(faceoffs) + prod(s))
        ir = findfirst(==(s), faceregsize)
        if isnothing(ir)
            push!(faceregsize, s)
            push!(facereg, length(faceregsize))
        else
            push!(facereg, ir)
        end
    end

    return DofHandler(
        elemoffs, _elemregions, _regionsizes,
        faceoffs, facereg, [faceregsize...],
    )
end

@inline Base.size(self::DofHandler, i::Integer) = self.elemregsize[self.elemreg[i]]

@inline nregions(self::DofHandler) = length(self.elemregsize)
@inline nfaceregions(self::DofHandler) = length(self.faceregsize)
@inline nelements(self::DofHandler) = length(self.elemoffs) - 1
@inline nfaces(self::DofHandler) = length(self.faceoffs) - 1
@inline ndofs(self::DofHandler) = last(self.elemoffs)
@inline ndofs(self::DofHandler, i::Integer) = prod(size(self, i))

@inline eachelement(self::DofHandler) = Base.OneTo(nelements(self))
@inline eachface(self::DofHandler) = Base.OneTo(nfaces(self))
@inline elem2region(self::DofHandler, i::Integer) = self.elemreg[i]
@inline face2region(self::DofHandler, i::Integer) = self.facereg[i]

#==========================================================================================#
#                                       Scalar field                                       #

struct ScalarField{ND,NDF,T}
    data::T
    dh::DofHandler{ND,NDF}
    function ScalarField(data::AbstractVector, dh::DofHandler{ND}) where {ND}
        return new{ND,ND - 1,typeof(data)}(data, dh)
    end
end

function ScalarField{T}(dh) where {T<:Number}
    data = Vector{T}(undef, ndofs(dh))
    return ScalarField(data, dh)
end

function Base.similar(self::ScalarField)
    data = similar(self.data)
    return ScalarField(data, self.dh)
end

function Base.similar(data, self::ScalarField)
    return ScalarField(data, self.dh)
end

@inline function Base.getindex(self::ScalarField, elem)
    dh = self.dh
    r = dh.elemreg[elem]
    return reshape(
        view(self.data, (dh.elemoffs[elem] + 1:dh.elemoffs[elem + 1])),
        dh.elemregsize[r],
    )
end

@inline Base.length(self::ScalarField) = length(self.data)
@inline Base.size(self::ScalarField) = size(self.data)
@inline function Base.iterate(self::ScalarField, state=1)
    return state > length(self.data) ? nothing : (self.data[state], state + 1)
end

#==========================================================================================#
#                                       Vector field                                       #

struct VectorField{ND,NDF,T}
    data::T
    dh::DofHandler{ND,NDF}
    function VectorField(data::AbstractVector, dh::DofHandler{ND}) where {ND}
        return new{ND,ND - 1,typeof(data)}(data, dh)
    end
end

function VectorField{T}(dh::DofHandler{ND}) where {ND,T<:Number}
    data = [MVector{ND,T}(undef) for _ in 1:(ndofs(dh))]
    return VectorField(data, dh)
end

function Base.similar(self::VectorField)
    data = similar(self.data)
    return VectorField(data, self.dh)
end

function Base.similar(data, self::VectorField)
    return VectorField(data, self.dh)
end

@inline function Base.getindex(self::VectorField{ND}, elem) where {ND}
    dh = self.dh
    r = dh.elemreg[elem]
    return reshape(
        view(self.data, (dh.elemoffs[elem] + 1:dh.elemoffs[elem + 1])),
        dh.elemregsize[r],
    )
end

@inline Base.length(self::VectorField) = length(self.data)
@inline Base.size(self::VectorField) = size(self.data)
@inline function Base.iterate(self::VectorField, state=1)
    return state > length(self.data) ? nothing : (self.data[state], state + 1)
end

#==========================================================================================#
#                                       State vector                                       #

struct StateVector{T,V}
    data::T
    vars::V
    function StateVector(data::AbstractVector, vars::Tuple)
        return new{typeof(data),typeof(vars)}(data, vars)
    end
end

function StateVector{T}(dh::DofHandler, nvars) where {T<:Number}
    data = Vector{T}(undef, nvars * ndofs(dh))
    vars = Any[]
    for v in 1:nvars
        push!(
            vars,
            ScalarField(
                view(data, ((v - 1) * ndofs(dh) + 1):(v * ndofs(dh))),
                dh,
            ),
        )
    end
    return StateVector(data, tuple(vars...))
end

function StateVector{T}(dh::AbstractVecOrTuple) where {T<:Number}
    nvars = length(dh)
    len = ndofs.(dh) |> sum
    data = Vector{T}(undef, len)
    cnt = 0
    vars = Any[]
    for v in 1:nvars
        push!(
            vars,
            ScalarField(
                view(data, (cnt + 1):(cnt + ndofs(dh[v]))),
                dh[v],
            ),
        )
        cnt += ndofs(dh[v])
    end
    return StateVector(data, tuple(vars...))
end

function Base.similar(self::StateVector)
    data = similar(self.data)
    return similar(data, self)
end

function Base.similar(data, self::StateVector)
    vars = similar(self.vars)
    return StateVector(data, vars)
end

function Base.show(io::IO, ::MIME"text/plain", self::StateVector{RT}) where {RT}
    @nospecialize
    nvars = nvariables(self)
    nelem = nelements(self)
    print(io, nelem, "-element state vector with ", nvars, " variable(s), ",
        " and eltype ", RT, ".")
    return nothing
end

@inline Base.getindex(self::StateVector, var) = self.vars[var]
@inline Base.length(self::StateVector) = length(self.vars)
@inline Base.size(self::StateVector) = (length(self.vars),)
@inline function Base.iterate(self::StateVector, state=1)
    return state > length(self) ? nothing : (self.vars[state], state + 1)
end

@inline nelements(self::StateVector) = nelements(self.dh)
@inline nvariables(self::StateVector) = length(self)
@inline eachelement(self::StateVector) = Base.OneTo(nelements(self))
@inline eachvariable(self::StateVector) = Base.OneTo(nvariables(self))

#==========================================================================================#

struct BlockVector{T,V}
    data::T
    vars::V
    function BlockVector(data::AbstractVector, vars::Tuple)
        return new{typeof(data),typeof(vars)}(data, vars)
    end
end

function BlockVector{T}(dh::DofHandler{ND}, nvars) where {ND,T<:Number}
    data = [MVector{ND,T}(undef) for _ in 1:(nvars * ndofs(dh))]
    vars = Any[]
    for v in 1:nvars
        push!(
            vars,
            VectorField(
                view(data, ((v - 1) * ndofs(dh) + 1):(v * ndofs(dh))),
                dh,
            ),
        )
    end
    return BlockVector(data, tuple(vars...))
end

function BlockVector{ND,T}(dh::AbstractVecOrTuple) where {ND,T<:Number}
    nvars = length(dh)
    len = ndofs.(dh) |> sum
    data = [MVector{ND,T}(undef) for _ in 1:len]
    cnt = 0
    vars = Any[]
    for v in 1:nvars
        push!(
            vars,
            VectorField(
                view(data, (cnt + 1):(cnt + ndofs(dh[v]))),
                dh[v],
            ),
        )
        cnt += ndofs(dh[v])
    end
    return BlockVector(data, tuple(vars...))
end

function Base.similar(self::BlockVector)
    data = similar(self.data)
    return similar(data, self)
end

function Base.similar(data, self::BlockVector)
    vars = similar(self.vars)
    return BlockVector(data, vars)
end

function Base.show(io::IO, ::MIME"text/plain", self::BlockVector{RT}) where {RT}
    @nospecialize
    nvars = nvariables(self)
    nelem = nelements(self)
    print(io, nelem, "-element block vector with ", nvars, " variable(s), ",
        " and eltype ", RT, ".")
    return nothing
end

@inline Base.getindex(self::BlockVector, var) = self.vars[var]
@inline Base.length(self::BlockVector) = length(self.vars)
@inline Base.size(self::BlockVector) = (length(self.vars),)
@inline function Base.iterate(self::BlockVector, state=1)
    return state > length(self) ? nothing : (self.vars[state], state + 1)
end

@inline nelements(self::BlockVector) = nelements(self.dh)
@inline nvariables(self::BlockVector) = length(self)
@inline eachelement(self::BlockVector) = Base.OneTo(nelements(self))
@inline eachvariable(self::BlockVector) = Base.OneTo(nvariables(self))

#==========================================================================================#
#                                     Face scalar field                                    #

struct FaceScalarField{T,ND}
    data::T
    dh::DofHandler{ND}
end

#==========================================================================================#
#                                     Face state vector                                    #

struct FaceStateVector{T,V}
    data::T
    vars::V
end
# struct MortarStateVector{RT<:Real} <: AbstractVector{RT}
#     data::Vector{Vector{Matrix{RT}}}   # [[ndofs, nvar] × nsides] × nfaces
# end

# function Base.similar(m::MortarStateVector)
#     data = similar(m.data)
#     for i in eachindex(data, m.data)
#         data[i] = [similar(m.data[i][1]), similar(m.data[i][2])]
#     end
#     return MortarStateVector(data)
# end

# function MortarStateVector{RT}(value, dims) where {RT}
#     nfaces = length(dims)
#     data = Vector{Vector{Matrix{RT}}}(undef, nfaces)
#     for i in 1:nfaces
#         data[i] = [Matrix{RT}(value, d...) for d in dims[i]]
#     end
#     return MortarStateVector(data)
# end

# function Base.show(io::IO, ::MIME"text/plain", m::MortarStateVector{RT}) where {RT}
#     @nospecialize
#     nface = length(m.data)
#     nvars = size(s.data[1][1], 2)
#     print(io, nface, " face mortar state vector with ", nvars,
#         " variable(s) and eltype ", RT)
# end
# Base.size(m::MortarStateVector) = size(m.data)
# Base.getindex(m::MortarStateVector, i::Int) = m.data[i]
# Base.fill!(m::MortarStateVector, v) = fill!.(m.data, Ref(v))
