abstract type AbstractQuadrature end

struct GaussQuadrature <: AbstractQuadrature end
struct GaussLobattoQuadrature <: AbstractQuadrature end

const GL = GaussQuadrature
const GLL = GaussLobattoQuadrature

abstract type AbstractStdRegion{ND,Q,Dims} end

Base.size(::AbstractStdRegion{ND,Q,Dims}) where {ND,Q,Dims} = Dims
Base.size(::AbstractStdRegion{ND,Q,Dims}, i) where {ND,Q,Dims} = Dims[i]
Base.length(::AbstractStdRegion{ND,Q,Dims}) where {ND,Q,Dims} = prod(Dims)
Base.eachindex(s::AbstractStdRegion, i) = Base.OneTo(size(s, i))
Base.eachindex(s::AbstractStdRegion) = Base.OneTo(length(s))
Base.LinearIndices(s::AbstractStdRegion) = s.lindices
Base.CartesianIndices(s::AbstractStdRegion) = s.cindices

"""
    is_tensor_product(std)

Return `true` if `std` is a tensor-product standard region, and `false` otherwise.
"""
function is_tensor_product end

function ndirections end

function massmatrix end

function project2equispaced! end

function slave2master end

function master2slave end

spatialdim(::AbstractStdRegion{ND}) where {ND} = ND
ndofs(s::AbstractStdRegion) = length(s)
faces(s::AbstractStdRegion{ND}) where {ND} = s.fstd
face(s::AbstractStdRegion{ND}, i) where {ND} = s.fstd[i]
quadrature(::AbstractStdRegion{ND,Q}) where {ND,Q} = Q
eachdirection(s::AbstractStdRegion) = Base.OneTo(ndirections(s))

#==========================================================================================#
#                                      Standard point                                      #

struct StdPoint{Dims,CI,LI} <: AbstractStdRegion{0,GaussQuadrature,Dims}
    cindices::CI
    lindices::LI
end

function StdPoint()
    dim = (1,)
    ci = CartesianIndices(dim) |> collect
    li = LinearIndices(dim) |> collect
    return StdPoint{dim,typeof(ci),typeof(li)}(ci, li)
end

is_tensor_product(::StdPoint) = true
ndirections(::StdPoint) = 1
nvertices(::StdPoint) = 1

function slave2master(i, _, ::StdPoint)
    return i
end

function master2slave(i, _, ::StdPoint)
    return i
end

#==========================================================================================#
#                                     Standard segment                                     #

struct StdSegment{QT,Dims,RT,MM,CI,LI,FS1,FS2} <: AbstractStdRegion{1,QT,Dims}
    fstd::Tuple{FS1,FS2}
    cindices::CI
    lindices::LI
    ??e::Vector{SVector{1,RT}}
    ??::Vector{SVector{1,RT}}
    ??::Vector{RT}
    M::MM
    D::Tuple{Transpose{RT,Matrix{RT}}}
    Q::Tuple{Transpose{RT,Matrix{RT}}}
    K::Tuple{Transpose{RT,Matrix{RT}}}
    Ks::Tuple{Transpose{RT,Matrix{RT}}}
    K???::Tuple{Transpose{RT,Matrix{RT}}}
    l::NTuple{2,Transpose{RT,Vector{RT}}}   # Row vectors
    l??::NTuple{2,Vector{RT}}
    _n2e::Transpose{RT,Matrix{RT}}
end

is_tensor_product(::StdSegment) = true
ndirections(::StdSegment) = 1
nvertices(::StdSegment) = 2

function StdSegment{RT}(np::Integer, qtype::AbstractQuadrature) where {RT<:Real}
    fstd = (StdPoint(), StdPoint())
    _??, ?? = if qtype isa(GaussQuadrature)
        gausslegendre(np)
    elseif qtype isa(GaussLobattoQuadrature)
        gausslobatto(np)
    else
        throw(ArgumentError("Only Gauss and Gauss-Lobatto quadratures are implemented."))
    end
    _??, ?? = convert.(RT, _??), convert.(RT, ??)
    ?? = [SVector(??i) for ??i in _??]

    # Equispaced nodes
    _??e = convert.(RT, range(-1, 1, np))
    ??e = [SVector(??) for ?? in _??e]

    cindices = CartesianIndices((np,)) |> collect
    lindices = LinearIndices((np,)) |> collect

    # Mass matrix
    M = Diagonal(??)

    # Lagrange basis
    D = Matrix{RT}(undef, np, np)
    _n2e = similar(D)
    l = (Vector{RT}(undef, np), Vector{RT}(undef, np))
    y = fill(zero(RT), np)
    for i in 1:np
        y[i] = one(RT)
        Li = convert(Polynomial, Lagrange(_??, y))
        ???Li = derivative(Li)
        D[:, i] .= ???Li.(_??)
        l[1][i] = Li(-one(RT))
        l[2][i] = Li(+one(RT))
        _n2e[:, i] .= Li.(_??e)
        y[i] = zero(RT)
    end

    # SBP matrices
    B = l[2] * l[2]' - l[1] * l[1]'
    Q = M * D

    # Volume operators
    K = Q' |> Matrix{RT}
    Ks = Q |> copy
    K??? = 2Q

    # Surface contribution
    l?? = l
    l = l .|> transpose |> Tuple
    Ks = -Ks + B
    K??? = -K??? + B

    return StdSegment{
        typeof(qtype),
        Tuple(np),
        eltype(??),
        typeof(M),
        typeof(cindices),
        typeof(lindices),
        typeof(fstd[1]),
        typeof(fstd[2]),
    }(
        fstd,
        cindices,
        lindices,
        ??e,
        ??,
        ??,
        M,
        D |> transpose |> collect |> transpose |> tuple,
        Q |> transpose |> collect |> transpose |> tuple,
        K |> transpose |> collect |> transpose |> tuple,
        Ks |> transpose |> collect |> transpose |> tuple,
        K??? |> transpose |> collect |> transpose |> tuple,
        l,
        l??,
        _n2e |> transpose |> collect |> transpose,
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::StdSegment{QT,D,RT}) where {QT,D,RT}
    @nospecialize
    print(io, "StdSegment{", RT, "}: ")
    if QT == GaussQuadrature
        print(io,  "Gauss quadrature with ", D[1], " nodes")
    elseif QT == GaussLobattoQuadrature
        print(io, "Gauss-Lobatto quadrature with ", D[1], " nodes")
    else
        @assert false "[StdRegion.show] You shouldn't be here..."
    end
    return nothing
end

function massmatrix(std::StdSegment, J)
    return Diagonal(J) * std.M
end

@inline function project2equispaced!(Qe, Q, s::StdSegment)
    @boundscheck length(Qe) == length(Q) || throw(DimensionMismatch(
        "`Qe` and `Q` must have the same length (== ndofs)."
    ))
    @inbounds Qe .= s._n2e * Q
    return nothing
end

function slave2master(i, orientation, std::StdSegment)
    return if orientation == 1
        i
    else # orientation == 2
        length(std) - i
    end
end

function master2slave(i, orientation, std::StdSegment)
    return if orientation == 1
        i
    else # orientation == 2
        length(std) - i
    end
end

_VTK_type(::StdSegment) = UInt8(68)

function _VTK_connectivities(s::StdSegment)
    conns = [1, length(s)]
    append!(conns, 2:(length(s) - 1))
    return conns .- 1
end

#==========================================================================================#
#                                       Standard quad                                      #

struct StdQuad{QT,Dims,RT,MM,CI,LI,FS1,FS2} <: AbstractStdRegion{2,QT,Dims}
    fstd::Tuple{FS1,FS2}
    cindices::CI
    lindices::LI
    ??e::Vector{SVector{2,RT}}
    ??::Vector{SVector{2,RT}}
    ??::Vector{RT}
    M::MM
    D::NTuple{2,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    Q::NTuple{2,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    K::NTuple{2,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    Ks::NTuple{2,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    K???::NTuple{2,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    l::NTuple{4,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    l??::NTuple{4,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    _n2e::NTuple{2,Transpose{RT,SparseMatrixCSC{RT,Int}}}
end

is_tensor_product(::StdQuad) = true
ndirections(::StdQuad) = 2
nvertices(::StdQuad) = 4
faces(s::StdQuad) = (s.fstd[1], s.fstd[1], s.fstd[2], s.fstd[2])
face(s::StdQuad, i) = s.fstd[(i - 1) ?? 2 + 1]

function StdQuad{RT}(np::AbstractVecOrTuple, qtype::AbstractQuadrature) where {RT<:Real}
    # Quadratures
    fstd = (
        StdSegment{RT}(np[2], qtype),   # Note the swap!!
        StdSegment{RT}(np[1], qtype),
    )
    ?? = vec([SVector(??x[1], ??y[1]) for ??x in fstd[2].??, ??y in fstd[1].??])
    ?? = vec([??x * ??y for ??x in fstd[2].??, ??y in fstd[1].??])

    # Equispaced nodes
    ??e = vec([SVector(??x[1], ??y[1]) for ??x in fstd[2].??e, ??y in fstd[1].??e])

    cindices = CartesianIndices((np...,)) |> collect
    lindices = LinearIndices((np...,)) |> collect

    # Mass matrix
    M = Diagonal(??)

    # Derivative matrices
    I = (Diagonal(ones(np[1])), Diagonal(ones(np[2])))
    I?? = (Diagonal(fstd[2].??), Diagonal(fstd[1].??))
    D = (
        kron(I[2], fstd[2].D[1]),
        kron(fstd[1].D[1], I[1]),
    )
    Q = (
        kron(I[2], fstd[2].Q[1]),
        kron(fstd[1].Q[1], I[2]),
    )
    _n2e = (
        kron(I[2], fstd[2]._n2e[1]),
        kron(fstd[1]._n2e[1], I[1]),
    )
    K = (
        kron(I??[2], fstd[2].K[1]),
        kron(fstd[1].K[1], I??[1]),
    )
    Ks = (
        kron(I??[2], fstd[2].Ks[1]),
        kron(fstd[1].Ks[1], I??[1]),
    )
    K??? = (
        kron(diag(I??[2]), fstd[2].K???[1]),
        kron(fstd[1].K???[1], diag(I??[1])),
    )

    # Projection operator
    l = (
        kron(I[2], fstd[2].l[1]),
        kron(I[2], fstd[2].l[2]),
        kron(fstd[1].l[1], I[1]),
        kron(fstd[1].l[2], I[1]),
    )

    # Surface contribution
    l?? = (
        kron(I??[2], fstd[2].l??[1]),
        kron(I??[2], fstd[2].l??[2]),
        kron(fstd[1].l??[1], I??[1]),
        kron(fstd[1].l??[2], I??[1]),
    )

    return StdQuad{
        typeof(qtype),
        Tuple(np),
        eltype(??),
        typeof(M),
        typeof(cindices),
        typeof(lindices),
        typeof(fstd[1]),
        typeof(fstd[2]),
    }(
        fstd,
        cindices,
        lindices,
        ??e,
        ??,
        ??,
        M,
        D .|> transpose .|> sparse .|> transpose |> Tuple,
        Q .|> transpose .|> sparse .|> transpose |> Tuple,
        K .|> transpose .|> sparse .|> transpose |> Tuple,
        Ks .|> transpose .|> sparse .|> transpose |> Tuple,
        K??? .|> transpose .|> collect .|> transpose |> Tuple,
        l .|> transpose .|> sparse .|> transpose |> Tuple,
        l?? .|> transpose .|> sparse .|> transpose |> Tuple,
        _n2e .|> transpose .|> sparse .|> transpose |> Tuple,
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::StdQuad{QT,D,RT}) where {QT,D,RT}
    @nospecialize
    println(io, "StdRegion{", RT, "}: ")
    if QT == GaussQuadrature
        println(io, " ??: Gauss quadrature with ", D[1], " nodes")
        print(io, " ??: Gauss quadrature with ", D[2], " nodes")
    elseif QT == GaussLobattoQuadrature
        println(io, " ??: Gauss-Lobatto quadrature with ", D[1], " nodes")
        print(io, " ??: Gauss-Lobatto quadrature with ", D[2], " nodes")
    else
        @assert false "[StdRegion.show] You shouldn't be here..."
    end
    return nothing
end

function massmatrix(std::StdQuad, J)
    return Diagonal(J) * std.M
end

@inline function project2equispaced!(Qe, Q, s::StdQuad)
    @boundscheck length(Qe) == length(Q) == ndofs(s) || throw(DimensionMismatch(
        "`Qe` and `Q` must have the same length (== ndofs)."
    ))
    @inbounds begin
        Qr = reshape(Q, size(s))
        Qer = reshape(Qe, size(s))
        Qer .= s._n2e[1] * Qr * s._n2e[2]'
    end
    return nothing
end

function slave2master(i, orientation, std::StdQuad)
    error("Not implemented yet!")
end

function master2slave(i, orientation, shape::Vararg{<:Integer,1})
    error("Not implemented yet!")
end

_VTK_type(::StdQuad) = UInt8(70)

function _VTK_connectivities(s::StdQuad)
    nx, ny = size(s)
    li = LinearIndices(s)
    conns = [
        li[1, 1], li[nx, 1], li[nx, ny], li[1, ny],
        li[2:(end - 1), 1]..., li[nx, 2:(end - 1)]...,
        li[2:(end - 1), ny]..., li[1, 2:(end - 1)]...,
        li[2:(end - 1), 2:(end - 1)]...,
    ]
    return conns .- 1
end

#==========================================================================================#
#                                     Standard triangle                                    #

struct StdTri{Q,Dims} <: AbstractStdRegion{2,Q,Dims} end

is_tensor_product(::StdTri) = false
ndirections(::StdTri) = 3
nvertices(::StdTri) = 3

function slave2master(i, orientation, std::StdTri)
    error("Not implemented yet!")
end

function master2slave(i, orientation, std::StdTri)
    error("Not implemented yet!")
end

_VTK_type(::StdTri) = UInt8(69)

function _VTK_connectivities(::StdTri)
    error("Not implemented yet!")
end
