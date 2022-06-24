abstract type AbstractQuadrature end

struct GaussQuadrature <: AbstractQuadrature end
struct GaussLobattoQuadrature <: AbstractQuadrature end

const GL = GaussQuadrature
const GLL = GaussLobattoQuadrature

abstract type AbstractStdRegion{ND,Dims} end

Base.size(::AbstractStdRegion{ND,Dims}) where {ND,Dims} = Dims
Base.size(::AbstractStdRegion{ND,Dims}, i) where {ND,Dims} = Dims[i]
Base.length(::AbstractStdRegion{ND,Dims}) where {ND,Dims} = prod(Dims)
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
faces(s::AbstractStdRegion{ND}) where {ND} = 1 <= ND <= 3 ? s.fstd : nothing
face(s::AbstractStdRegion{ND}, i) where {ND} = 1 <= ND <= 3 ? s.fstd[i] : nothing
quadratures(s::AbstractStdRegion) = s.quad
quadrature(s::AbstractStdRegion, i) = s.quad[i]
eachdirection(s::AbstractStdRegion) = Base.OneTo(ndirections(s))

#==========================================================================================#
#                                      Standard point                                      #

struct StdPoint{Dims,CI,LI} <: AbstractStdRegion{0,Dims}
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

struct StdSegment{Dims,QT,RT,MM,CI,LI,FS1,FS2} <: AbstractStdRegion{1,Dims}
    quad::Tuple{QT}
    fstd::Tuple{FS1,FS2}
    cindices::CI
    lindices::LI
    ξe::Vector{SVector{1,RT}}
    ξ::Vector{SVector{1,RT}}
    ω::Vector{RT}
    M::MM
    D::Tuple{Matrix{RT}}
    Q::Tuple{Matrix{RT}}
    K::Tuple{Matrix{RT}}
    Ks::Tuple{Matrix{RT}}
    K♯::Tuple{Matrix{RT}}
    l::NTuple{2,Vector{RT}}
    lω::NTuple{2,Vector{RT}}
    _n2e::Matrix{RT}
end

is_tensor_product(::StdSegment) = true
ndirections(::StdSegment) = 1
nvertices(::StdSegment) = 2

function StdSegment{RT}(np, qtype) where {RT<:Real}
    fstd = (StdPoint(), StdPoint())
    _ξ, ω = if qtype isa(GaussQuadrature)
        gausslegendre(np)
    elseif qtype isa(GaussLobattoQuadrature)
        gausslobatto(np)
    else
        throw(ArgumentError("Only Gauss and Gauss-Lobatto quadratures are implemented."))
    end
    _ξ, ω = convert.(RT, _ξ), convert.(RT, ω)
    ξ = [SVector(ξi) for ξi in _ξ]

    # Equispaced nodes
    _ξe = convert.(RT, range(-1, 1, np))
    ξe = [SVector(ξ) for ξ in _ξe]

    cindices = CartesianIndices((np,)) |> collect
    lindices = LinearIndices((np,)) |> collect

    # Mass matrix
    M = Diagonal(ω)

    # Lagrange basis
    D = Matrix{RT}(undef, np, np)
    _n2e = similar(D)
    l = (Vector{RT}(undef, np), Vector{RT}(undef, np))
    y = fill(zero(RT), np)
    for i in 1:np
        y[i] = one(RT)
        Li = convert(Polynomial, Lagrange(_ξ, y))
        ∂Li = derivative(Li)
        D[:, i] .= ∂Li.(_ξ)
        l[1][i] = Li(-one(RT))
        l[2][i] = Li(+one(RT))
        _n2e[:, i] .= Li.(_ξe)
        y[i] = zero(RT)
    end

    # SBP matrices
    B = l[2] * l[2]' - l[1] * l[1]'
    Q = M * D

    # Volume operators
    K = Q' |> Matrix{RT}
    Ks = copy(Q)
    K♯ = 2Q

    # Surface contribution
    lω = l
    @. Ks = -Ks + B
    @. K♯ = -K♯ + B

    return StdSegment{
        Tuple(np),
        typeof(qtype),
        eltype(ω),
        typeof(M),
        typeof(cindices),
        typeof(lindices),
        typeof(fstd[1]),
        typeof(fstd[2]),
    }(
        (qtype,),
        fstd,
        cindices,
        lindices,
        ξe,
        ξ,
        ω,
        M,
        (D,),
        (Q,),
        (K,),
        (Ks,),
        (K♯,),
        l,
        lω,
        _n2e,
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::StdSegment{D,QT,RT}) where {D,QT,RT}
    @nospecialize
    print(io, "StdSegment{", RT, "}: ")
    if QT == GaussQuadrature
        print(io,  "Gauss quadrature with ", size(s, 1), " nodes")
    elseif QT == GaussLobattoQuadrature
        print(io, "Gauss-Lobatto quadrature with ", size(s, 1), " nodes")
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

struct StdQuad{Dims,QT1,QT2,RT,MM,CI,LI,FS1,FS2} <: AbstractStdRegion{2,Dims}
    quad::Tuple{QT1,QT2}
    fstd::Tuple{FS1,FS2}
    cindices::CI
    lindices::LI
    ξe::Vector{SVector{2,RT}}
    ξ::Vector{SVector{2,RT}}
    ω::Vector{RT}
    M::MM
    D::NTuple{2,Matrix{RT}}
    Q::NTuple{2,Matrix{RT}}
    K::NTuple{2,Array{RT,3}}
    Ks::NTuple{2,Array{RT,3}}
    K♯::NTuple{2,Array{RT,3}}
    l::NTuple{4,Vector{RT}}
    lω::NTuple{4,Matrix{RT}}
    _n2e::NTuple{2,Matrix{RT}}
end

is_tensor_product(::StdQuad) = true
ndirections(::StdQuad) = 2
nvertices(::StdQuad) = 4
faces(s::StdQuad) = (s.fstd[1], s.fstd[1], s.fstd[2], s.fstd[2])
face(s::StdQuad, i) = s.fstd[(i - 1) ÷ 2 + 1]

function StdQuad{RT}(np, qtype) where {RT<:Real}
    # Quadratures
    fstd = (
        StdSegment{RT}(np[1], qtype[1]),
        StdSegment{RT}(np[2], qtype[2]),
    )
    ξ = vec([SVector(ξx[1], ξy[1]) for ξx in fstd[1].ξ, ξy in fstd[2].ξ])
    ω = vec([ωx * ωy for ωx in fstd[1].ω, ωy in fstd[2].ω])

    # Equispaced nodes
    ξe = vec([SVector(ξx[1], ξy[1]) for ξx in fstd[1].ξe, ξy in fstd[2].ξe])

    cindices = CartesianIndices((np...,)) |> collect
    lindices = LinearIndices((np...,)) |> collect

    # Mass matrix
    M = Diagonal(ω)

    # Derivative matrices
    D = (fstd[1].D[1], fstd[2].D[1])
    Q = (fstd[1].Q[1], fstd[2].Q[1])
    _n2e = (fstd[1]._n2e, fstd[2]._n2e)
    K = (
        Array{RT,3}(undef, np[1], np[2], np[1]),
        Array{RT,3}(undef, np[1], np[2], np[2]),
    )
    Ks = (
        Array{RT,3}(undef, np[1], np[2], np[1]),
        Array{RT,3}(undef, np[1], np[2], np[2]),
    )
    K♯ = (
        Array{RT,3}(undef, np[1], np[2], np[1]),
        Array{RT,3}(undef, np[1], np[2], np[2]),
    )
    for j in 1:np[2]
        @. K[1][:, j, :] = fstd[1].K[1] * fstd[2].ω[j]
        @. Ks[1][:, j, :] = fstd[1].Ks[1] * fstd[2].ω[j]
        @. K♯[1][:, j, :] = fstd[1].K♯[1] * fstd[2].ω[j]
    end
    for i in 1:np[1]
        @. K[2][i, :, :] = fstd[2].K[1] * fstd[1].ω[i]
        @. Ks[2][i, :, :] = fstd[2].Ks[1] * fstd[1].ω[i]
        @. K♯[2][i, :, :] = fstd[2].K♯[1] * fstd[1].ω[i]
    end

    # Projection operator
    l = (
        fstd[1].l[1],
        fstd[1].l[2],
        fstd[2].l[1],
        fstd[2].l[2],
    )

    # Surface contribution
    lω = (
        fstd[1].l[1] * fstd[2].ω',
        fstd[1].l[2] * fstd[2].ω',
        fstd[2].l[1] * fstd[1].ω',
        fstd[2].l[2] * fstd[1].ω',
    )

    return StdQuad{
        Tuple(np),
        typeof(qtype[1]),
        typeof(qtype[2]),
        eltype(ω),
        typeof(M),
        typeof(cindices),
        typeof(lindices),
        typeof(fstd[1]),
        typeof(fstd[2]),
    }(
        Tuple(qtype),
        fstd,
        cindices,
        lindices,
        ξe,
        ξ,
        ω,
        M,
        D,
        Q,
        K,
        Ks,
        K♯,
        l,
        lω,
        _n2e,
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::StdQuad{D,Q1,Q2,RT}) where {D,Q1,Q2,RT}
    @nospecialize
    println(io, "StdRegion{", RT, "}: ")
    if Q1 == GaussQuadrature
        println(io, " ξ: Gauss quadrature with ", size(s, 1), " nodes")
    elseif Q1 == GaussLobattoQuadrature
        println(io, " ξ: Gauss-Lobatto quadrature with ", size(s, 1), " nodes")
    else
        @assert false "[StdRegion.show] You shouldn't be here..."
    end
    if Q2 == GaussQuadrature
        print(io, " η: Gauss quadrature with ", size(s, 2), " nodes")
    elseif Q2 == GaussLobattoQuadrature
        print(io, " η: Gauss-Lobatto quadrature with ", size(s, 2), " nodes")
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

struct StdTri{Dims} <: AbstractStdRegion{2,Dims} end

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

    # NOTE: Non tensor-product approach for a quad element
    # D = (zeros(RT, npts, npts), zeros(RT, npts, npts))
    # K = (zeros(RT, npts, npts), zeros(RT, npts, npts))
    # Ds = (zeros(RT, npts, npts), zeros(RT, npts, npts))
    # for j in 1:np[2], i in 1:np[1]
    #     k = lindices[i, j]
    #     D[1][k, lindices[:, j]] = fstd[1].D[i, :]
    #     D[2][k, lindices[i, :]] = fstd[2].D[j, :]
    #     K[1][k, lindices[:, j]] = fstd[1].D[:, i] .* ω[lindices[:, j]]
    #     K[2][k, lindices[i, :]] = fstd[2].D[:, j] .* ω[lindices[i, :]]
    #     Ds[1][k, lindices[:, j]] = fstd[1].Ds[i, :]
    #     Ds[2][k, lindices[i, :]] = fstd[2].Ds[j, :]
    # end
    # D = sparse.(D)
    # K = sparse.(K)
    # Ds = sparse.(Ds)
    # l = (
    #     zeros(RT, np[2], npts),
    #     zeros(RT, np[2], npts),
    #     zeros(RT, np[1], npts),
    #     zeros(RT, np[1], npts),
    # )
    # for j in 1:np[2]
    #     l[1][j, lindices[:, j]] = fstd[1].l[1]
    #     l[2][j, lindices[:, j]] = fstd[1].l[2]
    # end
    # for i in 1:np[1]
    #     l[3][i, lindices[i, :]] = fstd[2].l[1]
    #     l[4][i, lindices[i, :]] = fstd[2].l[2]
    # end
    # l = sparse.(l)
    # lω = (
    #     zeros(RT, npts, np[2]),
    #     zeros(RT, npts, np[2]),
    #     zeros(RT, npts, np[1]),
    #     zeros(RT, npts, np[1]),
    # )
    # for j in 1:np[2], i in 1:np[1]
    #     k = lindices[i, j]
    #     lω[1][k, j] = fstd[1].l[1][i] * fstd[2].ω[j]
    #     lω[2][k, j] = fstd[1].l[2][i] * fstd[2].ω[j]
    #     lω[3][k, i] = fstd[2].l[1][j] * fstd[1].ω[i]
    #     lω[4][k, i] = fstd[2].l[2][j] * fstd[1].ω[i]
    # end
    # lω = sparse.(lω)
