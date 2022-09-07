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
Base.LinearIndices(s::AbstractStdRegion, _=false) = s.lindices
Base.CartesianIndices(s::AbstractStdRegion, _=false) = s.cindices

"""
    is_tensor_product(std)

Return `true` if `std` is a tensor-product standard region, and `false` otherwise.
"""
function is_tensor_product end

function ndirections end

function massmatrix end

@inline function project2equispaced!(Qe, Q, s::AbstractStdRegion)
    @boundscheck length(Qe) == length(Q) || throw(DimensionMismatch(
        "`Qe` and `Q` must have the same length (== ndofs)."
    ))
    @inbounds Qe .= s._n2e * Q
    return nothing
end

function slave2master end

function master2slave end

get_spatialdim(::AbstractStdRegion{ND}) where {ND} = ND
ndofs(s::AbstractStdRegion) = length(s)
get_faces(s::AbstractStdRegion{ND}) where {ND} = s.fstd
get_face(s::AbstractStdRegion{ND}, i) where {ND} = s.fstd[i]
get_quadrature(::AbstractStdRegion{ND,Q}) where {ND,Q} = Q
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
    ξe::Vector{SVector{1,RT}}
    ξc::Tuple{Vector{SVector{1,RT}}}
    ξ::Vector{SVector{1,RT}}
    ω::Vector{RT}
    M::MM
    D::Tuple{Transpose{RT,Matrix{RT}}}
    Q::Tuple{Transpose{RT,Matrix{RT}}}
    K::Tuple{Transpose{RT,Matrix{RT}}}
    Ks::Tuple{Transpose{RT,Matrix{RT}}}
    K♯::Tuple{Transpose{RT,Matrix{RT}}}
    l::NTuple{2,Transpose{RT,Vector{RT}}}   # Row vectors
    lω::NTuple{2,Vector{RT}}
    _n2e::Transpose{RT,Matrix{RT}}
end

is_tensor_product(::StdSegment) = true
ndirections(::StdSegment) = 1
nvertices(::StdSegment) = 2

function StdSegment{RT}(np::Integer, qtype::AbstractQuadrature, npe=np) where {RT<:Real}
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
    _ξe = convert.(RT, range(-1, 1, npe))
    ξe = [SVector(ξ) for ξ in _ξe]

    # Complementary nodes
    ξc1 = Vector{SVector{1,RT}}(undef, np + 1)
    ξc1[1] = SVector(-one(RT))
    for i in 1:np
        ξc1[i + 1] = ξc1[i] .+ ω[i]
    end
    ξc2 = Vector{SVector{1,RT}}(undef, np + 1)
    ξc2[np + 1] = SVector(one(RT))
    for i in np:-1:1
        ξc2[i] = ξc2[i + 1] .- ω[i]
    end
    ξc = (ξc1 .+ ξc2) ./ 2
    ξc[1] = SVector(-one(RT))
    ξc[np + 1] = SVector(one(RT))
    ξc = tuple(ξc)

    cindices = CartesianIndices((np,)) |> collect
    lindices = LinearIndices((np,)) |> collect

    # Mass matrix
    M = Diagonal(ω)

    # Lagrange basis
    D = Matrix{RT}(undef, np, np)
    _n2e = Matrix{RT}(undef, npe, np)
    l = (Vector{RT}(undef, np), Vector{RT}(undef, np))
    y = fill(zero(RT), np)
    for i in 1:np
        y[i] = one(RT)
        Li = convert(Polynomial, Lagrange(_ξ, y))
        dLi = derivative(Li)
        D[:, i] .= dLi.(_ξ)
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
    Ks = Q |> copy
    K♯ = 2Q

    # Surface contribution
    lω = l
    l = l .|> transpose |> Tuple
    Ks = -Ks + B
    K♯ = -K♯ + B

    return StdSegment{
        typeof(qtype),
        Tuple(np),
        eltype(ω),
        typeof(M),
        typeof(cindices),
        typeof(lindices),
        typeof(fstd[1]),
        typeof(fstd[2]),
    }(
        fstd,
        cindices,
        lindices,
        ξe,
        ξc,
        ξ,
        ω,
        M,
        D |> transpose |> collect |> transpose |> tuple,
        Q |> transpose |> collect |> transpose |> tuple,
        K |> transpose |> collect |> transpose |> tuple,
        Ks |> transpose |> collect |> transpose |> tuple,
        K♯ |> transpose |> collect |> transpose |> tuple,
        l,
        lω,
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

function slave2master(i, orientation, std::StdSegment)
    return if orientation == 0
        i
    else # orientation == 1
        length(std) - (i - 1)
    end
end

function master2slave(i, orientation, std::StdSegment)
    return if orientation == 0
        i
    else # orientation == 1
        length(std) - (i - 1)
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
    cindices_t::CI
    lindices_t::LI
    ξe::Vector{SVector{2,RT}}
    ξc::NTuple{2,Matrix{SVector{2,RT}}}
    ξ::Vector{SVector{2,RT}}
    ω::Vector{RT}
    M::MM
    D::NTuple{2,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    Q::NTuple{2,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    K::NTuple{2,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    Ks::NTuple{2,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    K♯::NTuple{2,Transpose{RT,Matrix{RT}}}
    l::NTuple{4,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    lω::NTuple{4,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    _n2e::Transpose{RT,Matrix{RT}}
end

Base.LinearIndices(s::StdQuad, transpose=false) = transpose ? s.lindices_t : s.lindices
Base.CartesianIndices(s::StdQuad, transpose=false) = transpose ? s.cindices_t : s.cindices

is_tensor_product(::StdQuad) = true
ndirections(::StdQuad) = 2
nvertices(::StdQuad) = 4
get_faces(s::StdQuad) = (s.fstd[1], s.fstd[1], s.fstd[2], s.fstd[2])
get_face(s::StdQuad, i) = s.fstd[(i - 1) ÷ 2 + 1]

function StdQuad{RT}(
    np::AbstractVecOrTuple,
    qtype::AbstractQuadrature,
    npe=nothing,
) where {
    RT<:Real,
}
    # Quadratures
    if isnothing(npe)
        npe = maximum(np)
    end
    fstd = (
        StdSegment{RT}(np[2], qtype, npe),
        StdSegment{RT}(np[1], qtype, npe),
    )
    ξ = vec([SVector(ξx[1], ξy[1]) for ξx in fstd[2].ξ, ξy in fstd[1].ξ])
    ω = vec([ωx * ωy for ωx in fstd[2].ω, ωy in fstd[1].ω])

    # Equispaced nodes
    ξe = vec([SVector(ξx[1], ξy[1]) for ξx in fstd[2].ξe, ξy in fstd[1].ξe])
    _n2e = kron(fstd[1]._n2e, fstd[2]._n2e)

    # Complementary nodes
    ξc1 = [SVector(ξx[1], ξy[1]) for ξx in fstd[2].ξc[1], ξy in fstd[1].ξ]
    ξc2 = [SVector(ξx[1], ξy[1]) for ξx in fstd[2].ξ, ξy in fstd[1].ξc[1]]
    ξc = (ξc1, ξc2)

    cindices = CartesianIndices((np[1], np[2])) |> collect
    lindices = LinearIndices((np[1], np[2])) |> collect
    cindices_t = CartesianIndices((np[2], np[1])) |> collect
    lindices_t = LinearIndices((np[2], np[1])) |> collect

    # Mass matrix
    M = Diagonal(ω)

    # Derivative matrices
    I = (Diagonal(ones(np[1])), Diagonal(ones(np[2])))
    Iω = (Diagonal(fstd[2].ω), Diagonal(fstd[1].ω))
    D = (
        kron(I[2], fstd[2].D[1]),
        kron(fstd[1].D[1], I[1]),
    )
    Q = (
        kron(I[2], fstd[2].Q[1]),
        kron(fstd[1].Q[1], I[2]),
    )
    K = (
        kron(Iω[2], fstd[2].K[1]),
        kron(fstd[1].K[1], Iω[1]),
    )
    Ks = (
        kron(Iω[2], fstd[2].Ks[1]),
        kron(fstd[1].Ks[1], Iω[1]),
    )
    K♯ = (
        kron(diag(Iω[2]), fstd[2].K♯[1]),
        kron(fstd[1].K♯[1], diag(Iω[1])),
    )

    # Projection operator
    l = (
        kron(I[2], fstd[2].l[1]),
        kron(I[2], fstd[2].l[2]),
        kron(fstd[1].l[1], I[1]),
        kron(fstd[1].l[2], I[1]),
    )

    # Surface contribution
    lω = (
        kron(Iω[2], fstd[2].lω[1]),
        kron(Iω[2], fstd[2].lω[2]),
        kron(fstd[1].lω[1], Iω[1]),
        kron(fstd[1].lω[2], Iω[1]),
    )

    return StdQuad{
        typeof(qtype),
        Tuple(np),
        eltype(ω),
        typeof(M),
        typeof(cindices),
        typeof(lindices),
        typeof(fstd[1]),
        typeof(fstd[2]),
    }(
        fstd,
        cindices,
        lindices,
        cindices_t,
        lindices_t,
        ξe,
        ξc,
        ξ,
        ω,
        M,
        D .|> transpose .|> sparse .|> transpose |> Tuple,
        Q .|> transpose .|> sparse .|> transpose |> Tuple,
        K .|> transpose .|> sparse .|> transpose |> Tuple,
        Ks .|> transpose .|> sparse .|> transpose |> Tuple,
        K♯ .|> transpose .|> collect .|> transpose |> Tuple,
        l .|> transpose .|> sparse .|> transpose |> Tuple,
        lω .|> transpose .|> sparse .|> transpose |> Tuple,
        _n2e |> transpose |> collect |> transpose,
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::StdQuad{QT,D,RT}) where {QT,D,RT}
    @nospecialize
    println(io, "StdQuad{", RT, "}: ")
    if QT == GaussQuadrature
        println(io, " ξ: Gauss quadrature with ", D[1], " nodes")
        print(io, " η: Gauss quadrature with ", D[2], " nodes")
    elseif QT == GaussLobattoQuadrature
        println(io, " ξ: Gauss-Lobatto quadrature with ", D[1], " nodes")
        print(io, " η: Gauss-Lobatto quadrature with ", D[2], " nodes")
    else
        @assert false "[StdRegion.show] You shouldn't be here..."
    end
    return nothing
end

function massmatrix(std::StdQuad, J)
    return Diagonal(J) * std.M
end

function slave2master(i, orientation, std::StdQuad)
    s = CartesianIndices(std)[i]
    st = CartesianIndices(std, true)[i]
    li = LinearIndices(std)
    return if orientation == 0
        i
    elseif orientation == 1
        li[size(std, 1) - st[2] + 1, st[1]]
    elseif orientation == 2
        li[size(std, 1) - s[1] + 1, size(std, 2) - s[2] + 1]
    elseif orientation == 3
        li[st[2], size(std, 2) - st[1] + 1]
    elseif orientation == 4
        li[st[2], st[1]]
    elseif orientation == 5
        li[size(std, 1) - s[1] + 1, s[2]]
    elseif orientation == 6
        li[size(std, 1) - st[2] + 1, size(std, 2) - st[1] + 1]
    else # orientation == 7
        li[s[1], size(std, 2) - s[2] + 1]
    end
end

function master2slave(i, orientation, std::StdQuad)
    m = CartesianIndices(std)[i]
    li = LinearIndices(std)
    lit = LinearIndices(std, true)
    return if orientation == 0
        i
    elseif orientation == 1
        lit[m[2], size(std, 1) - m[1] + 1]
    elseif orientation == 2
        li[size(std, 1) - m[1] + 1, size(std, 2) - m[2] + 1]
    elseif orientation == 3
        lit[size(std, 2) - m[2] + 1, m[1]]
    elseif orientation == 4
        lit[m[2], m[1]]
    elseif orientation == 5
        li[size(std, 1) - m[1] + 1, m[2]]
    elseif orientation == 6
        lit[size(std, 2) - m[2] + 1, size(std, 1) - m[1] + 1]
    else # orientation == 7
        li[m[1], size(std, 2) - m[2] + 1]
    end
end

_VTK_type(::StdQuad) = UInt8(70)

function _VTK_connectivities(s::StdQuad)
    nx, ny = size(s)
    li = LinearIndices(s)
    corners = [li[1, 1], li[nx, 1], li[nx, ny], li[1, ny]]
    edges = reduce(vcat, [
        li[2:(end - 1), 1], li[nx, 2:(end - 1)],
        li[2:(end - 1), ny], li[1, 2:(end - 1)],
    ])
    interior = vec(li[2:(end - 1), 2:(end - 1)])
    return mapreduce(x -> x .- 1, vcat, (corners, edges, interior))
end

#==========================================================================================#
#                                     Standard triangle                                    #

struct StdTri{Q,Dims} <: AbstractStdRegion{2,Q,Dims} end

is_tensor_product(::StdTri) = false
ndirections(::StdTri) = 3
nvertices(::StdTri) = 3

function slave2master(_, _, _::StdTri)
    error("Not implemented yet!")
end

function master2slave(_, _, _::StdTri)
    error("Not implemented yet!")
end

_VTK_type(::StdTri) = UInt8(69)

function _VTK_connectivities(::StdTri)
    error("Not implemented yet!")
end

#==========================================================================================#
#                                       Standard hex                                       #

struct StdHex{QT,Dims,RT,MM,CI,LI,FS1,FS2,FS3} <: AbstractStdRegion{3,QT,Dims}
    fstd::Tuple{FS1,FS2,FS3}
    cindices::CI
    lindices::LI
    ξe::Vector{SVector{3,RT}}
    ξc::NTuple{3,Array{SVector{3,RT},3}}
    ξ::Vector{SVector{3,RT}}
    ω::Vector{RT}
    M::MM
    D::NTuple{3,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    Q::NTuple{3,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    K::NTuple{3,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    Ks::NTuple{3,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    K♯::NTuple{3,Transpose{RT,Matrix{RT}}}
    l::NTuple{6,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    lω::NTuple{6,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    _n2e::Transpose{RT,Matrix{RT}}
end

is_tensor_product(::StdHex) = true
ndirections(::StdHex) = 3
nvertices(::StdHex) = 8
get_faces(s::StdHex) = (s.fstd[1], s.fstd[1], s.fstd[2], s.fstd[2], s.fstd[3], s.fstd[3])
get_face(s::StdHex, i) = s.fstd[(i - 1) ÷ 2 + 1]

function StdHex{RT}(
    np::AbstractVecOrTuple,
    qtype::AbstractQuadrature,
    npe=nothing,
) where {
    RT<:Real,
}
    # Quadratures
    if isnothing(npe)
        npe = maximum(np)
    end
    fstd = (
        StdQuad{RT}((np[2], np[3]), qtype, npe),
        StdQuad{RT}((np[1], np[3]), qtype, npe),
        StdQuad{RT}((np[1], np[2]), qtype, npe),
    )
    lstd = (get_face(fstd[3], 3), get_face(fstd[3], 1), get_face(fstd[1], 1))
    ξ = vec([
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in lstd[1].ξ, ξy in lstd[2].ξ, ξz in lstd[3].ξ
    ])
    ω = vec([ωx * ωy * ωz for ωx in lstd[1].ω, ωy in lstd[2].ω, ωz in lstd[3].ω])

    # Equispaced nodes
    ξe = vec([
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in lstd[1].ξe, ξy in lstd[2].ξe, ξz in lstd[3].ξe
    ])
    _n2e = kron(lstd[3]._n2e, lstd[2]._n2e, lstd[1]._n2e)

    # Complementary nodes
    ξc1 = [
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in lstd[1].ξc[1], ξy in lstd[2].ξ, ξz in lstd[3].ξ
    ]
    ξc2 = [
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in lstd[1].ξ, ξy in lstd[2].ξc[1], ξz in lstd[3].ξ
    ]
    ξc3 = [
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in lstd[1].ξ, ξy in lstd[2].ξ, ξz in lstd[3].ξc[1]
    ]
    ξc = (ξc1, ξc2, ξc3)

    cindices = CartesianIndices((np...,)) |> collect
    lindices = LinearIndices((np...,)) |> collect

    # Mass matrix
    M = Diagonal(ω)

    # Derivative matrices
    I = (Diagonal(ones(np[1])), Diagonal(ones(np[2])), Diagonal(ones(np[3])))
    Iω = (Diagonal(lstd[1].ω), Diagonal(lstd[2].ω), Diagonal(lstd[3].ω))
    D = (
        kron(I[3], I[2], lstd[1].D[1]),
        kron(I[3], lstd[2].D[1], I[1]),
        kron(lstd[3].D[1], I[2], I[1]),
    )
    Q = (
        kron(I[3], I[2], lstd[1].Q[1]),
        kron(I[3], lstd[2].Q[1], I[1]),
        kron(lstd[3].Q[1], I[2], I[1]),
    )
    K = (
        kron(Iω[3], Iω[2], lstd[1].K[1]),
        kron(Iω[3], lstd[2].K[1], Iω[1]),
        kron(lstd[3].K[1], Iω[2], Iω[1]),
    )
    Ks = (
        kron(Iω[3], Iω[2], lstd[1].Ks[1]),
        kron(Iω[3], lstd[2].Ks[1], Iω[1]),
        kron(lstd[3].Ks[1], Iω[2], Iω[1]),
    )
    K♯ = (
        kron(diag(Iω[3]), diag(Iω[2]), lstd[1].K♯[1]),
        kron(diag(Iω[3]), lstd[2].K♯[1], diag(Iω[1])),
        kron(lstd[3].K♯[1], diag(Iω[2]), diag(Iω[1])),
    )

    # Projection operator
    l = (
        kron(I[3], I[2], lstd[1].l[1]),
        kron(I[3], I[2], lstd[1].l[2]),
        kron(I[3], lstd[2].l[1], I[1]),
        kron(I[3], lstd[2].l[2], I[1]),
        kron(lstd[3].l[1], I[1], I[2]),
        kron(lstd[3].l[2], I[1], I[2]),
    )

    # Surface contribution
    lω = (
        kron(Iω[3], Iω[2], lstd[1].lω[1]),
        kron(Iω[3], Iω[2], lstd[1].lω[2]),
        kron(Iω[3], lstd[2].lω[1], Iω[1]),
        kron(Iω[3], lstd[2].lω[2], Iω[1]),
        kron(lstd[3].lω[1], Iω[2], Iω[1]),
        kron(lstd[3].lω[2], Iω[2], Iω[1]),
    )

    return StdHex{
        typeof(qtype),
        Tuple(np),
        eltype(ω),
        typeof(M),
        typeof(cindices),
        typeof(lindices),
        typeof(fstd[1]),
        typeof(fstd[2]),
        typeof(fstd[3]),
    }(
        fstd,
        cindices,
        lindices,
        ξe,
        ξc,
        ξ,
        ω,
        M,
        D .|> transpose .|> sparse .|> transpose |> Tuple,
        Q .|> transpose .|> sparse .|> transpose |> Tuple,
        K .|> transpose .|> sparse .|> transpose |> Tuple,
        Ks .|> transpose .|> sparse .|> transpose |> Tuple,
        K♯ .|> transpose .|> collect .|> transpose |> Tuple,
        l .|> transpose .|> sparse .|> transpose |> Tuple,
        lω .|> transpose .|> sparse .|> transpose |> Tuple,
        _n2e |> transpose |> collect |> transpose,
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::StdHex{QT,D,RT}) where {QT,D,RT}
    @nospecialize
    println(io, "StdHex{", RT, "}: ")
    if QT == GaussQuadrature
        println(io, " ξ: Gauss quadrature with ", D[1], " nodes")
        println(io, " η: Gauss quadrature with ", D[2], " nodes")
        print(io, " ζ: Gauss quadrature with ", D[3], " nodes")
    elseif QT == GaussLobattoQuadrature
        println(io, " ξ: Gauss-Lobatto quadrature with ", D[1], " nodes")
        println(io, " η: Gauss-Lobatto quadrature with ", D[2], " nodes")
        print(io, " ζ: Gauss-Lobatto quadrature with ", D[3], " nodes")
    else
        @assert false "[StdRegion.show] You shouldn't be here..."
    end
    return nothing
end

function massmatrix(std::StdHex, J)
    return Diagonal(J) * std.M
end

function slave2master(_, _, _::StdHex)
    error("Not implemented yet!")
end

function master2slave(_, _, _::StdHex)
    error("Not implemented yet!")
end

_VTK_type(::StdHex) = UInt8(72)

function _VTK_connectivities(s::StdHex)
    nx, ny, nz = size(s)
    li = LinearIndices(s)
    corners = [
        li[1, 1, 1], li[nx, 1, 1], li[nx, ny, 1], li[1, ny, 1],
        li[1, 1, nz], li[nx, 1, nz], li[nx, ny, nz], li[1, ny, nz],
    ]
    edges = reduce(vcat, [
        li[2:(end - 1), 1, 1], li[nx, 2:(end - 1), 1],
        li[2:(end - 1), ny, 1], li[1, 2:(end - 1), 1],
        li[2:(end - 1), 1, nz], li[nx, 2:(end - 1), nz],
        li[2:(end - 1), ny, nz], li[1, 2:(end - 1), nz],
        li[1, 1, 2:(end - 1)], li[nx, 1, 2:(end - 1)],
        li[nx, ny, 2:(end - 1)], li[1, ny, 2:(end - 1)],
    ])
    faces = reduce(vcat, [
        li[1, 2:(end - 1), 2:(end - 1)], li[nx, 2:(end - 1), 2:(end - 1)],
        li[2:(end - 1), 1, 2:(end - 1)], li[2:(end - 1), ny, 2:(end - 1)],
        li[2:(end - 1), 2:(end - 1), 1], li[2:(end - 1), 2:(end - 1), nz],
    ] .|> vec)
    interior = vec(li[2:(end - 1), 2:(end - 1), 2:(end - 1)])
    return mapreduce(x -> x .- 1, vcat, (corners, edges, faces, interior))
end
