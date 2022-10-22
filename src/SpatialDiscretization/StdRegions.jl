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
Base.LinearIndices(s::AbstractStdRegion, _=false) = LinearIndices(size(s))
Base.CartesianIndices(s::AbstractStdRegion, _=false) = CartesianIndices(size(s))

"""
    is_tensor_product(std)

Return `true` if `std` is a tensor-product standard region, and `false` otherwise.
"""
function is_tensor_product end

function ndirections end

function massmatrix end

Base.@propagate_inbounds function project2equispaced!(Qe, Q, s::AbstractStdRegion)
    mul!(Qe, s._n2e, Q)
    return nothing
end

function slave2master end

function master2slave end

get_spatialdim(::AbstractStdRegion{ND}) where {ND} = ND
get_quadrature(::AbstractStdRegion{ND,Q}) where {ND,Q} = Q

ndofs(s::AbstractStdRegion, equispaced=false) = equispaced ? length(s.ξe) : length(s)

eachdirection(s::AbstractStdRegion) = Base.OneTo(ndirections(s))
eachdof(s::AbstractStdRegion) = Base.OneTo(ndofs(s))

#==========================================================================================#
#                                     Temporary cache                                      #

struct StdRegionCache{ND,RT,V}
    scalar::Vector{NTuple{3,Matrix{RT}}}
    vector::Vector{NTuple{3,Array{RT,3}}}
    sharp::Vector{NTuple{ND,Array{RT,3}}}
    sharptensor::Vector{NTuple{ND,V}}
    subcell::Vector{NTuple{ND,Matrix{RT}}}
end

function StdRegionCache{RT}(np, nvars) where {RT}
    ndims = length(np)
    nthr = Threads.nthreads()
    tmps = [
        ntuple(_ -> Matrix{RT}(undef, prod(np), nvars), 3)
        for _ in 1:nthr
    ]
    tmpv = [
        ntuple(_ -> Array{RT,3}(undef, prod(np), nvars, ndims), 3)
        for _ in 1:nthr
    ]
    tmp♯ = [
        ntuple(i -> Array{RT,3}(undef, np[i], prod(np), nvars), ndims)
        for _ in 1:nthr
    ]
    tmp♯tensor = [
        ntuple(i -> reshape(tmp♯[ithr][i], np[i], np..., nvars), ndims)
        for ithr in 1:nthr
    ]
    tmpsubcell = [
        ntuple(i -> Matrix{RT}(undef, np[i] + 1, nvars), ndims)
        for _ in 1:nthr
    ]
    return StdRegionCache(tmps, tmpv, tmp♯, tmp♯tensor, tmpsubcell)
end

#==========================================================================================#
#                                      Standard point                                      #

struct StdPoint <: AbstractStdRegion{0,GaussQuadrature,(1,)} end

is_tensor_product(::StdPoint) = true
ndirections(::StdPoint) = 1
nvertices(::StdPoint) = 1

function slave2master(i::Integer, _, ::StdPoint)
    return i
end

function master2slave(i::Integer, _, ::StdPoint)
    return i
end

#==========================================================================================#
#                                     Standard segment                                     #

struct StdSegment{QT,Dims,RT,MM,F,C} <: AbstractStdRegion{1,QT,Dims}
    faces::F
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
    cache::C
end

is_tensor_product(::StdSegment) = true
ndirections(::StdSegment) = 1
nvertices(::StdSegment) = 2

function StdSegment{RT}(
    np::Integer,
    qtype::AbstractQuadrature,
    nvars::Integer=1;
    npe=np,
) where {
    RT<:Real,
}
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

    # Temporary storage
    cache = StdRegionCache{RT}(np, nvars)

    return StdSegment{
        typeof(qtype),
        Tuple(np),
        eltype(ω),
        typeof(M),
        typeof(fstd),
        typeof(cache),
    }(
        fstd,
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
        cache,
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

function slave2master(i::Integer, orientation, std::StdSegment)
    return if orientation == 0
        i
    else # orientation == 1
        ndofs(std) - (i - 1)
    end
end

function master2slave(i::Integer, orientation, std::StdSegment)
    return if orientation == 0
        i
    else # orientation == 1
        ndofs(std) - (i - 1)
    end
end

_VTK_type(::StdSegment) = UInt8(68)

function _VTK_connectivities(s::StdSegment)
    conns = [1, ndofs(s)]
    append!(conns, 2:(ndofs(s) - 1))
    return conns .- 1
end

#==========================================================================================#
#                                       Standard quad                                      #

struct StdQuad{QT,Dims,RT,MM,F,C} <: AbstractStdRegion{2,QT,Dims}
    faces::F
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
    cache::C
end

function Base.LinearIndices(s::StdQuad, transpose=false)
    return if transpose
        LinearIndices((size(s, 2), size(s, 1)))
    else
        LinearIndices(size(s))
    end
end
function Base.CartesianIndices(s::StdQuad, transpose=false)
    return if transpose
        CartesianIndices((size(s, 2), size(s, 1)))
    else
        CartesianIndices(size(s))
    end
end

is_tensor_product(::StdQuad) = true
ndirections(::StdQuad) = 2
nvertices(::StdQuad) = 4

function StdQuad{RT}(
    np::AbstractVecOrTuple,
    qtype::AbstractQuadrature,
    nvars::Integer=1;
    npe=nothing,
) where {
    RT<:Real,
}
    # Quadratures
    if isnothing(npe)
        npe = maximum(np)
    end
    fstd = (
        StdSegment{RT}(np[2], qtype, nvars; npe=npe),
        StdSegment{RT}(np[1], qtype, nvars; npe=npe),
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

    # Temporary storage
    cache = StdRegionCache{RT}(np, nvars)

    fstd = (fstd[1], fstd[1], fstd[2], fstd[2])
    return StdQuad{
        typeof(qtype),
        Tuple(np),
        eltype(ω),
        typeof(M),
        typeof(fstd),
        typeof(cache),
    }(
        fstd,
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
        cache,
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

function slave2master(i::Integer, orientation, std::StdQuad)
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

function master2slave(i::Integer, orientation, std::StdQuad)
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
    n = size(s) |> maximum
    li = LinearIndices((n, n))
    corners = [li[1, 1], li[n, 1], li[n, n], li[1, n]]
    edges = reduce(vcat, [
        li[2:(n - 1), 1], li[n, 2:(n - 1)],
        li[2:(n - 1), n], li[1, 2:(n - 1)],
    ])
    interior = vec(li[2:(n - 1), 2:(n - 1)])
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

struct StdHex{QT,Dims,RT,MM,F,E,C} <: AbstractStdRegion{3,QT,Dims}
    faces::F
    edges::E
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
    cache::C
end

is_tensor_product(::StdHex) = true
ndirections(::StdHex) = 3
nvertices(::StdHex) = 8

function StdHex{RT}(
    np::AbstractVecOrTuple,
    qtype::AbstractQuadrature,
    nvars::Integer=1;
    npe=nothing,
) where {
    RT<:Real,
}
    # Quadratures
    if isnothing(npe)
        npe = maximum(np)
    end
    fstd = (
        StdQuad{RT}((np[2], np[3]), qtype, nvars; npe=npe),
        StdQuad{RT}((np[1], np[3]), qtype, nvars; npe=npe),
        StdQuad{RT}((np[1], np[2]), qtype, nvars; npe=npe),
    )
    estd = (fstd[3].faces[3], fstd[3].faces[1], fstd[1].faces[1])
    ξ = vec([
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in estd[1].ξ, ξy in estd[2].ξ, ξz in estd[3].ξ
    ])
    ω = vec([ωx * ωy * ωz for ωx in estd[1].ω, ωy in estd[2].ω, ωz in estd[3].ω])

    # Equispaced nodes
    ξe = vec([
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in estd[1].ξe, ξy in estd[2].ξe, ξz in estd[3].ξe
    ])
    _n2e = kron(estd[3]._n2e, estd[2]._n2e, estd[1]._n2e)

    # Complementary nodes
    ξc1 = [
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in estd[1].ξc[1], ξy in estd[2].ξ, ξz in estd[3].ξ
    ]
    ξc2 = [
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in estd[1].ξ, ξy in estd[2].ξc[1], ξz in estd[3].ξ
    ]
    ξc3 = [
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in estd[1].ξ, ξy in estd[2].ξ, ξz in estd[3].ξc[1]
    ]
    ξc = (ξc1, ξc2, ξc3)

    # Mass matrix
    M = Diagonal(ω)

    # Derivative matrices
    I = (Diagonal(ones(np[1])), Diagonal(ones(np[2])), Diagonal(ones(np[3])))
    Iω = (Diagonal(estd[1].ω), Diagonal(estd[2].ω), Diagonal(estd[3].ω))
    D = (
        kron(I[3], I[2], estd[1].D[1]),
        kron(I[3], estd[2].D[1], I[1]),
        kron(estd[3].D[1], I[2], I[1]),
    )
    Q = (
        kron(I[3], I[2], estd[1].Q[1]),
        kron(I[3], estd[2].Q[1], I[1]),
        kron(estd[3].Q[1], I[2], I[1]),
    )
    K = (
        kron(Iω[3], Iω[2], estd[1].K[1]),
        kron(Iω[3], estd[2].K[1], Iω[1]),
        kron(estd[3].K[1], Iω[2], Iω[1]),
    )
    Ks = (
        kron(Iω[3], Iω[2], estd[1].Ks[1]),
        kron(Iω[3], estd[2].Ks[1], Iω[1]),
        kron(estd[3].Ks[1], Iω[2], Iω[1]),
    )
    K♯ = (
        kron(diag(Iω[3]), diag(Iω[2]), estd[1].K♯[1]),
        kron(diag(Iω[3]), estd[2].K♯[1], diag(Iω[1])),
        kron(estd[3].K♯[1], diag(Iω[2]), diag(Iω[1])),
    )

    # Projection operator
    l = (
        kron(I[3], I[2], estd[1].l[1]),
        kron(I[3], I[2], estd[1].l[2]),
        kron(I[3], estd[2].l[1], I[1]),
        kron(I[3], estd[2].l[2], I[1]),
        kron(estd[3].l[1], I[1], I[2]),
        kron(estd[3].l[2], I[1], I[2]),
    )

    # Surface contribution
    lω = (
        kron(Iω[3], Iω[2], estd[1].lω[1]),
        kron(Iω[3], Iω[2], estd[1].lω[2]),
        kron(Iω[3], estd[2].lω[1], Iω[1]),
        kron(Iω[3], estd[2].lω[2], Iω[1]),
        kron(estd[3].lω[1], Iω[2], Iω[1]),
        kron(estd[3].lω[2], Iω[2], Iω[1]),
    )

    # Temporary storage
    cache = StdRegionCache{RT}(np, nvars)

    fstd = (fstd[1], fstd[1], fstd[2], fstd[2], fstd[3], fstd[3])
    return StdHex{
        typeof(qtype),
        Tuple(np),
        eltype(ω),
        typeof(M),
        typeof(fstd),
        typeof(estd),
        typeof(cache),
    }(
        fstd,
        estd,
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
        cache,
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
    n = size(s) |> maximum
    li = LinearIndices((n, n, n))
    corners = [
        li[1, 1, 1], li[n, 1, 1], li[n, n, 1], li[1, n, 1],
        li[1, 1, n], li[n, 1, n], li[n, n, n], li[1, n, n],
    ]
    edges = reduce(vcat, [
        li[2:(n - 1), 1, 1], li[n, 2:(n - 1), 1],
        li[2:(n - 1), n, 1], li[1, 2:(n - 1), 1],
        li[2:(n - 1), 1, n], li[n, 2:(n - 1), n],
        li[2:(n - 1), n, n], li[1, 2:(n - 1), n],
        li[1, 1, 2:(n - 1)], li[n, 1, 2:(n - 1)],
        li[n, n, 2:(n - 1)], li[1, n, 2:(n - 1)],
    ])
    faces = reduce(vcat, [
        li[1, 2:(n - 1), 2:(n - 1)], li[n, 2:(n - 1), 2:(n - 1)],
        li[2:(n - 1), 1, 2:(n - 1)], li[2:(n - 1), n, 2:(n - 1)],
        li[2:(n - 1), 2:(n - 1), 1], li[2:(n - 1), 2:(n - 1), n],
    ] .|> vec)
    interior = vec(li[2:(end - 1), 2:(end - 1), 2:(end - 1)])
    return mapreduce(x -> x .- 1, vcat, (corners, edges, faces, interior))
end
