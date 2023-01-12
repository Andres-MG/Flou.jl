abstract type AbstractQuadrature end

struct GaussQuadrature <: AbstractQuadrature end
struct GaussLobattoQuadrature <: AbstractQuadrature end

const GL = GaussQuadrature
const GLL = GaussLobattoQuadrature

abstract type AbstractStdRegion{QT,ND,NP} end

Base.size(::AbstractStdRegion{QT,ND,NP}) where {QT,ND,NP} = ntuple(_ -> NP, ND)
Base.size(::AbstractStdRegion{QT,ND,NP}, i) where {QT,ND,NP} = NP
Base.length(::AbstractStdRegion{QT,ND,NP}) where {QT,ND,NP} = NP^ND
Base.eachindex(s::AbstractStdRegion, i) = Base.OneTo(size(s, i))
Base.eachindex(s::AbstractStdRegion) = Base.OneTo(length(s))
Base.LinearIndices(s::AbstractStdRegion) = LinearIndices(size(s))
Base.CartesianIndices(s::AbstractStdRegion) = CartesianIndices(size(s))

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

get_spatialdim(::AbstractStdRegion{QT,ND}) where {QT,ND} = ND
get_quadrature(::AbstractStdRegion{QT}) where {QT} = QT

ndofs(s::AbstractStdRegion; equispaced=false) = equispaced ? length(s.ξe) : length(s)

eachdirection(s::AbstractStdRegion) = Base.OneTo(ndirections(s))
eachdof(s::AbstractStdRegion) = Base.OneTo(ndofs(s))

#==========================================================================================#
#                                     Temporary cache                                      #

struct StdRegionCache{S,T,B,SH,SC}
    scalar::Vector{NTuple{3,S}}
    state::Vector{NTuple{3,T}}
    block::Vector{NTuple{3,B}}
    sharp::Vector{NTuple{3,SH}}
    subcell::Vector{NTuple{3,SC}}
end

function StdRegionCache{RT}(::Val{ND}, np, ::Val{NV}, ne) where {ND,RT,NV}
    nthr = Threads.nthreads()
    npts = np^ND
    tmps = [
        ntuple(_ -> Vector{RT}(undef, npts), 3)
        for _ in 1:nthr
    ]
    tmpst = [
        ntuple(_ -> HybridVector{NV,RT}(undef, npts), 3)
        for _ in 1:nthr
    ]
    tmpb = [
        ntuple(_ -> HybridMatrix{NV,RT}(undef, npts, ND), 3)
        for _ in 1:nthr
    ]
    tmp♯ = [
        ntuple(i -> HybridMatrix{NV,RT}(undef, np, np), 3)
        for _ in 1:nthr
    ]
    tmpsubcell = [
        ntuple(i -> HybridVector{NV,RT}(undef, np + 1), 3)
        for _ in 1:nthr
    ]
    return StdRegionCache(tmps, tmpst, tmpb, tmp♯, tmpsubcell)
end

#==========================================================================================#
#                                      Standard point                                      #

struct StdPoint <: AbstractStdRegion{GaussQuadrature,0,1} end

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

struct StdSegment{QT,NP,RT,MM,F,C} <: AbstractStdRegion{QT,1,NP}
    face::F
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

function StdSegment{RT}(np::Integer, qtype, nvars; npe=np[1]) where {RT}
    return StdSegment{RT}(Val(np), qtype, Val(nvars); npe=npe)
end

function StdSegment{RT}(
    ::Val{NP},
    qtype::AbstractQuadrature,
    nvars::Val{NV}=Val(1);
    npe=NP[1],
) where {
    NP,
    RT<:Real,
    NV
}
    fstd = StdPoint()
    _ξ, ω = if qtype isa(GaussQuadrature)
        gausslegendre(NP)
    elseif qtype isa(GaussLobattoQuadrature)
        gausslobatto(NP)
    else
        throw(ArgumentError("Only Gauss and Gauss-Lobatto quadratures are implemented."))
    end
    _ξ, ω = convert.(RT, _ξ), convert.(RT, ω)
    ξ = [SVector(ξi) for ξi in _ξ]

    # Equispaced nodes
    _ξe = convert.(RT, range(-1, 1, npe))
    ξe = [SVector(ξ) for ξ in _ξe]

    # Complementary nodes
    ξc1 = Vector{SVector{1,RT}}(undef, NP + 1)
    ξc1[1] = SVector(-one(RT))
    for i in 1:NP
        ξc1[i + 1] = ξc1[i] .+ ω[i]
    end
    ξc2 = Vector{SVector{1,RT}}(undef, NP + 1)
    ξc2[NP + 1] = SVector(one(RT))
    for i in NP:-1:1
        ξc2[i] = ξc2[i + 1] .- ω[i]
    end
    ξc = (ξc1 .+ ξc2) ./ 2
    ξc[1] = SVector(-one(RT))
    ξc[NP + 1] = SVector(one(RT))
    ξc = tuple(ξc)

    # Mass matrix
    M = Diagonal(ω)

    # Lagrange basis
    D = Matrix{RT}(undef, NP, NP)
    _n2e = Matrix{RT}(undef, npe, NP)
    l = (Vector{RT}(undef, NP), Vector{RT}(undef, NP))
    y = fill(zero(RT), NP)
    for i in 1:NP
        y[i] = one(RT)
        Li = fit(_ξ, y)
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
    cache = StdRegionCache{RT}(Val(1), NP, nvars, npe)

    return StdSegment{
        typeof(qtype),
        NP,
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

function Base.show(io::IO, ::MIME"text/plain", s::StdSegment)
    @nospecialize

    rt = eltype(s.D[1])
    qt = get_quadrature(s)
    np = ndofs(s)

    print(io, "StdSegment{", rt, "}: ")
    if qt == GaussQuadrature
        print(io, "Gauss quadrature with ", np, " nodes")
    elseif qt == GaussLobattoQuadrature
        print(io, "Gauss-Lobatto quadrature with ", np, " nodes")
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

struct StdQuad{QT,NP,RT,MM,F,C} <: AbstractStdRegion{QT,2,NP}
    face::F
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

is_tensor_product(::StdQuad) = true
ndirections(::StdQuad) = 2
nvertices(::StdQuad) = 4

function StdQuad{RT}(np::Integer, qtype, nvars; npe=nothing) where {RT}
    return StdQuad{RT}(Val(np), qtype, Val(nvars); npe=npe)
end

function StdQuad{RT}(
    ::Val{NP},
    qtype::AbstractQuadrature,
    nvars::Val{NV}=Val(1);
    npe=nothing,
) where {
    NP,
    RT<:Real,
    NV
}
    # Quadratures
    npe = npe === nothing ? NP : npe

    fstd = StdSegment{RT}(Val(NP), qtype, nvars; npe=npe)

    ξ = vec([SVector(ξx[1], ξy[1]) for ξx in fstd.ξ, ξy in fstd.ξ])
    ω = vec([ωx * ωy for ωx in fstd.ω, ωy in fstd.ω])

    # Equispaced nodes
    ξe = vec([SVector(ξx[1], ξy[1]) for ξx in fstd.ξe, ξy in fstd.ξe])
    _n2e = kron(fstd._n2e, fstd._n2e)

    # Complementary nodes
    ξc1 = [SVector(ξx[1], ξy[1]) for ξx in fstd.ξc[1], ξy in fstd.ξ]
    ξc2 = [SVector(ξx[1], ξy[1]) for ξx in fstd.ξ, ξy in fstd.ξc[1]]
    ξc = (ξc1, ξc2)

    # Mass matrix
    M = Diagonal(ω)

    # Derivative matrices
    I = Diagonal(ones(NP))
    Iω = Diagonal(fstd.ω)
    D = (
        kron(I, fstd.D[1]),
        kron(fstd.D[1], I),
    )
    Q = (
        kron(I, fstd.Q[1]),
        kron(fstd.Q[1], I),
    )
    K = (
        kron(Iω, fstd.K[1]),
        kron(fstd.K[1], Iω),
    )
    Ks = (
        kron(Iω, fstd.Ks[1]),
        kron(fstd.Ks[1], Iω),
    )
    K♯ = ntuple(_ -> fstd.K♯[1], 2)

    # Projection operator
    l = (
        kron(I, fstd.l[1]),
        kron(I, fstd.l[2]),
        kron(fstd.l[1], I),
        kron(fstd.l[2], I),
    )

    # Surface contribution
    lω = (
        kron(Iω, fstd.lω[1]),
        kron(Iω, fstd.lω[2]),
        kron(fstd.lω[1], Iω),
        kron(fstd.lω[2], Iω),
    )

    # Temporary storage
    cache = StdRegionCache{RT}(Val(2), NP, nvars, npe)

    return StdQuad{
        typeof(qtype),
        NP,
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

function Base.show(io::IO, ::MIME"text/plain", s::StdQuad)
    @nospecialize

    rt = eltype(s.D[1])
    np = size(s, 1)
    qt = get_quadrature(s)

    print(io, "StdQuad{", rt, "}: ")
    nodes = "$(np)×$(np)"
    if qt == GaussQuadrature
        print(io, "Gauss quadrature with ", nodes, " nodes")
    elseif qt == GaussLobattoQuadrature
        print(io, "Gauss-Lobatto quadrature with ", nodes, " nodes")
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
    li = LinearIndices(std)
    return if orientation == 0
        i
    elseif orientation == 1
        li[size(std, 1) - s[2] + 1, s[1]]
    elseif orientation == 2
        li[size(std, 1) - s[1] + 1, size(std, 2) - s[2] + 1]
    elseif orientation == 3
        li[s[2], size(std, 2) - s[1] + 1]
    elseif orientation == 4
        li[s[2], s[1]]
    elseif orientation == 5
        li[size(std, 1) - s[1] + 1, s[2]]
    elseif orientation == 6
        li[size(std, 1) - s[2] + 1, size(std, 2) - s[1] + 1]
    else # orientation == 7
        li[s[1], size(std, 2) - s[2] + 1]
    end
end

function master2slave(i::Integer, orientation, std::StdQuad)
    m = CartesianIndices(std)[i]
    li = LinearIndices(std)
    return if orientation == 0
        i
    elseif orientation == 1
        li[m[2], size(std, 1) - m[1] + 1]
    elseif orientation == 2
        li[size(std, 1) - m[1] + 1, size(std, 2) - m[2] + 1]
    elseif orientation == 3
        li[size(std, 2) - m[2] + 1, m[1]]
    elseif orientation == 4
        li[m[2], m[1]]
    elseif orientation == 5
        li[size(std, 1) - m[1] + 1, m[2]]
    elseif orientation == 6
        li[size(std, 2) - m[2] + 1, size(std, 1) - m[1] + 1]
    else # orientation == 7
        li[m[1], size(std, 2) - m[2] + 1]
    end
end

_VTK_type(::StdQuad) = UInt8(70)

function _VTK_connectivities(s::StdQuad)
    n = size(s, 1)
    li = LinearIndices(s)
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

struct StdTri{Q,NP} <: AbstractStdRegion{Q,2,NP} end

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

struct StdHex{QT,NP,RT,MM,F,E,C} <: AbstractStdRegion{QT,3,NP}
    face::F
    edge::E
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

function StdHex{RT}(np::Integer, qtype, nvars; npe=nothing) where {RT}
    return StdHex{RT}(Val(np), qtype, Val(nvars); npe=npe)
end

function StdHex{RT}(
    ::Val{NP},
    qtype::AbstractQuadrature,
    nvars::Val{NV}=Val(1);
    npe=nothing,
) where {
    NP,
    RT<:Real,
    NV
}
    # Quadratures
    npe = npe === nothing ? NP : npe

    fstd = StdQuad{RT}(Val(NP), qtype, nvars; npe=npe)
    estd = fstd.face
    ξ = vec([
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in estd.ξ, ξy in estd.ξ, ξz in estd.ξ
    ])
    ω = vec([ωx * ωy * ωz for ωx in estd.ω, ωy in estd.ω, ωz in estd.ω])

    # Equispaced nodes
    ξe = vec([
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in estd.ξe, ξy in estd.ξe, ξz in estd.ξe
    ])
    _n2e = kron(estd._n2e, estd._n2e, estd._n2e)

    # Complementary nodes
    ξc1 = [
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in estd.ξc[1], ξy in estd.ξ, ξz in estd.ξ
    ]
    ξc2 = [
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in estd.ξ, ξy in estd.ξc[1], ξz in estd.ξ
    ]
    ξc3 = [
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in estd.ξ, ξy in estd.ξ, ξz in estd.ξc[1]
    ]
    ξc = (ξc1, ξc2, ξc3)

    # Mass matrix
    M = Diagonal(ω)

    # Derivative matrices
    I = Diagonal(ones(NP))
    Iω = Diagonal(estd.ω)
    D = (
        kron(I, I, estd.D[1]),
        kron(I, estd.D[1], I),
        kron(estd.D[1], I, I),
    )
    Q = (
        kron(I, I, estd.Q[1]),
        kron(I, estd.Q[1], I),
        kron(estd.Q[1], I, I),
    )
    K = (
        kron(Iω, Iω, estd.K[1]),
        kron(Iω, estd.K[1], Iω),
        kron(estd.K[1], Iω, Iω),
    )
    Ks = (
        kron(Iω, Iω, estd.Ks[1]),
        kron(Iω, estd.Ks[1], Iω),
        kron(estd.Ks[1], Iω, Iω),
    )
    K♯ = ntuple(_ -> estd.K♯[1], 3)

    # Projection operator
    l = (
        kron(I, I, estd.l[1]),
        kron(I, I, estd.l[2]),
        kron(I, estd.l[1], I),
        kron(I, estd.l[2], I),
        kron(estd.l[1], I, I),
        kron(estd.l[2], I, I),
    )

    # Surface contribution
    lω = (
        kron(Iω, Iω, estd.lω[1]),
        kron(Iω, Iω, estd.lω[2]),
        kron(Iω, estd.lω[1], Iω),
        kron(Iω, estd.lω[2], Iω),
        kron(estd.lω[1], Iω, Iω),
        kron(estd.lω[2], Iω, Iω),
    )

    # Temporary storage
    cache = StdRegionCache{RT}(Val(3), NP, nvars, npe)

    return StdHex{
        typeof(qtype),
        NP,
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

function Base.show(io::IO, ::MIME"text/plain", s::StdHex)
    @nospecialize

    rt = eltype(s.D[1])
    np = size(s, 1)
    qt = get_quadrature(s)

    print(io, "StdHex{", rt, "}: ")
    nodes = "$(np)×$(np)×$(np)"
    if qt == GaussQuadrature
        print(io, "Gauss quadrature with ", nodes, " nodes")
    elseif qt == GaussLobattoQuadrature
        print(io, "Gauss-Lobatto quadrature with ", nodes, " nodes")
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
    n = size(s, 1)
    li = LinearIndices(s)
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
