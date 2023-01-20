#==========================================================================================#
#                                     Temporary cache                                      #

struct FRStdRegionCache{S,T,B,SH,SC}
    scalar::Vector{NTuple{3,S}}
    state::Vector{NTuple{3,T}}
    block::Vector{NTuple{3,B}}
    sharp::Vector{NTuple{3,SH}}
    subcell::Vector{NTuple{3,SC}}
end

function FRStdRegionCache{RT}(nd, np, ::Val{NV}) where {RT,NV}
    nthr = Threads.nthreads()
    npts = np^nd
    tmps = [
        ntuple(_ -> Vector{RT}(undef, npts), 3)
        for _ in 1:nthr
    ]
    tmpst = [
        ntuple(_ -> HybridVector{NV,RT}(undef, npts), 3)
        for _ in 1:nthr
    ]
    tmpb = [
        ntuple(_ -> HybridMatrix{NV,RT}(undef, npts, nd), 3)
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
    return FRStdRegionCache(tmps, tmpst, tmpb, tmp♯, tmpsubcell)
end

#==========================================================================================#
#                                      Standard point                                      #

struct FRStdPoint <: AbstractStdPoint end

#==========================================================================================#
#                                     Standard segment                                     #

struct FRStdSegment{NP,SN,RT,C} <: AbstractStdSegment{NP,SN}
    face::FRStdPoint
    ξe::Vector{SVector{1,RT}}
    ξc::Tuple{Vector{SVector{1,RT}}}
    ξ::Vector{SVector{1,RT}}
    ω::Vector{RT}
    D::Tuple{Transpose{RT,Matrix{RT}}}
    Dw::Tuple{Transpose{RT,Matrix{RT}}}
    D♯::Tuple{Transpose{RT,Matrix{RT}}}
    l::NTuple{2,Transpose{RT,Vector{RT}}}   # Row vectors
    ∂g::NTuple{2,Vector{RT}}
    node2eq::Transpose{RT,Matrix{RT}}
    cache::C
    reconstruction::Symbol
end

function FRStdSegment{RT}(
    np::Integer,
    qtype,
    reconstruction,
    nvars=1;
    nequispaced=np,
) where {
    RT,
}
    return FRStdSegment{RT}(
        Val(np), qtype, reconstruction, Val(nvars); nequispaced=nequispaced,
    )
end

function FRStdSegment{RT}(
    ::Val{NP},
    qtype::AbstractNodeDistribution,
    reconstruction::Symbol,
    nvars::Val{NV}=Val(1);
    nequispaced=NP,
) where {
    RT<:Real,
    NP,
    NV,
}
    # Face regions
    fstd = FRStdPoint()

    # Quadrature
    _ξ, ω = if qtype isa(GaussNodes)
        gausslegendre(NP)
    elseif qtype isa(GaussLobattoNodes)
        gausslobatto(NP)
    else
        throw(ArgumentError("Only Gauss and Gauss-Lobatto quadratures are implemented."))
    end
    _ξ, ω = convert.(RT, _ξ), convert.(RT, ω)
    ξ = [SVector(ξi) for ξi in _ξ]

    # Equispaced nodes
    _ξe = convert.(RT, range(-1, 1, nequispaced))
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

    # Lagrange basis
    D = Matrix{RT}(undef, NP, NP)
    node2eq = Matrix{RT}(undef, nequispaced, NP)
    l = (Vector{RT}(undef, NP), Vector{RT}(undef, NP))
    y = fill(zero(RT), NP)
    for i in 1:NP
        y[i] = one(RT)
        li = Polynomials.fit(_ξ, y)
        dli = Polynomials.derivative(li)
        D[:, i] .= dli.(_ξ)
        l[1][i] = li(-one(RT))
        l[2][i] = li(+one(RT))
        node2eq[:, i] .= li.(_ξe)
        y[i] = zero(RT)
    end

    # Surface contribution
    if reconstruction == :DGSEM
        ∂g = (l[1] ./ ω, l[2] ./ ω)
    else
        throw(ArgumentError("Unkown flux reconstruction type: $(reconstruction)"))
    end
    l = l .|> transpose |> Tuple

    # Derivative matrices
    B = ∂g[2] * l[2] - ∂g[1] * l[1]
    Dw = -D' .* ((1 ./ ω) * ω')
    D♯ = 2D - B

    # Temporary storage
    cache = FRStdRegionCache{RT}(1, NP, nvars)

    return FRStdSegment{
        NP,
        typeof(qtype),
        RT,
        typeof(cache),
    }(
        fstd,
        ξe,
        ξc,
        ξ,
        ω,
        D |> transpose |> collect |> transpose |> tuple,
        Dw |> transpose |> collect |> transpose |> tuple,
        D♯ |> transpose |> collect |> transpose |> tuple,
        l,
        ∂g,
        node2eq |> transpose |> collect |> transpose,
        cache,
        reconstruction,
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::FRStdSegment)
    @nospecialize

    rt = eltype(s.D[1])
    qt = nodetype(s)
    np = ndofs(s)

    qname = if qt == GaussNodes
        "Gauss"
    elseif qt == GaussLobattoNodes
        "Gauss-Lobatto"
    else
        @assert false "[StdRegion.show] You shouldn't be here..."
    end
    print(io, "FRStdSegment{", rt, "}: ")
    print(io, np, " ", qname, " node(s) with ", s.reconstruction, " reconstruction")

    return nothing
end

#==========================================================================================#
#                                       Standard quad                                      #

struct FRStdQuad{NP,SN,RT,F,C} <: AbstractStdQuad{NP,SN}
    face::F
    ξe::Vector{SVector{2,RT}}
    ξc::NTuple{2,Matrix{SVector{2,RT}}}
    ξ::Vector{SVector{2,RT}}
    ω::Vector{RT}
    D::NTuple{2,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    Dw::NTuple{2,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    D♯::NTuple{2,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    l::NTuple{4,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    ∂g::NTuple{4,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    node2eq::Transpose{RT,Matrix{RT}}
    cache::C
    reconstruction::Symbol
end

function FRStdQuad{RT}(
    np::Integer,
    qtype,
    reconstruction,
    nvars=1;
    nequispaced=np,
) where {
    RT,
}
    return FRStdQuad{RT}(
        Val(np), qtype, reconstruction, Val(nvars); nequispaced=nequispaced,
    )
end

function FRStdQuad{RT}(
    ::Val{NP},
    qtype::AbstractNodeDistribution,
    reconstruction::Symbol,
    nvars::Val{NV}=Val(1);
    nequispaced=NP,
) where {
    RT<:Real,
    NP,
    NV,
}
    # Face regions
    fstd = FRStdSegment{RT}(Val(NP), qtype, reconstruction, nvars; nequispaced=nequispaced)

    # Quadrature
    ξ = vec([SVector(ξx[1], ξy[1]) for ξx in fstd.ξ, ξy in fstd.ξ])
    ω = vec([ωx * ωy for ωx in fstd.ω, ωy in fstd.ω])

    # Equispaced nodes
    ξe = vec([SVector(ξx[1], ξy[1]) for ξx in fstd.ξe, ξy in fstd.ξe])
    node2eq = kron(fstd.node2eq, fstd.node2eq)

    # Complementary nodes
    ξc1 = [SVector(ξx[1], ξy[1]) for ξx in fstd.ξc[1], ξy in fstd.ξ]
    ξc2 = [SVector(ξx[1], ξy[1]) for ξx in fstd.ξ, ξy in fstd.ξc[1]]
    ξc = (ξc1, ξc2)

    # Derivative matrices
    I = Diagonal(ones(NP))
    D = (
        kron(I, fstd.D[1]),
        kron(fstd.D[1], I),
    )
    Dw = (
        kron(I, fstd.Dw[1]),
        kron(fstd.Dw[1], I),
    )
    D♯ = ntuple(_ -> fstd.D♯[1], 2)

    # Projection operator
    l = (
        kron(I, fstd.l[1]),
        kron(I, fstd.l[2]),
        kron(fstd.l[1], I),
        kron(fstd.l[2], I),
    )

    # Surface contribution
    ∂g = (
        kron(I, fstd.∂g[1]),
        kron(I, fstd.∂g[2]),
        kron(fstd.∂g[1], I),
        kron(fstd.∂g[2], I),
    )

    # Temporary storage
    cache = FRStdRegionCache{RT}(2, NP, nvars)

    return FRStdQuad{
        NP,
        typeof(qtype),
        RT,
        typeof(fstd),
        typeof(cache),
    }(
        fstd,
        ξe,
        ξc,
        ξ,
        ω,
        D .|> transpose .|> sparse .|> transpose |> Tuple,
        Dw .|> transpose .|> sparse .|> transpose |> Tuple,
        D♯ .|> transpose .|> collect .|> transpose |> Tuple,
        l .|> transpose .|> sparse .|> transpose |> Tuple,
        ∂g .|> transpose .|> sparse .|> transpose |> Tuple,
        node2eq |> transpose |> collect |> transpose,
        cache,
        reconstruction,
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::FRStdQuad)
    @nospecialize

    rt = eltype(s.D[1])
    np = ndofs(s, 1)
    qt = nodetype(s)

    nodes = "$(np)×$(np)"
    qname = if qt == GaussNodes
        "Gauss"
    elseif qt == GaussLobattoNodes
        "Gauss-Lobatto"
    else
        @assert false "[StdRegion.show] You shouldn't be here..."
    end
    print(io, "FRStdQuad{", rt, "}: ")
    print(io, nodes, " ", qname, " node(s) with ", s.reconstruction, " reconstruction")

    return nothing
end

#==========================================================================================#
#                                     Standard triangle                                    #

struct FRStdTri{NP,SN} <: AbstractStdTri{NP,SN} end

#==========================================================================================#
#                                       Standard hex                                       #

struct FRStdHex{NP,SN,RT,F,E,C} <: AbstractStdHex{NP,SN}
    face::F
    edge::E
    ξe::Vector{SVector{3,RT}}
    ξc::NTuple{3,Array{SVector{3,RT},3}}
    ξ::Vector{SVector{3,RT}}
    ω::Vector{RT}
    D::NTuple{3,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    Dw::NTuple{3,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    D♯::NTuple{3,Transpose{RT,Matrix{RT}}}
    l::NTuple{6,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    ∂g::NTuple{6,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    node2eq::Transpose{RT,Matrix{RT}}
    cache::C
    reconstruction::Symbol
end

function FRStdHex{RT}(
    np::Integer,
    qtype,
    reconstruction,
    nvars=1;
    nequispaced=np,
) where {
    RT,
}
    return FRStdHex{RT}(
        Val(np), qtype, reconstruction, Val(nvars); nequispaced=nequispaced,
    )
end

function FRStdHex{RT}(
    ::Val{NP},
    qtype::AbstractNodeDistribution,
    reconstruction::Symbol,
    nvars::Val{NV}=Val(1);
    nequispaced=NP,
) where {
    RT<:Real,
    NP,
    NV,
}
    # Face and edge regions
    fstd = FRStdQuad{RT}(Val(NP), qtype, reconstruction, nvars; nequispaced=nequispaced)
    estd = fstd.face

    # Quadrature
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
    node2eq = kron(estd.node2eq, estd.node2eq, estd.node2eq)

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

    # Derivative matrices
    I = Diagonal(ones(NP))
    D = (
        kron(I, I, estd.D[1]),
        kron(I, estd.D[1], I),
        kron(estd.D[1], I, I),
    )
    Dw = (
        kron(I, I, estd.Dw[1]),
        kron(I, estd.Dw[1], I),
        kron(estd.Dw[1], I, I),
    )
    D♯ = ntuple(_ -> estd.D♯[1], 3)

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
    ∂g = (
        kron(I, I, estd.∂g[1]),
        kron(I, I, estd.∂g[2]),
        kron(I, estd.∂g[1], I),
        kron(I, estd.∂g[2], I),
        kron(estd.∂g[1], I, I),
        kron(estd.∂g[2], I, I),
    )

    # Temporary storage
    cache = FRStdRegionCache{RT}(3, NP, nvars)

    return FRStdHex{
        NP,
        typeof(qtype),
        RT,
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
        D .|> transpose .|> sparse .|> transpose |> Tuple,
        Dw .|> transpose .|> sparse .|> transpose |> Tuple,
        D♯ .|> transpose .|> collect .|> transpose |> Tuple,
        l .|> transpose .|> sparse .|> transpose |> Tuple,
        ∂g .|> transpose .|> sparse .|> transpose |> Tuple,
        node2eq |> transpose |> collect |> transpose,
        cache,
        reconstruction,
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::FRStdHex)
    @nospecialize

    rt = eltype(s.D[1])
    np = ndofs(s, 1)
    qt = nodetype(s)

    nodes = "$(np)×$(np)×$(np)"
    qname = if qt == GaussNodes
        "Gauss"
    elseif qt == GaussLobattoNodes
        "Gauss-Lobatto"
    else
        @assert false "[StdRegion.show] You shouldn't be here..."
    end
    print(io, "FRStdHex{", rt, "}: ")
    print(io, nodes, " ", qname, " node(s) with ", s.reconstruction, " reconstruction")

    return nothing
end
