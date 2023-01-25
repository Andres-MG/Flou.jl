# Copyright (C) 2023 Andrés Mateo Gabín
#
# This file is part of Flou.jl.
#
# Flou.jl is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# Flou.jl is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Flou.jl. If
# not, see <https://www.gnu.org/licenses/>.

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

struct FRStdPoint{RT} <: AbstractStdPoint
    nodetype::GaussNodes
    function FRStdPoint{RT}() where {RT}
        return new(GaussNodes{RT}(1))
    end
end

#==========================================================================================#
#                                     Standard segment                                     #

struct FRStdSegment{SN,RT,C,R} <: AbstractStdSegment{SN}
    nodetype::SN
    face::FRStdPoint
    ξe::Vector{SVector{1,RT}}
    ξc::Tuple{Vector{SVector{1,RT}}}
    ξ::Vector{SVector{1,RT}}
    ω::Vector{RT}
    D::Tuple{Transpose{RT,Matrix{RT}}}
    Ds::Tuple{Transpose{RT,Matrix{RT}}}
    D♯::Tuple{Transpose{RT,Matrix{RT}}}
    l::NTuple{2,Transpose{RT,Vector{RT}}}   # Row vectors
    ∂g::NTuple{2,Vector{RT}}
    node2eq::Transpose{RT,Matrix{RT}}
    cache::C
    reconstruction::R
end

function FRStdSegment(ntype, reconstruction, nvars=1; nequispaced=nnodes(ntype))
    return FRStdSegment(ntype, reconstruction, Val(nvars); nequispaced=nequispaced)
end

function FRStdSegment(
    ntype::AbstractNodeDistribution{RT},
    rtype::AbstractReconstruction{RT},
    nvars::Val{NV}=Val(1);
    nequispaced=nnodes(ntype),
) where {
    RT,
    NV,
}
    # Face regions
    fstd = FRStdPoint{RT}()

    # Quadrature
    np = nnodes(ntype)
    _ξ, ω = ntype.ξ, ntype.ω
    ξ = [SVector(ξi) for ξi in _ξ]

    # Equispaced nodes
    _ξe = convert.(RT, range(-1, 1, nequispaced))
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

    # Some matrix operators
    D = derivative_matrix(_ξ, ntype)
    node2eq = interp_matrix(_ξe, ntype)
    l = (
        interp_matrix([-one(RT)], ntype) |> vec,
        interp_matrix([+one(RT)], ntype) |> vec,
    )

    # Surface contribution
    ∂g = reconstruction(_ξ, rtype)
    ∂g[1] .= -∂g[1]
    l = l .|> transpose |> Tuple

    # Derivative matrices
    B = ∂g[2] * l[2] - ∂g[1] * l[1]
    Ds = D - B
    D♯ = 2D - B

    # Temporary storage
    cache = FRStdRegionCache{RT}(1, np, nvars)

    return FRStdSegment{
        typeof(ntype),
        RT,
        typeof(cache),
        typeof(rtype),
    }(
        ntype,
        fstd,
        ξe,
        ξc,
        ξ,
        ω,
        D |> transpose |> collect |> transpose |> tuple,
        Ds |> transpose |> collect |> transpose |> tuple,
        D♯ |> transpose |> collect |> transpose |> tuple,
        l,
        ∂g,
        node2eq |> transpose |> collect |> transpose,
        cache,
        rtype,
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::FRStdSegment)
    @nospecialize

    rt = eltype(s.D[1])
    qt = nodetype(s)
    np = nnodes(qt)

    rname = reconstruction_name(s.reconstruction)
    qname = if qt isa GaussNodes
        "Gauss"
    elseif qt isa GaussLobattoNodes
        "Gauss-Lobatto"
    else
        @assert false "[StdRegion.show] You shouldn't be here..."
    end
    print(io, "FRStdSegment{", rt, "}: ")
    print(io, np, " ", qname, " node(s) with ", rname, " reconstruction")

    return nothing
end

#==========================================================================================#
#                                       Standard quad                                      #

struct FRStdQuad{SN,RT,F,C,R} <: AbstractStdQuad{SN}
    nodetype::SN
    face::F
    ξe::Vector{SVector{2,RT}}
    ξc::NTuple{2,Matrix{SVector{2,RT}}}
    ξ::Vector{SVector{2,RT}}
    ω::Vector{RT}
    D::NTuple{2,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    Ds::NTuple{2,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    D♯::NTuple{2,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    l::NTuple{4,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    ∂g::NTuple{4,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    node2eq::Transpose{RT,Matrix{RT}}
    cache::C
    reconstruction::R
end

function FRStdQuad(ntype, reconstruction, nvars=1; nequispaced=nnodes(ntype))
    return FRStdQuad(ntype, reconstruction, Val(nvars); nequispaced=nequispaced)
end

function FRStdQuad(
    ntype::AbstractNodeDistribution{RT},
    rtype::AbstractReconstruction{RT},
    nvars::Val{NV}=Val(1);
    nequispaced=nnodes(ntype),
) where {
    RT,
    NV,
}
    # Face regions
    fstd = FRStdSegment(ntype, rtype, nvars; nequispaced=nequispaced)

    # Quadrature
    np = nnodes(ntype)
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
    I = Diagonal(ones(np))
    D = (
        kron(I, fstd.D[1]),
        kron(fstd.D[1], I),
    )
    Ds = (
        kron(I, fstd.Ds[1]),
        kron(fstd.Ds[1], I),
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
    cache = FRStdRegionCache{RT}(2, np, nvars)

    return FRStdQuad{
        typeof(ntype),
        RT,
        typeof(fstd),
        typeof(cache),
        typeof(rtype),
    }(
        ntype,
        fstd,
        ξe,
        ξc,
        ξ,
        ω,
        D .|> transpose .|> sparse .|> transpose |> Tuple,
        Ds .|> transpose .|> sparse .|> transpose |> Tuple,
        D♯ .|> transpose .|> collect .|> transpose |> Tuple,
        l .|> transpose .|> sparse .|> transpose |> Tuple,
        ∂g .|> transpose .|> sparse .|> transpose |> Tuple,
        node2eq |> transpose |> collect |> transpose,
        cache,
        rtype,
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::FRStdQuad)
    @nospecialize

    rt = eltype(s.D[1])
    qt = nodetype(s)
    np = nnodes(qt)

    nodes = "$(np)×$(np)"
    rname = reconstruction_name(s.reconstruction)
    qname = if qt isa GaussNodes
        "Gauss"
    elseif qt isa GaussLobattoNodes
        "Gauss-Lobatto"
    else
        @assert false "[StdRegion.show] You shouldn't be here..."
    end
    print(io, "FRStdQuad{", rt, "}: ")
    print(io, nodes, " ", qname, " node(s) with ", rname, " reconstruction")

    return nothing
end

#==========================================================================================#
#                                       Standard hex                                       #

struct FRStdHex{SN,RT,F,E,C,R} <: AbstractStdHex{SN}
    nodetype::SN
    face::F
    edge::E
    ξe::Vector{SVector{3,RT}}
    ξc::NTuple{3,Array{SVector{3,RT},3}}
    ξ::Vector{SVector{3,RT}}
    ω::Vector{RT}
    D::NTuple{3,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    Ds::NTuple{3,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    D♯::NTuple{3,Transpose{RT,Matrix{RT}}}
    l::NTuple{6,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    ∂g::NTuple{6,Transpose{RT,SparseMatrixCSC{RT,Int}}}
    node2eq::Transpose{RT,Matrix{RT}}
    cache::C
    reconstruction::R
end

function FRStdHex(ntype, reconstruction, nvars=1; nequispaced=nnodes(ntype))
    return FRStdHex(ntype, reconstruction, Val(nvars); nequispaced=nequispaced)
end

function FRStdHex(
    ntype::AbstractNodeDistribution{RT},
    rtype::AbstractReconstruction{RT},
    nvars::Val{NV}=Val(1);
    nequispaced=nnodes(ntype),
) where {
    RT,
    NV,
}
    # Face and edge regions
    fstd = FRStdQuad(ntype, rtype, nvars; nequispaced=nequispaced)
    estd = fstd.face

    # Quadrature
    np = nnodes(ntype)
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
    I = Diagonal(ones(np))
    D = (
        kron(I, I, estd.D[1]),
        kron(I, estd.D[1], I),
        kron(estd.D[1], I, I),
    )
    Ds = (
        kron(I, I, estd.Ds[1]),
        kron(I, estd.Ds[1], I),
        kron(estd.Ds[1], I, I),
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
    cache = FRStdRegionCache{RT}(3, np, nvars)

    return FRStdHex{
        typeof(ntype),
        RT,
        typeof(fstd),
        typeof(estd),
        typeof(cache),
        typeof(rtype),
    }(
        ntype,
        fstd,
        estd,
        ξe,
        ξc,
        ξ,
        ω,
        D .|> transpose .|> sparse .|> transpose |> Tuple,
        Ds .|> transpose .|> sparse .|> transpose |> Tuple,
        D♯ .|> transpose .|> collect .|> transpose |> Tuple,
        l .|> transpose .|> sparse .|> transpose |> Tuple,
        ∂g .|> transpose .|> sparse .|> transpose |> Tuple,
        node2eq |> transpose |> collect |> transpose,
        cache,
        rtype,
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::FRStdHex)
    @nospecialize

    rt = eltype(s.D[1])
    qt = nodetype(s)
    np = nnodes(qt)

    nodes = "$(np)×$(np)×$(np)"
    rname = reconstruction_name(s.reconstruction)
    qname = if qt isa GaussNodes
        "Gauss"
    elseif qt isa GaussLobattoNodes
        "Gauss-Lobatto"
    else
        @assert false "[StdRegion.show] You shouldn't be here..."
    end
    print(io, "FRStdHex{", rt, "}: ")
    print(io, nodes, " ", qname, " node(s) with ", rname, " reconstruction")

    return nothing
end
