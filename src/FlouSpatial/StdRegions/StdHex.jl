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

struct StdHex{RT,SN,SF,F,E,C,R} <: AbstractStdRegion{3}
    solbasis::SN
    basis::SF
    face::F
    edge::E
    ω::Vector{RT}
    ξ::Vector{SVector{3,RT}}
    ξe::Vector{SVector{3,RT}}
    ξc::NTuple{3,Vector{SVector{3,RT}}}
    D::Matrix{RT}
    Ds::Matrix{RT}
    D♯::Matrix{RT}
    l::NTuple{2,Vector{RT}}
    ∂g::NTuple{2,Vector{RT}}
    Pover::Matrix{RT}
    node2eq::Matrix{RT}
    cache::C
    reconstruction::R
end

function StdHex(
    solbasis::AbstractNodalBasis,
    rec::AbstractReconstruction,
    nvars=1,
    ftype=Float64;
    nequispaced=nnodes(solbasis),
)
    # Face and edge regions
    fstd = StdQuad(solbasis, rec, nvars, ftype; nequispaced=nequispaced)
    estd = fstd.face
    fbasis = basis(rec)

    # Flux nodes
    np = nnodes(fbasis)
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

    # Subgrid nodes
    ξc1 = vec([
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in estd.ξc[1], ξy in estd.ξ, ξz in estd.ξ
    ])
    ξc2 = vec([
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in estd.ξ, ξy in estd.ξc[1], ξz in estd.ξ
    ])
    ξc3 = vec([
        SVector(ξx[1], ξy[1], ξz[1])
        for ξx in estd.ξ, ξy in estd.ξ, ξz in estd.ξc[1]
    ])
    ξc = (ξc1, ξc2, ξc3)

    # Projection operator
    l = estd.l

    # Derivatives and reconstruction
    D = estd.D
    Ds = estd.Ds
    D♯ = estd.D♯
    ∂g = estd.∂g

    # Projection to solution basis even if we keep using flux nodes
    Pover = kron(estd.Pover, estd.Pover, estd.Pover)

    # Temporary storage
    cache = StdRegionCache{nvars}(3, np, ftype)

    return StdHex(
        solbasis,
        fbasis,
        fstd,
        estd,
        ω,
        ξ,
        ξe,
        ξc,
        D,
        Ds,
        D♯,
        l,
        ∂g,
        Pover,
        node2eq,
        cache,
        rec,
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::StdHex{RT}) where {RT}
    @nospecialize

    println(io, "StdHex{", RT, "}: ")

    sbasis = s.solbasis
    np = nnodes(sbasis)
    nodes = "$(np)×$(np)×$(np)"
    bname = basisname(sbasis)
    nname = nodesname(sbasis)
    println(io, " Approximation: ", bname, " with ", nodes, " ", nname, " nodes")

    rname = s.reconstruction |> reconstruction_name
    print(io, " Flux reconstruction: ", rname)

    return nothing
end

ndirections(::StdHex) = 3
nvertices(::StdHex) = 8

function tpdofs(s::StdHex, ::Val{D}) where {D}
    shape = dofsize(s)
    indices = LinearIndices(shape)
    return if D == 1
        (view(indices, :, i, j) for i in 1:dofsize(s, 2), j in 1:dofsize(s, 3))
    elseif D == 2
        (view(indices, i, :, j) for i in 1:dofsize(s, 1), j in 1:dofsize(s, 3))
    else # D == 3
        (view(indices, i, j, :) for i in 1:dofsize(s, 1), j in 1:dofsize(s, 2))
    end
end

function tpdofs_subgrid(s::StdHex, ::Val{D}) where {D}
    shape = dofsize(s)
    return if D == 1
        shape = (shape[1] + 1, shape[2], shape[3])
        indices = LinearIndices(shape)
        (view(indices, :, i, j) for i in 1:dofsize(s, 2), j in 1:dofsize(s, 3))
    elseif D == 2
        shape = (shape[1], shape[2] + 1, shape[3])
        indices = LinearIndices(shape)
        (view(indices, i, :, j) for i in 1:dofsize(s, 1), j in 1:dofsize(s, 3))
    else # D == 3
        shape = (shape[1], shape[2], shape[3] + 1)
        indices = LinearIndices(shape)
        (view(indices, i, j, :) for i in 1:dofsize(s, 1), j in 1:dofsize(s, 2))
    end
end

function slave2master(_, _, _::StdHex)
    error("Not implemented!")
end

function master2slave(_, _, _::StdHex)
    error("Not implemented!")
end

vtk_type(::StdHex) = UInt8(72)

function vtk_connectivities(s::StdHex)
    n = ndofs(s, 1)
    li = lineardofs(s)
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
