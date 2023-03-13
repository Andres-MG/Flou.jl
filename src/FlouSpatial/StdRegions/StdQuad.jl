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

struct StdQuad{RT,SN,SF,F,C,R} <: AbstractStdRegion{2}
    solbasis::SN
    basis::SF
    face::F
    ω::Vector{RT}
    ξ::Vector{SVector{2,RT}}
    ξe::Vector{SVector{2,RT}}
    ξc::NTuple{2,Vector{SVector{2,RT}}}
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

function StdQuad(
    solbasis::AbstractNodalBasis,
    rec::AbstractReconstruction,
    nvars=1,
    ftype=Float64;
    nequispaced=nnodes(solbasis),
)
    # Face regions
    fstd = StdSegment(solbasis, rec, nvars, ftype; nequispaced=nequispaced)
    fbasis = basis(rec)

    # Flux nodes
    np = nnodes(fbasis)
    ξ = vec([SVector(ξx[1], ξy[1]) for ξx in fstd.ξ, ξy in fstd.ξ])
    ω = vec([ωx * ωy for ωx in fstd.ω, ωy in fstd.ω])

    # Equispaced nodes
    ξe = vec([SVector(ξx[1], ξy[1]) for ξx in fstd.ξe, ξy in fstd.ξe])
    node2eq = kron(fstd.node2eq, fstd.node2eq)

    # Subgrid nodes
    ξc1 = vec([SVector(ξx[1], ξy[1]) for ξx in fstd.ξc[1], ξy in fstd.ξ])
    ξc2 = vec([SVector(ξx[1], ξy[1]) for ξx in fstd.ξ, ξy in fstd.ξc[1]])
    ξc = (ξc1, ξc2)

    # Projection operator
    l = fstd.l

    # Derivatives and reconstruction
    D = fstd.D
    Ds = fstd.Ds
    D♯ = fstd.D♯
    ∂g = fstd.∂g

    # Projection to solution basis even if we keep using flux nodes
    Pover = kron(fstd.Pover, fstd.Pover)

    # Temporary storage
    cache = StdRegionCache{nvars}(2, np, ftype)

    return StdQuad(
        solbasis,
        fbasis,
        fstd,
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

function Base.show(io::IO, ::MIME"text/plain", s::StdQuad{RT}) where {RT}
    @nospecialize

    println(io, "StdQuad{", RT, "}: ")

    sbasis = s.solbasis
    np = nnodes(sbasis)
    nodes = "$(np)×$(np)"
    bname = basisname(sbasis)
    nname = nodesname(sbasis)
    println(io, " Approximation: ", bname, " with ", nodes, " ", nname, " nodes")

    rname = s.reconstruction |> reconstruction_name
    print(io, " Flux reconstruction: ", rname)

    return nothing
end

ndirections(::StdQuad) = 2
nvertices(::StdQuad) = 4

function tpdofs(s::StdQuad, ::Val{D}) where {D}
    shape = dofsize(s)
    indices = LinearIndices(shape)
    return if D == 1
        (view(indices, :, i) for i in 1:dofsize(s, 2))
    else # D == 2
        (view(indices, i, :) for i in 1:dofsize(s, 1))
    end
end

function tpdofs_subgrid(s::StdQuad, ::Val{D}) where {D}
    shape = dofsize(s)
    return if D == 1
        shape = (shape[1] + 1, shape[2])
        indices = LinearIndices(shape)
        (view(indices, :, i) for i in 1:dofsize(s, 2))
    else # D == 2
        shape = (shape[1], shape[2] + 1)
        indices = LinearIndices(shape)
        (view(indices, i, :) for i in 1:dofsize(s, 1))
    end
end

function slave2master(i::Integer, orientation, std::StdQuad)
    s = cartesiandofs(std)[i]
    li = lineardofs(std)
    return if orientation == 0
        i
    elseif orientation == 1
        li[ndofs(std, 1) - s[2] + 1, s[1]]
    elseif orientation == 2
        li[ndofs(std, 1) - s[1] + 1, ndofs(std, 2) - s[2] + 1]
    elseif orientation == 3
        li[s[2], ndofs(std, 2) - s[1] + 1]
    elseif orientation == 4
        li[s[2], s[1]]
    elseif orientation == 5
        li[ndofs(std, 1) - s[1] + 1, s[2]]
    elseif orientation == 6
        li[ndofs(std, 1) - s[2] + 1, ndofs(std, 2) - s[1] + 1]
    else # orientation == 7
        li[s[1], ndofs(std, 2) - s[2] + 1]
    end
end

function master2slave(i::Integer, orientation, std::StdQuad)
    m = cartesiandofs(std)[i]
    li = lineardofs(std)
    return if orientation == 0
        i
    elseif orientation == 1
        li[m[2], ndofs(std, 1) - m[1] + 1]
    elseif orientation == 2
        li[ndofs(std, 1) - m[1] + 1, ndofs(std, 2) - m[2] + 1]
    elseif orientation == 3
        li[ndofs(std, 2) - m[2] + 1, m[1]]
    elseif orientation == 4
        li[m[2], m[1]]
    elseif orientation == 5
        li[ndofs(std, 1) - m[1] + 1, m[2]]
    elseif orientation == 6
        li[ndofs(std, 2) - m[2] + 1, ndofs(std, 1) - m[1] + 1]
    else # orientation == 7
        li[m[1], ndofs(std, 2) - m[2] + 1]
    end
end

vtk_type(::StdQuad) = UInt8(70)

function vtk_connectivities(s::StdQuad)
    n = ndofs(s, 1)
    li = lineardofs(s)
    corners = [li[1, 1], li[n, 1], li[n, n], li[1, n]]
    edges = reduce(vcat, [
        li[2:(n - 1), 1], li[n, 2:(n - 1)],
        li[2:(n - 1), n], li[1, 2:(n - 1)],
    ])
    interior = vec(li[2:(n - 1), 2:(n - 1)])
    return mapreduce(x -> x .- 1, vcat, (corners, edges, interior))
end
