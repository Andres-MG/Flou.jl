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

struct StdSegment{RT,SN,SF,C,R} <: AbstractStdRegion{1}
    solbasis::SN
    basis::SF
    face::StdPoint{RT}
    ω::Vector{RT}
    ξ::Vector{SVector{1,RT}}
    ξe::Vector{SVector{1,RT}}
    ξc::Tuple{Vector{SVector{1,RT}}}
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

function StdSegment(
    solbasis::AbstractNodalBasis,
    rec::AbstractReconstruction,
    nvars=1,
    ftype=Float64;
    nequispaced=nnodes(solbasis),
)
    # Do all computations on the flux basis. This is the easiest way to implement
    # overintegration and is equivalent to the usual approaches when both bases are
    # the same.
    fbasis = basis(rec)

    # Face regions
    fstd = StdPoint(ftype)

    # Flux nodes
    np = nnodes(fbasis)
    _ξ, ω = convert.(ftype, fbasis.ξ), convert.(ftype, fbasis.ω)
    ξ = [SVector(ξi) for ξi in _ξ]

    # Equispaced nodes
    _ξe = nequispaced > 1 ? convert.(ftype, range(-1, 1, nequispaced)) : [zero(ftype)]
    ξe = [SVector(ξ) for ξ in _ξe]
    node2eq = interp_matrix(_ξe, fbasis)

    # Subgrid nodes
    ξc1 = Vector{SVector{1,ftype}}(undef, np + 1)
    ξc1[1] = SVector(-one(ftype))
    for i in 1:np
        ξc1[i + 1] = ξc1[i] .+ ω[i]
    end
    ξc2 = Vector{SVector{1,ftype}}(undef, np + 1)
    ξc2[np + 1] = SVector(one(ftype))
    for i in np:-1:1
        ξc2[i] = ξc2[i + 1] .- ω[i]
    end
    ξc = (ξc1 .+ ξc2) ./ 2
    ξc[1] = SVector(-one(ftype))
    ξc[np + 1] = SVector(one(ftype))
    ξc = tuple(ξc)

    # Projection to faces
    l = (
        interp_matrix([-one(ftype)], fbasis) |> vec,
        interp_matrix([+one(ftype)], fbasis) |> vec,
    )

    # Derivatives and reconstruction
    D = derivative_matrix(_ξ, fbasis)
    ∂g = reconstruction(rec, _ξ)
    ∂g = (-∂g[1], ∂g[2])

    B = ∂g[2] * l[2]' - ∂g[1] * l[1]'
    Ds = D - B
    D♯ = 2D - B

    # Projection to solution basis even if we keep using flux nodes
    if solbasis == fbasis
        Pover = diagm(ones(ftype, np))
    else
        Psf = interp_matrix(_ξ, solbasis)
        W = Diagonal(ω)
        Pover = Psf * (Psf' * W * Psf) \ (Psf' * W)
    end

    # Temporary storage
    cache = StdRegionCache{nvars}(1, np, ftype)

    return StdSegment(
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

function Base.show(io::IO, ::MIME"text/plain", s::StdSegment{RT}) where {RT}
    @nospecialize

    println(io, "StdSegment{", RT, "}: ")

    sbasis = s.solbasis
    np = nnodes(sbasis)
    bname = basisname(sbasis)
    nname = nodesname(sbasis)
    println(io, " Approximation: ", bname, " with ", np, " ", nname, " nodes")

    rname = s.reconstruction |> reconstruction_name
    print(io, " Flux reconstruction: ", rname)

    return nothing
end

ndirections(::StdSegment) = 1
nvertices(::StdSegment) = 2

function tpdofs(s::StdSegment, _)
    shape = (ndofs(s),)
    return (LinearIndices(shape),)
end

function tpdofs_subgrid(s::StdSegment, _)
    shape = (ndofs(s) + 1,)
    return (LinearIndices(shape),)
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

vtk_type(::StdSegment) = UInt8(68)

function vtk_connectivities(s::StdSegment)
    conns = [1, ndofs(s)]
    append!(conns, 2:(ndofs(s) - 1))
    return conns .- 1
end
