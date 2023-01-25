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

# Abstract spatial discretization for flux-reconstruction methods
abstract type AbstractFluxReconstruction{ND,RT} <: AbstractSpatialDiscretization{ND,RT} end

# Flux reconstruction functions, gₗ(x), gᵣ(x)
include("Reconstruction.jl")

# Standard regions
include("StdRegions.jl")

struct FR{ND,RT,MT<:AbstractMesh{ND,RT},ST,G,O,C,BCT,M,FT} <:
        AbstractFluxReconstruction{ND,RT}
    mesh::MT
    std::ST
    dofhandler::DofHandler
    geometry::G
    operators::O
    cache::C
    bcs::BCT
    massinv::M
    source!::FT
end

function FR(
    mesh::AbstractMesh{ND,RT},
    std,
    equation,
    operators,
    bcs,
    source=nothing,
) where {
    ND,
    RT,
}
    # Boundary conditions
    nbounds = nboundaries(mesh)
    length(bcs) == nbounds || throw(ArgumentError(
        "The number of BCs does not match the number of boundaries."
    ))
    _bcs = Vector{Any}(undef, nbounds)
    for (key, value) in bcs
        j = findfirst(==(key), mesh.bdnames)
        i = mesh.bdmap[j]
        _bcs[i] = value
    end
    _bcs = Tuple(_bcs)

    # DOF handler
    dofhandler = DofHandler(mesh, std)

    if operators isa AbstractOperator
        _operators = (operators,)
    else
        _operators = Tuple(operators)
    end

    # Physical elements
    subgrid = any(requires_subgrid.(_operators, Ref(std)))
    geometry = Geometry(std, dofhandler, mesh, subgrid)

    # Equation cache
    cache = construct_cache(:fr, RT, dofhandler, equation)

    # Mass matrix
    massinv = BlockDiagonal([Diagonal(elem.invjac) for elem in geometry.elements])

    # Source term
    sourceterm = if isnothing(source)
        (_, _, _, _) -> nothing
    else
        source
    end

    return FR(
        mesh,
        std,
        dofhandler,
        geometry,
        _operators,
        cache,
        _bcs,
        massinv,
        sourceterm,
    )
end

FlouCommon.nelements(fr::FR) = nelements(fr.dofhandler)
FlouCommon.nfaces(fr::FR) = nfaces(fr.dofhandler)
ndofs(fr::FR) = ndofs(fr.dofhandler)

FlouCommon.eachelement(fr::FR) = eachelement(fr.dofhandler)
FlouCommon.eachface(fr::FR) = eachface(fr.dofhandler)
eachdof(fr::FR) = eachdof(fr.dofhandler)

function Base.show(io::IO, m::MIME"text/plain", fr::FR)
    @nospecialize

    # Header
    println(io, "=========================================================================")
    println(io, "FR spatial discretization")

    # Mesh
    println(io, "\nMesh:")
    println(io, "-----")
    show(io, m, fr.mesh)

    # Standard regions
    println(io, "\n\nStandard region:")
    println(io, "----------------")
    show(io, m, fr.std)

    # Operators
    println(io, "\n\nOperators:")
    println(io, "----------")
    for op in fr.operators
        show(io, m, op)
        print(io, "\n")
    end

    print(io, "=========================================================================")

    return nothing
end

function project2faces!(Qf, Q, fr::FR)
    # Unpack
    (; mesh, std) = fr

    @flouthreads for ie in eachelement(fr)
        iface = mesh.elements[ie].faceinds
        facepos = mesh.elements[ie].facepos
        @inbounds for (s, (face, pos)) in enumerate(zip(iface, facepos))
            mul!(Qf.face[face][pos], std.l[s], Q.element[ie])
        end
    end
    return nothing
end

function apply_massmatrix!(dQ, fr::FR)
    @flouthreads for ie in eachelement(fr)
        lmul!(blocks(fr.massinv)[ie], dQ.element[ie])
    end
    return nothing
end

function FlouCommon.get_max_dt(q, fr::FR, eq::AbstractEquation, cfl)
    Q = StateVector{nvariables(eq)}(q, fr.dofhandler)
    Δt = typemax(eltype(Q))
    std = fr.std
    d = spatialdim(fr)
    n = ndofs(std)

    for ie in eachelement(fr)
        Δx = fr.geometry.elements[ie].volume[] / n
        Δx = d == 1 ? Δx : (d == 2 ? sqrt(Δx) : cbrt(Δx))
        @inbounds for i in eachdof(std)
            Δt = min(Δt, get_max_dt(Q.element[ie][i], Δx, cfl, eq))
        end
    end

    return Δt
end

# Operators
include("OpDivergence.jl")
include("OpGradient.jl")

# Equations
include("Hyperbolic.jl")
include("LinearAdvection.jl")
include("Burgers.jl")
include("KPP.jl")
include("Euler.jl")

include("Gradient.jl")
