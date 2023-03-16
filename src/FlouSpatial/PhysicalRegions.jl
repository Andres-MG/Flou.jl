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
#                                     Reference frame                                      #

struct ReferenceFrame{ND,RT}
    n::SVector{ND,RT}
    t::SVector{ND,RT}
    b::SVector{ND,RT}
end

#==========================================================================================#
#                                 Physical element subgrid                                 #

struct PhysicalSubgrid{F,J}
    frames::F
    jac::J
end

struct PhysicalSubgridVector{ND,RT} <: AbstractVector{PhysicalSubgrid}
    dh::DofHandler
    frames::Vector{NTuple{ND,Vector{ReferenceFrame{ND,RT}}}}
    jac::Vector{NTuple{ND,Vector{RT}}}
    function PhysicalSubgridVector(dh, frames, jac)
        length(frames) == length(jac) && length(frames) == nelements(dh) ||
            throw(DimensionMismatch(
                "There must be as many frames and Jacobians as elements in the DofHandler."
            ))
        nd = length(first(frames))
        rt = eltype(first(jac)[1])
        return new{nd,rt}(dh, frames, jac)
    end
end

Base.length(sv::PhysicalSubgridVector) = length(sv.frames)
Base.size(sv::PhysicalSubgridVector) = (length(sv),)

@inline function Base.getindex(sv::PhysicalSubgridVector, i::Integer)
    @boundscheck 1 <= i <= nelements(sv.dh) || throw(BoundsError(sv, i))
    return @inbounds PhysicalSubgrid(
        sv.frames[i],
        sv.jac[i],
    )
end

function PhysicalSubgridVector(ξc, dh::DofHandler, mesh::AbstractMesh)
    subgridvec = [PhysicalSubgrid(ξc, mesh, ie) for ie in eachelement(mesh)]
    return PhysicalSubgridVector(
        dh,
        [s.frames for s in subgridvec],
        [s.jac for s in subgridvec],
    )
end

function PhysicalSubgrid(ξc, ::CartesianMesh{1,RT}, _) where {RT}
    n = length(ξc[1])
    frames = ([
        ReferenceFrame(
            SVector(one(RT)),
            SVector(zero(RT)),
            SVector(zero(RT)),
        )
        for _ in 1:n
    ],)
    jac = (fill(one(RT), n),)
    return PhysicalSubgrid(frames, jac)
end

function PhysicalSubgrid(ξc, mesh::CartesianMesh{2,RT}, _) where {RT}
    (; Δx) = mesh
    n1 = length(ξc[1])
    n2 = length(ξc[2])
    frames = (
        [
            ReferenceFrame(
                SVector(one(RT), zero(RT)),
                SVector(zero(RT), one(RT)),
                SVector(zero(RT), zero(RT)),
            )
            for _ in 1:n1
        ],
        [
            ReferenceFrame(
                SVector(zero(RT), one(RT)),
                SVector(-one(RT), zero(RT)),
                SVector(zero(RT), zero(RT)),
            )
            for _ in 1:n2
        ],
    )
    jac = (
        [Δx[2] / 2 for _ in 1:n1],
        [Δx[1] / 2 for _ in 1:n2],
    )
    return PhysicalSubgrid(frames, jac)
end

function PhysicalSubgrid(ξc, mesh::CartesianMesh{3,RT}, _) where {RT}
    (; Δx) = mesh
    n1 = length(ξc[1])
    n2 = length(ξc[2])
    n3 = length(ξc[3])
    frames = (
        [
            ReferenceFrame(
                SVector(one(RT), zero(RT), zero(RT)),
                SVector(zero(RT), one(RT), zero(RT)),
                SVector(zero(RT), zero(RT), one(RT)),
            )
            for _ in 1:n1
        ],
        [
            ReferenceFrame(
                SVector(zero(RT), one(RT), zero(RT)),
                SVector(zero(RT), zero(RT), one(RT)),
                SVector(one(RT), zero(RT), zero(RT)),
            )
            for _ in 1:n2
        ],
        [
            ReferenceFrame(
                SVector(zero(RT), zero(RT), one(RT)),
                SVector(one(RT), zero(RT), zero(RT)),
                SVector(zero(RT), one(RT), zero(RT)),
            )
            for _ in 1:n3
        ],
    )
    jac = (
        [Δx[2] * Δx[3] / 4 for _ in 1:n1],
        [Δx[1] * Δx[3] / 4 for _ in 1:n2],
        [Δx[1] * Δx[2] / 4 for _ in 1:n3],
    )
    return PhysicalSubgrid(frames, jac)
end

function PhysicalSubgrid(ξc, mesh::StepMesh{RT}, ie) where {RT}
    (; Δx) = mesh
    ireg = region(mesh, ie)
    n1 = length(ξc[1])
    n2 = length(ξc[2])
    frames = (
        [
            ReferenceFrame(
                SVector(one(RT), zero(RT)),
                SVector(zero(RT), one(RT)),
                SVector(zero(RT), zero(RT)),
            )
            for _ in 1:n1
        ],
        [
            ReferenceFrame(
                SVector(zero(RT), one(RT)),
                SVector(-one(RT), zero(RT)),
                SVector(zero(RT), zero(RT)),
            )
            for _ in 1:n2
        ],
    )
    jac = (
        [Δx[ireg][2] / 2 for _ in 1:n1],
        [Δx[ireg][1] / 2 for _ in 1:n2],
    )
    return PhysicalSubgrid(frames, jac)
end

function PhysicalSubgrid(ξc, mesh::UnstructuredMesh{1,RT}, ie) where {RT}
    n = length(ξc[1])
    frames = (Vector{ReferenceFrame{1,RT}}(undef, n),)
    jac = (Vector{RT}(undef, n),)
    for i in 1:n
        ξ = ξc[1][i]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        s = map_jacobian(main, mesh, ie) |> sign
        frames[1][i] = ReferenceFrame(
            SVector(s * dual[1]),
            SVector(zero(RT)),
            SVector(zero(RT)),
        )
        jac[1][i] = one(RT)
    end
    return PhysicalSubgrid(frames, jac)
end

function PhysicalSubgrid(ξc, mesh::UnstructuredMesh{2,RT}, ie) where {RT}
    n1 = length(ξc[1])
    n2 = length(ξc[2])
    frames = (
        Vector{ReferenceFrame{2,RT}}(undef, n1),
        Vector{ReferenceFrame{2,RT}}(undef, n2),
    )
    jac = (Vector{RT}(undef, n1), Vector{RT}(undef, n2))

    # Vertical faces
    for i in 1:n1
        ξ = ξc[1][i]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        s = map_jacobian(main, mesh, ie) |> sign
        n = s * dual[1]
        t = s * normalize(main[2])
        b = SVector(zero(RT), zero(RT))
        jac[1][i] = norm(n)
        n /= jac[1][i]
        frames[1][i] = ReferenceFrame(n, t, b)
    end

    # Horizontal faces
    for i in 1:n2
        ξ = ξc[2][i]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        s = map_jacobian(main, mesh, ie) |> sign
        n = s * dual[2]
        t = -s * normalize(main[1])
        b = SVector(zero(RT), zero(RT))
        jac[2][i] = norm(n)
        n /= jac[2][i]
        frames[2][i] = ReferenceFrame(n, t, b)
    end
    return PhysicalSubgrid(frames, jac)
end

function PhysicalSubgrid(ξc, mesh::UnstructuredMesh{3,RT}, ie) where {RT}
    n1 = length(ξc[1])
    n2 = length(ξc[2])
    n3 = length(ξc[3])
    frames = (
        Vector{ReferenceFrame{3,RT}}(undef, n1),
        Vector{ReferenceFrame{3,RT}}(undef, n2),
        Vector{ReferenceFrame{3,RT}}(undef, n3),
    )
    jac = (
        Vector{RT}(undef, n1),
        Vector{RT}(undef, n2),
        Vector{RT}(undef, n3),
    )

    # X faces
    for i in 1:n1
        ξ = ξc[1][i]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        n = dual[1]
        t = normalize(main[2])
        b = normalize(cross(n, t))
        jac[1][i] = norm(n)
        n /= jac[1][i]
        frames[1][i] = ReferenceFrame(n, t, b)
    end

    # Y faces
    for i in 1:n2
        ξ = ξc[2][i]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        n = dual[2]
        t = normalize(main[3])
        b = normalize(cross(n, t))
        jac[2][i] = norm(n)
        n /= jac[2][i]
        frames[2][i] = ReferenceFrame(n, t, b)
    end

    # Z faces
    for i in 1:n3
        ξ = ξc[3][i]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        n = dual[3]
        t = normalize(main[1])
        b = normalize(cross(n, t))
        jac[3][i] = norm(n)
        n /= jac[3][i]
        frames[3][i] = ReferenceFrame(n, t, b)
    end
    return PhysicalSubgrid(frames, jac)
end

#==========================================================================================#
#                                     Physical element                                     #

struct PhysicalElement{C,J,D,V}
    coords::C
    jac::J
    invjac::J
    Jω::J
    metric::D
    volume::V
end

struct PhysicalElementVector{ND,RT,ND2} <: AbstractVector{PhysicalElement}
    dh::DofHandler
    coords::Vector{SVector{ND,RT}}
    jac::Vector{RT}
    invjac::Vector{RT}
    Jω::Vector{RT}
    metric::Vector{SMatrix{ND,ND,RT,ND2}}
    volume::Vector{RT}
end

Base.length(ev::PhysicalElementVector) = length(ev.volume)
Base.size(ev::PhysicalElementVector) = (length(ev),)

@inline function Base.getindex(ev::PhysicalElementVector, i::Integer)
    @boundscheck 1 <= i <= nelements(ev.dh) || throw(BoundsError(ev, i))
    @inbounds begin
        i1 = ev.dh.elem_offsets[i] + 1
        i2 = ev.dh.elem_offsets[i + 1]
        return PhysicalElement(
            view(ev.coords, i1:i2),
            view(ev.jac, i1:i2),
            view(ev.invjac, i1:i2),
            view(ev.Jω, i1:i2),
            view(ev.metric, i1:i2),
            ev.volume[i],
        )
    end
end

function PhysicalElementVector(
    ξ,
    ω,
    dh::DofHandler,
    mesh::AbstractMesh{ND,RT},
) where {
    ND,
    RT,
}
    coords = SVector{ND,RT}[]
    jac = RT[]
    invjac = RT[]
    Jω = RT[]
    metric = []
    volume = RT[]

    for ie in eachelement(mesh)
        elem = PhysicalElement(ξ, ω, mesh, ie)
        append!(coords, elem.coords)
        append!(jac, elem.jac)
        append!(invjac, elem.invjac)
        append!(Jω, elem.Jω)
        append!(metric, elem.metric)
        push!(volume, elem.volume)
    end

    metric = [metric...]

    return PhysicalElementVector(dh, coords, jac, invjac, Jω, metric, volume)
end

function integrate(f::AbstractVector, elem::PhysicalElement)
    return elem.Jω' * f
end

function PhysicalElement(ξ, ω, mesh::CartesianMesh{ND,RT}, ie) where {ND,RT}
    (; Δx) = mesh

    # Coordinates
    np = length(ξ)
    xyz = [phys_coords(ξ, mesh, ie) for ξ in ξ]

    # Jacobian and dual basis
    jac = fill(prod(Δx) / (2^ND), np)
    invjac = inv.(jac)
    metric = if ND == 1
        fill(SMatrix{1,1}(one(RT)), np)
    elseif ND == 2
        fill(
            SMatrix{2,2}(
                Δx[2] / 2, 0,
                0, Δx[1] / 2,
            ),
            np,
        )
    else # ND == 3
        fill(
            SMatrix{3,3}(
                Δx[2] * Δx[3] / 4, 0, 0,
                0, Δx[1] * Δx[3] / 4, 0,
                0, 0, Δx[1] * Δx[2] / 4,
            ),
            np,
        )
    end

    # Integration weights
    Jω = jac .* ω

    # Volume
    vol = sum(Jω)

    return PhysicalElement(xyz, jac, invjac, Jω, metric, vol)
end

function PhysicalElement(ξ, ω, mesh::StepMesh{RT}, ie) where {RT}
    (; Δx) = mesh
    ireg = region(mesh, ie)

    # Coordinates
    np = length(ξ)
    xyz = [phys_coords(ξ, mesh, ie) for ξ in ξ]

    # Jacobian and dual basis
    jac = fill(prod(Δx[ireg]) / 4, np)
    invjac = inv.(jac)
    metric = fill(
        SMatrix{2,2}(
            Δx[ireg][2] / 2, 0,
            0, Δx[ireg][1] / 2,
        ),
        np,
    )

    # Integration weights
    Jω = jac .* ω

    # Volume
    vol = sum(Jω)

    return PhysicalElement(xyz, jac, invjac, Jω, metric, vol)
end

function PhysicalElement(ξ, ω, mesh::UnstructuredMesh{ND,RT}, ie) where {ND,RT}
    # Coordinates
    np = length(ξ)
    xyz = [phys_coords(ξ, mesh, ie) for ξ in ξ]

    # Jacobian and dual basis
    jac = Vector{RT}(undef, np)
    metric = Vector{SMatrix{ND,ND,RT,ND * ND}}(undef, np)
    for i in eachindex(ξ)
        main = map_basis(ξ[i], mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        jac[i] = map_jacobian(main, mesh, ie)
        if ND == 1
            metric[i] = SMatrix{1,1}(dual[1])
            jac[i] = abs(jac[i])
        elseif ND == 2
            metric[i] = SMatrix{2,2}(vcat(dual[1], dual[2]))
            jac[i] = abs(jac[i])
        else # ND == 3
            metric[i] = SMatrix{3,3}(vcat(dual[1], dual[2], dual[3]))
            jac[i] > 0 || throw(DomainError(
                jac[i], "Found a negative Jacobian in element $(ie)."
            ))
        end
    end
    invjac = inv.(jac)

    # Integration weights
    Jω = jac .* ω

    # Volume
    vol = sum(Jω)

    return PhysicalElement(xyz, jac, invjac, Jω, metric, vol)
end

#==========================================================================================#
#                                       Physical face                                      #

struct PhysicalFace{C,F,J,S}
    coords::C
    frames::F
    jac::J
    Jω::J
    surface::S
end

struct PhysicalFaceVector{ND,RT} <: AbstractVector{PhysicalFace}
    dh::DofHandler
    coords::Vector{SVector{ND,RT}}
    frames::Vector{ReferenceFrame{ND,RT}}
    jac::Vector{RT}
    Jω::Vector{RT}
    surface::Vector{RT}
end

Base.length(fv::PhysicalFaceVector) = length(fv.surface)
Base.size(fv::PhysicalFaceVector) = (length(fv),)

@inline function Base.getindex(fv::PhysicalFaceVector, i::Integer)
    @boundscheck 1 <= i <= nfaces(fv.dh) || throw(BoundsError(fv, i))
    @inline begin
        i1 = fv.dh.face_offsets[i] + 1
        i2 = fv.dh.face_offsets[i+1]
        return PhysicalFace(
            view(fv.coords, i1:i2),
            view(fv.frames, i1:i2),
            view(fv.jac, i1:i2),
            view(fv.Jω, i1:i2),
            fv.surface[i],
        )
    end
end

function PhysicalFaceVector(ξf, ω, dh, mesh)
    coords = []
    frames = []
    jac = []
    Jω = []
    surface = []

    for iface in eachface(mesh)
        face = PhysicalFace(ξf, ω, mesh, iface)
        append!(coords, face.coords)
        append!(frames, face.frames)
        append!(jac, face.jac)
        append!(Jω, face.Jω)
        push!(surface, face.surface)
    end

    coords = [coords...]
    frames = [frames...]
    jac = [jac...]
    Jω = [Jω...]
    surface = [surface...]

    return PhysicalFaceVector(dh, coords, frames, jac, Jω, surface)
end

function integrate(f::AbstractVector, face::PhysicalFace)
    return face.Jω' * f
end

function PhysicalFace(_, _, mesh::CartesianMesh{1,RT}, iface) where {RT}
    pos = mesh.faces[iface].elempos[1]
    nind = mesh.faces[iface].nodeinds[1]
    x = [mesh.nodes[nind]]
    jac = [one(RT)]
    if pos == 1
        frames = [
            ReferenceFrame(
                SVector(-one(RT)),
                SVector(zero(RT)),
                SVector(zero(RT)),
            )
        ]
    else # pos == 2
        frames = [
            ReferenceFrame(
                SVector(one(RT)),
                SVector(zero(RT)),
                SVector(zero(RT)),
            )
        ]
    end
    Jω = [zero(RT)]
    surf = one(RT)
    return PhysicalFace(x, frames, jac, Jω, surf)
end

function PhysicalFace(ξf, ω, mesh::CartesianMesh{2,RT}, iface) where {RT}
    (; Δx) = mesh
    np = length(ξf)
    pos = mesh.faces[iface].elempos[1]
    if pos == 1 || pos == 2  # Vertical
        jac = fill(Δx[2] / 2, np)
        if pos == 1
            frames = [
                ReferenceFrame(
                    SVector(-one(RT), zero(RT)),
                    SVector(zero(RT), -one(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:np
            ]
        else # pos == 2
            frames = [
                ReferenceFrame(
                    SVector(one(RT), zero(RT)),
                    SVector(zero(RT), one(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:np
            ]
        end

    else  # pos == 3 || pos == 4  # Horizontal
        jac = fill(Δx[1] / 2, np)
        if pos == 3
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), -one(RT)),
                    SVector(one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:np
            ]
        else # pos == 4
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), one(RT)),
                    SVector(-one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:np
            ]
        end
    end

    xy = [face_phys_coords(ξ, mesh, iface) for ξ in ξf]
    Jω = jac .* ω
    surf = sum(Jω)
    return PhysicalFace(xy, frames, jac, Jω, surf)
end

function PhysicalFace(ξf, ω, mesh::CartesianMesh{3,RT}, iface) where {RT}
    (; Δx) = mesh
    np = length(ξf)
    pos = mesh.faces[iface].elempos[1]
    if pos == 1 || pos == 2  # X faces
        jac = fill(Δx[2] * Δx[3] / 4, np)
        if pos == 1
            frames = [
                ReferenceFrame(
                    SVector(-one(RT), zero(RT), zero(RT)),
                    SVector(zero(RT), -one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT), one(RT)),
                )
                for _ in 1:np
            ]
        else # pos == 2
            frames = [
                ReferenceFrame(
                    SVector(one(RT), zero(RT), zero(RT)),
                    SVector(zero(RT), one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT), one(RT)),
                )
                for _ in 1:np
            ]
        end
    elseif pos == 3 || pos == 4  # Y faces
        jac = fill(Δx[1] * Δx[3] / 4, np)
        if pos == 3
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), -one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT), -one(RT)),
                    SVector(one(RT), zero(RT), zero(RT)),
                )
                for _ in 1:np
            ]
        else # pos == 4
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT), one(RT)),
                    SVector(one(RT), zero(RT), zero(RT)),
                )
                for _ in 1:np
            ]
        end
    else # pos == 5 || pos == 6  # Z faces
        jac = fill(Δx[1] * Δx[2] / 4, np)
        if pos == 5
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), zero(RT), -one(RT)),
                    SVector(-one(RT), zero(RT), zero(RT)),
                    SVector(zero(RT), one(RT), zero(RT)),
                )
                for _ in 1:np
            ]
        else # pos == 6
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), zero(RT), one(RT)),
                    SVector(one(RT), zero(RT), zero(RT)),
                    SVector(zero(RT), one(RT), zero(RT)),
                )
                for _ in 1:np
            ]
        end
    end

    xy = [face_phys_coords(ξ, mesh, iface) for ξ in ξf]
    Jω = jac .* ω
    surf = sum(Jω)
    return PhysicalFace(xy, frames, jac, Jω, surf)
end

function PhysicalFace(ξf, ω, mesh::StepMesh{RT}, iface) where {RT}
    (; Δx) = mesh
    np = length(ξf)
    ie = mesh.faces[iface].eleminds[1]
    ireg = region(mesh, ie)
    pos = mesh.faces[iface].elempos[1]

    if pos == 1 || pos == 2  # Vertical
        jac = fill(Δx[ireg][2] / 2, np)
        if pos == 1
            frames = [
                ReferenceFrame(
                    SVector(-one(RT), zero(RT)),
                    SVector(zero(RT), -one(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:np
            ]
        else # pos == 2
            frames = [
                ReferenceFrame(
                    SVector(one(RT), zero(RT)),
                    SVector(zero(RT), one(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:np
            ]
        end

    else  # pos == 3 || pos == 4  # Horizontal
        jac = fill(Δx[ireg][1] / 2, ndofs(fstd))
        if pos == 3
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), -one(RT)),
                    SVector(one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:np
            ]
        else # pos == 4
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), one(RT)),
                    SVector(-one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:np
            ]
        end
    end

    xy = [face_phys_coords(ξf, mesh, iface) for ξ in ξf]
    Jω = jac .* ω
    surf = sum(Jω)
    return PhysicalFace(xy, frames, jac, Jω, surf)
end

function PhysicalFace(_, _, mesh::UnstructuredMesh{1,RT}, iface) where {RT}
    face = mesh.faces[iface]
    ielem = face.eleminds[1]
    pos = face.elempos[1]

    x = Vector{SVector{1,RT}}(undef, 1)
    frames = Vector{ReferenceFrame{1,RT}}(undef, 1)
    jac = Vector{RT}(undef, 1)

    if pos == 1
        ξ = SVector(-one(RT))
        x[1] = phys_coords(ξ, mesh, ielem)
        main = map_basis(ξ, mesh, ielem)
        dual = map_dual_basis(main, mesh, ielem)
        s = map_jacobian(main, mesh, ielem) |> sign
        n = -dual[1] * s
        t = SVector(zero(RT))
        b = SVector(zero(RT))
        jac[1] = norm(n)
        n /= jac[1]
        frames[1] = ReferenceFrame(n, t, b)

    else # pos == 2
        ξ = SVector(one(RT))
        x[1] = phys_coords(ξ, mesh, ielem)
        main = map_basis(ξ, mesh, ielem)
        dual = map_dual_basis(main, mesh, ielem)
        s = map_jacobian(main, mesh, ielem) |> sign
        n = dual[1] * s
        t = SVector(zero(RT))
        b = SVector(zero(RT))
        jac[1] = norm(n)
        n /= jac[1]
        frames[1] = ReferenceFrame(n, t, b)
    end

    Jω = [zero(RT)]
    surf = one(RT)
    return PhysicalFace(x, frames, jac, Jω, surf)
end

function PhysicalFace(ξf, ω, mesh::UnstructuredMesh{2,RT}, iface) where {RT}
    np = length(ξf)
    face = mesh.faces[iface]
    ielem = face.eleminds[1]
    pos = face.elempos[1]

    xy = Vector{SVector{2,RT}}(undef, np)
    frames = Vector{ReferenceFrame{2,RT}}(undef, np)
    jac = Vector{RT}(undef, np)

    if pos == 1  # Left
        for i in eachindex(ξf)
            ξ = SVector(-one(RT), ξf[i][1])
            xy[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            s = map_jacobian(main, mesh, ielem) |> sign
            n = -dual[1] * s
            t = -normalize(main[2]) * s
            b = SVector(zero(RT), zero(RT))
            jac[i] = norm(n)
            n /= jac[i]
            frames[i] = ReferenceFrame(n, t, b)
        end

    elseif pos == 2  # Right
        for i in eachindex(ξf)
            ξ = SVector(one(RT), ξf[i][1])
            xy[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            s = map_jacobian(main, mesh, ielem) |> sign
            n = dual[1] * s
            t = normalize(main[2]) * s
            b = SVector(zero(RT), zero(RT))
            jac[i] = norm(n)
            n /= jac[i]
            frames[i] = ReferenceFrame(n, t, b)
        end

    elseif pos == 3  # Bottom
        for i in eachindex(ξf)
            ξ = SVector(ξf[i][1], -one(RT))
            xy[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            s = map_jacobian(main, mesh, ielem) |> sign
            n = -dual[2] * s
            t = normalize(main[1]) * s
            b = SVector(zero(RT), zero(RT))
            jac[i] = norm(n)
            n /= jac[i]
            frames[i] = ReferenceFrame(n, t, b)
        end

    else # pos == 4  # Top
        for i in eachindex(ξf)
            ξ = SVector(ξf[i][1], one(RT))
            xy[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            s = map_jacobian(main, mesh, ielem) |> sign
            n = dual[2] * s
            t = -normalize(main[1]) * s
            b = SVector(zero(RT), zero(RT))
            jac[i] = norm(n)
            n /= jac[i]
            frames[i] = ReferenceFrame(n, t, b)
        end
    end

    Jω = jac .* ω
    surf = sum(Jω)
    return PhysicalFace(xy, frames, jac, Jω, surf)
end

function PhysicalFace(ξf, ω, mesh::UnstructuredMesh{3,RT}, iface) where {RT}
    np = length(ξf)
    face = mesh.faces[iface]
    ielem = face.eleminds[1]
    pos = face.elempos[1]

    xyz = Vector{SVector{3,RT}}(undef, np)
    frames = Vector{ReferenceFrame{3,RT}}(undef, np)
    jac = Vector{RT}(undef, np)

    if pos == 1  # -X face
        for i in eachindex(ξf)
            ξ = SVector(-one(RT), ξf[i][1], ξf[i][2])
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n = -dual[1]
            t = -normalize(main[2])
            b = normalize(cross(n, t))
            jac[i] = norm(n)
            n /= jac[i]
            frames[i] = ReferenceFrame(n, t, b)
        end

    elseif pos == 2  # +X face
        for i in eachindex(ξf)
            ξ = SVector(one(RT), ξf[i][1], ξf[i][2])
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n = dual[1]
            t = normalize(main[2])
            b = normalize(cross(n, t))
            jac[i] = norm(n)
            n /= jac[i]
            frames[i] = ReferenceFrame(n, t, b)
        end

    elseif pos == 3  # -Y face
        for i in eachindex(ξf)
            ξ = SVector(ξf[i][1], -one(RT), ξf[i][2])
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n = -dual[2]
            t = -normalize(main[3])
            b = normalize(cross(n, t))
            jac[i] = norm(n)
            n /= jac[i]
            frames[i] = ReferenceFrame(n, t, b)
        end

    elseif pos == 4  # +Y face
        for i in eachindex(ξf)
            ξ = SVector(ξf[i][1], one(RT), ξf[i][2])
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n = dual[2]
            t = normalize(main[3])
            b = normalize(cross(n, t))
            jac[i] = norm(n)
            n /= jac[i]
            frames[i] = ReferenceFrame(n, t, b)
        end

    elseif pos == 5  # -Z face
        for i in eachindex(ξf)
            ξ = SVector(ξf[i][1], ξf[i][2], -one(RT))
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n = -dual[3]
            t = -normalize(main[1])
            b = normalize(cross(n, t))
            jac[i] = norm(n)
            n /= jac[i]
            frames[i] = ReferenceFrame(n, t, b)
        end

    else # pos == 6  # +Z face
        for i in eachindex(ξf)
            ξ = SVector(ξf[i][1], ξf[i][2], one(RT))
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n = dual[3]
            t = normalize(main[1])
            b = normalize(cross(n, t))
            jac[i] = norm(n)
            n /= jac[i]
            frames[i] = ReferenceFrame(n, t, b)
        end
    end

    Jω = jac .* ω
    surf = sum(Jω)
    return PhysicalFace(xyz, frames, jac, Jω, surf)
end

#==========================================================================================#
#                                         Geometry                                         #

struct Geometry{PE,PF,SG}
    elements::PE
    faces::PF
    subgrids::SG
end

function Geometry(std, dh::DofHandler, mesh::AbstractMesh, subgrid=false)
    elemvec = PhysicalElementVector(std.ξ, std.ω, dh, mesh)
    facevec = PhysicalFaceVector(std.face.ξ, std.face.ω, dh, mesh)
    subgridvec = if subgrid
        PhysicalSubgridVector(std.ξc, dh, mesh)
    else
        nothing
    end
    return Geometry(elemvec, facevec, subgridvec)
end

FlouCommon.nelements(v::PhysicalSubgridVector) = nelements(v.dh)
FlouCommon.nelements(v::PhysicalElementVector) = nelements(v.dh)
FlouCommon.nelements(g::Geometry) = nelements(g.elements)
FlouCommon.nfaces(v::PhysicalFaceVector) = nfaces(v.dh)
FlouCommon.nfaces(g::Geometry) = nfaces(g.faces)
ndofs(v::PhysicalElementVector) = ndofs(v.dh)
ndofs(g::Geometry) = ndofs(g.elements)

@inline function contravariant(F, metric::SMatrix{1,1}, dir)
    @boundscheck checkbounds(metric, dir, dir)
    return F[1] * metric[1, 1]
end

function contravariant(F, metric::SMatrix{1,1})
    return @inbounds (contravariant(F, metric, 1),)
end

@inline function contravariant(F, metric::SMatrix{2,2}, dir)
    @boundscheck checkbounds(metric, dir, dir)
    return F[1] * metric[1, dir] + F[2] * metric[2, dir]
end

function contravariant(F, metric::SMatrix{2,2})
    return @inbounds (
        contravariant(F, metric, 1),
        contravariant(F, metric, 2),
    )
end

@inline function contravariant(F, metric::SMatrix{3,3}, dir)
    @boundscheck checkbounds(metric, dir, dir)
    return F[1] * metric[1, dir] + F[2] * metric[2, dir] + F[3] * metric[3, dir]
end

function contravariant(F, metric::SMatrix{3,3})
    return @inbounds (
        contravariant(F, metric, 1),
        contravariant(F, metric, 2),
        contravariant(F, metric, 3),
    )
end