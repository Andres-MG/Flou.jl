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
    frames::Vector{NTuple{ND,Array{ReferenceFrame{ND,RT},ND}}}
    jac::Vector{NTuple{ND,Array{RT,ND}}}
    function PhysicalSubgridVector(dh, frames, jac)
        length(frames) == length(jac) && length(frames) == nelements(dh) ||
            throw(DimensionMismatch(
                "There must be as many frames and Jacobians as elements in the DofHandler."
            ))
        nd = length(frames[1])
        rt = eltype(jac[1][1])
        return new{nd,rt}(dh, frames, jac)
    end
end

Base.length(sv::PhysicalSubgridVector) = length(sv.frames)
Base.size(sv::PhysicalSubgridVector) = (length(sv),)

@inline function Base.getindex(sv::PhysicalSubgridVector, i)
    @boundscheck 1 <= i <= nelements(sv.dh) || throw(BoundsError(sv, i))
    return @inbounds PhysicalSubgrid(
        view(sv.frames, i),
        view(sv.jac, i),
    )
end

function PhysicalSubgridVector(std, dh::DofHandler, mesh::AbstractMesh)
    subgridvec = [PhysicalSubgrid(std, mesh, ie) for ie in eachelement(mesh)]
    return PhysicalSubgridVector(
        dh,
        [s.frames for s in subgridvec],
        [s.jac for s in subgridvec],
    )
end

function PhysicalSubgrid(std, ::CartesianMesh{1,RT}, _) where {RT}
    nx = ndofs(std) + 1
    frames = ([
        ReferenceFrame(
            SVector(one(RT)),
            SVector(zero(RT)),
            SVector(zero(RT)),
        )
        for _ in 1:nx
    ],)
    jac = (fill(one(RT), nx),)
    return PhysicalSubgrid(frames, jac)
end

function PhysicalSubgrid(std, mesh::CartesianMesh{2,RT}, _) where {RT}
    (; Δx) = mesh
    nx, ny = dofsize(std)
    frames = (
        [
            ReferenceFrame(
                SVector(one(RT), zero(RT)),
                SVector(zero(RT), one(RT)),
                SVector(zero(RT), zero(RT)),
            )
            for _ in 1:(nx+1), _ in 1:ny
        ],
        [
            ReferenceFrame(
                SVector(zero(RT), one(RT)),
                SVector(-one(RT), zero(RT)),
                SVector(zero(RT), zero(RT)),
            )
            for _ in 1:nx, _ in 1:(ny+1)
        ],
    )
    jac = (
        fill(Δx[2] / 2, nx + 1, ny),
        fill(Δx[1] / 2, nx, ny + 1),
    )
    return PhysicalSubgrid(frames, jac)
end

function PhysicalSubgrid(std, mesh::CartesianMesh{3,RT}, _) where {RT}
    (; Δx) = mesh
    nx, ny, nz = dofsize(std)
    frames = (
        [
            ReferenceFrame(
                SVector(one(RT), zero(RT), zero(RT)),
                SVector(zero(RT), one(RT), zero(RT)),
                SVector(zero(RT), zero(RT), one(RT)),
            )
            for _ in 1:(nx+1), _ in 1:ny, _ in 1:nz
        ],
        [
            ReferenceFrame(
                SVector(zero(RT), one(RT), zero(RT)),
                SVector(zero(RT), zero(RT), one(RT)),
                SVector(one(RT), zero(RT), zero(RT)),
            )
            for _ in 1:nx, _ in 1:(ny+1), _ in 1:nz
        ],
        [
            ReferenceFrame(
                SVector(zero(RT), zero(RT), one(RT)),
                SVector(one(RT), zero(RT), zero(RT)),
                SVector(zero(RT), one(RT), zero(RT)),
            )
            for _ in 1:nx, _ in 1:ny, _ in 1:(nz+1)
        ],
    )
    jac = (
        fill(Δx[2] * Δx[3] / 4, nx + 1, ny, nz),
        fill(Δx[1] * Δx[3] / 4, nx, ny + 1, nz),
        fill(Δx[1] * Δx[2] / 4, nx, ny, nz + 1),
    )
    return PhysicalSubgrid(frames, jac)
end

function PhysicalSubgrid(std, mesh::StepMesh{RT}, ie) where {RT}
    (; Δx) = mesh
    ireg = region(mesh, ie)
    nx, ny = dofsize(std)
    frames = (
        [
            ReferenceFrame(
                SVector(one(RT), zero(RT)),
                SVector(zero(RT), one(RT)),
                SVector(zero(RT), zero(RT)),
            )
            for _ in 1:(nx+1), _ in 1:ny
        ],
        [
            ReferenceFrame(
                SVector(zero(RT), one(RT)),
                SVector(-one(RT), zero(RT)),
                SVector(zero(RT), zero(RT)),
            )
            for _ in 1:nx, _ in 1:(ny+1)
        ],
    )
    jac = (
        fill(Δx[ireg][2] / 2, nx + 1, ny),
        fill(Δx[ireg][1] / 2, nx, ny + 1),
    )
    return PhysicalSubgrid(frames, jac)
end

function PhysicalSubgrid(std, mesh::UnstructuredMesh{1,RT}, ie) where {RT}
    nx = ndofs(std)
    frames = (Vector{ReferenceFrame{1,RT}}(undef, nx + 1),)
    jac = (Vector{RT}(undef, nx + 1),)
    for i in 1:(nx+1)
        ξ = std.ξc[1][i]
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

function PhysicalSubgrid(std, mesh::UnstructuredMesh{2,RT}, ie) where {RT}
    nx, ny = dofsize(std)
    frames = (
        Matrix{ReferenceFrame{2,RT}}(undef, nx + 1, ny),
        Matrix{ReferenceFrame{2,RT}}(undef, nx, ny + 1),
    )
    jac = (Matrix{RT}(undef, nx + 1, ny), Matrix{RT}(undef, nx, ny + 1))
    for j in 1:ny, i in 1:(nx+1)  # Vertical faces
        ξ = std.ξc[1][i, j]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        s = map_jacobian(main, mesh, ie) |> sign
        n = s * dual[1]
        t = s * normalize(main[2])
        b = SVector(zero(RT), zero(RT))
        jac[1][i, j] = norm(n)
        n /= jac[1][i, j]
        frames[1][i, j] = ReferenceFrame(n, t, b)
    end
    for j in 1:(ny+1), i in 1:nx  # Horizontal faces
        ξ = std.ξc[2][i, j]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        s = map_jacobian(main, mesh, ie) |> sign
        n = s * dual[2]
        t = -s * normalize(main[1])
        b = SVector(zero(RT), zero(RT))
        jac[2][i, j] = norm(n)
        n /= jac[2][i, j]
        frames[2][i, j] = ReferenceFrame(n, t, b)
    end
    return PhysicalSubgrid(frames, jac)
end

function PhysicalSubgrid(std, mesh::UnstructuredMesh{3,RT}, ie) where {RT}
    nx, ny, nz = dofsize(std)
    frames = (
        Array{ReferenceFrame{3,RT},3}(undef, nx + 1, ny, nz),
        Array{ReferenceFrame{3,RT},3}(undef, nx, ny + 1, nz),
        Array{ReferenceFrame{3,RT},3}(undef, nx, ny, nz + 1),
    )
    jac = (
        Array{RT,3}(undef, nx + 1, ny, nz),
        Array{RT,3}(undef, nx, ny + 1, nz),
        Array{RT,3}(undef, nx, ny, nz + 1),
    )
    for k in 1:nz, j in 1:ny, i in 1:(nx+1)  # X faces
        ξ = std.ξc[1][i, j, k]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        n = dual[1]
        t = normalize(main[2])
        b = normalize(cross(n, t))
        jac[1][i, j, k] = norm(n)
        n /= jac[1][i, j, k]
        frames[1][i, j, k] = ReferenceFrame(n, t, b)
    end
    for k in 1:nz, j in 1:(ny+1), i in 1:nx  # Y faces
        ξ = std.ξc[2][i, j, k]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        n = dual[2]
        t = normalize(main[3])
        b = normalize(cross(n, t))
        jac[2][i, j, k] = norm(n)
        n /= jac[2][i, j, k]
        frames[2][i, j, k] = ReferenceFrame(n, t, b)
    end
    for k in 1:(nz+1), j in 1:ny, i in 1:nx  # Z faces
        ξ = std.ξc[3][i, j, k]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        n = dual[3]
        t = normalize(main[1])
        b = normalize(cross(n, t))
        jac[3][i, j, k] = norm(n)
        n /= jac[3][i, j, k]
        frames[3][i, j, k] = ReferenceFrame(n, t, b)
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
        i2 = ev.dh.elem_offsets[i+1]
        return PhysicalElement(
            view(ev.coords, i1:i2),
            view(ev.jac, i1:i2),
            view(ev.invjac, i1:i2),
            view(ev.Jω, i1:i2),
            view(ev.metric, i1:i2),
            view(ev.volume, i),
        )
    end
end

function PhysicalElementVector(
    std,
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
        elem = PhysicalElement(std, mesh, ie)
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
    return dot(elem.Jω, f)
end

function PhysicalElement(std, mesh::CartesianMesh{ND,RT}, ie) where {ND,RT}
    (; Δx) = mesh

    # Coordinates
    xyz = [phys_coords(ξ, mesh, ie) for ξ in std.ξ]

    # Jacobian and dual basis
    jac = fill(prod(Δx) / (2^ND), ndofs(std))
    invjac = inv.(jac)
    metric = if ND == 1
        fill(SMatrix{1,1}(one(RT)), ndofs(std))
    elseif ND == 2
        fill(
            SMatrix{2,2}(
                Δx[2] / 2, 0,
                0, Δx[1] / 2,
            ),
            ndofs(std),
        )
    else # ND == 3
        fill(
            SMatrix{3,3}(
                Δx[2] * Δx[3] / 4, 0, 0,
                0, Δx[1] * Δx[3] / 4, 0,
                0, 0, Δx[1] * Δx[2] / 4,
            ),
            ndofs(std),
        )
    end

    # Integration weights
    Jω = jac .* std.ω

    # Volume
    vol = sum(Jω)

    return PhysicalElement(xyz, jac, invjac, Jω, metric, vol)
end

function PhysicalElement(std, mesh::StepMesh{RT}, ie) where {RT}
    (; Δx) = mesh
    ireg = region(mesh, ie)

    # Coordinates
    xyz = [phys_coords(ξ, mesh, ie) for ξ in std.ξ]

    # Jacobian and dual basis
    jac = fill(prod(Δx[ireg]) / 4, ndofs(std))
    invjac = inv.(jac)
    metric = fill(
        SMatrix{2,2}(
            Δx[ireg][2] / 2, 0,
            0, Δx[ireg][1] / 2,
        ),
        ndofs(std),
    )

    # Integration weights
    Jω = jac .* std.ω

    # Volume
    vol = sum(Jω)

    return PhysicalElement(xyz, jac, invjac, Jω, metric, vol)
end

function PhysicalElement(std, mesh::UnstructuredMesh{ND,RT}, ie) where {ND,RT}
    # Coordinates
    xyz = [phys_coords(ξ, mesh, ie) for ξ in std.ξ]

    # Jacobian and dual basis
    jac = Vector{RT}(undef, ndofs(std))
    metric = Vector{SMatrix{ND,ND,RT,ND * ND}}(undef, ndofs(std))
    for i in eachdof(std)
        main = map_basis(std.ξ[i], mesh, ie)
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
    Jω = jac .* std.ω

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
            view(fv.surface, i),
        )
    end
end

function PhysicalFaceVector(std, dh, mesh)
    coords = []
    frames = []
    jac = []
    Jω = []
    surface = []

    for iface in eachface(mesh)
        face = PhysicalFace(std, mesh, iface)
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
    return dot(face.Jω, f)
end

function PhysicalFace(_, mesh::CartesianMesh{1,RT}, iface) where {RT}
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

function PhysicalFace(std, mesh::CartesianMesh{2,RT}, iface) where {RT}
    (; Δx) = mesh
    fstd = std.face
    pos = mesh.faces[iface].elempos[1]
    if pos == 1 || pos == 2  # Vertical
        jac = fill(Δx[2] / 2, ndofs(fstd))
        if pos == 1
            frames = [
                ReferenceFrame(
                    SVector(-one(RT), zero(RT)),
                    SVector(zero(RT), -one(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        else # pos == 2
            frames = [
                ReferenceFrame(
                    SVector(one(RT), zero(RT)),
                    SVector(zero(RT), one(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        end

    else  # pos == 3 || pos == 4  # Horizontal
        jac = fill(Δx[1] / 2, ndofs(fstd))
        if pos == 3
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), -one(RT)),
                    SVector(one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        else # pos == 4
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), one(RT)),
                    SVector(-one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        end
    end

    xy = [face_phys_coords(ξ, mesh, iface) for ξ in fstd.ξ]
    Jω = jac .* fstd.ω
    surf = sum(Jω)
    return PhysicalFace(xy, frames, jac, Jω, surf)
end

function PhysicalFace(std, mesh::CartesianMesh{3,RT}, iface) where {RT}
    (; Δx) = mesh
    fstd = std.face
    pos = mesh.faces[iface].elempos[1]
    if pos == 1 || pos == 2  # X faces
        jac = fill(Δx[2] * Δx[3] / 4, ndofs(fstd))
        if pos == 1
            frames = [
                ReferenceFrame(
                    SVector(-one(RT), zero(RT), zero(RT)),
                    SVector(zero(RT), -one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT), one(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        else # pos == 2
            frames = [
                ReferenceFrame(
                    SVector(one(RT), zero(RT), zero(RT)),
                    SVector(zero(RT), one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT), one(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        end
    elseif pos == 3 || pos == 4  # Y faces
        jac = fill(Δx[1] * Δx[3] / 4, ndofs(fstd))
        if pos == 3
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), -one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT), -one(RT)),
                    SVector(one(RT), zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        else # pos == 4
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT), one(RT)),
                    SVector(one(RT), zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        end
    else # pos == 5 || pos == 6  # Z faces
        jac = fill(Δx[1] * Δx[2] / 4, ndofs(fstd))
        if pos == 5
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), zero(RT), -one(RT)),
                    SVector(-one(RT), zero(RT), zero(RT)),
                    SVector(zero(RT), one(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        else # pos == 6
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), zero(RT), one(RT)),
                    SVector(one(RT), zero(RT), zero(RT)),
                    SVector(zero(RT), one(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        end
    end

    xy = [face_phys_coords(ξ, mesh, iface) for ξ in fstd.ξ]
    Jω = jac .* fstd.ω
    surf = sum(Jω)
    return PhysicalFace(xy, frames, jac, Jω, surf)
end

function PhysicalFace(std, mesh::StepMesh{RT}, iface) where {RT}
    (; Δx) = mesh
    fstd = std.face
    ie = mesh.faces[iface].eleminds[1]
    ireg = region(mesh, ie)
    pos = mesh.faces[iface].elempos[1]

    if pos == 1 || pos == 2  # Vertical
        jac = fill(Δx[ireg][2] / 2, ndofs(fstd))
        xy = [face_phys_coords(ξ, mesh, iface) for ξ in fstd.ξ]
        if pos == 1
            frames = [
                ReferenceFrame(
                    SVector(-one(RT), zero(RT)),
                    SVector(zero(RT), -one(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        else # pos == 2
            frames = [
                ReferenceFrame(
                    SVector(one(RT), zero(RT)),
                    SVector(zero(RT), one(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        end

    else  # pos == 3 || pos == 4  # Horizontal
        jac = fill(Δx[ireg][1] / 2, ndofs(fstd))
        xy = [face_phys_coords(ξ, mesh, iface) for ξ in fstd.ξ]
        if pos == 3
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), -one(RT)),
                    SVector(one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        else # pos == 4
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), one(RT)),
                    SVector(-one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        end
    end

    Jω = jac .* fstd.ω
    surf = sum(Jω)
    return PhysicalFace(xy, frames, jac, Jω, surf)
end

function PhysicalFace(_, mesh::UnstructuredMesh{1,RT}, iface) where {RT}
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

function PhysicalFace(std, mesh::UnstructuredMesh{2,RT}, iface) where {RT}
    fstd = std.face
    face = mesh.faces[iface]
    ielem = face.eleminds[1]
    pos = face.elempos[1]

    xy = Vector{SVector{2,RT}}(undef, ndofs(fstd))
    frames = Vector{ReferenceFrame{2,RT}}(undef, ndofs(fstd))
    jac = Vector{RT}(undef, ndofs(fstd))

    if pos == 1  # Left
        for i in eachdof(fstd)
            ξ = SVector(-one(RT), fstd.ξ[i][1])
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
        for i in eachdof(fstd)
            ξ = SVector(one(RT), fstd.ξ[i][1])
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
        for i in eachdof(fstd)
            ξ = SVector(fstd.ξ[i][1], -one(RT))
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
        for i in eachdof(fstd)
            ξ = SVector(fstd.ξ[i][1], one(RT))
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

    Jω = jac .* fstd.ω
    surf = sum(Jω)
    return PhysicalFace(xy, frames, jac, Jω, surf)
end

function PhysicalFace(std, mesh::UnstructuredMesh{3,RT}, iface) where {RT}
    fstd = std.face
    face = mesh.faces[iface]
    ielem = face.eleminds[1]
    pos = face.elempos[1]

    xyz = Vector{SVector{3,RT}}(undef, ndofs(fstd))
    frames = Vector{ReferenceFrame{3,RT}}(undef, ndofs(fstd))
    jac = Vector{RT}(undef, ndofs(fstd))

    if pos == 1  # -X face
        for i in eachdof(fstd)
            ξ = SVector(-one(RT), fstd.ξ[i][1], fstd.ξ[i][2])
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
        for i in eachdof(fstd)
            ξ = SVector(one(RT), fstd.ξ[i][1], fstd.ξ[i][2])
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
        for i in eachdof(fstd)
            ξ = SVector(fstd.ξ[i][1], -one(RT), fstd.ξ[i][2])
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
        for i in eachdof(fstd)
            ξ = SVector(fstd.ξ[i][1], one(RT), fstd.ξ[i][2])
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
        for i in eachdof(fstd)
            ξ = SVector(fstd.ξ[i][1], fstd.ξ[i][2], -one(RT))
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
        for i in eachdof(fstd)
            ξ = SVector(fstd.ξ[i][1], fstd.ξ[i][2], one(RT))
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

    Jω = jac .* fstd.ω
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
    elemvec = PhysicalElementVector(std, dh, mesh)
    facevec = PhysicalFaceVector(std, dh, mesh)
    subgridvec = if subgrid
        PhysicalSubgridVector(std, dh, mesh)
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

Base.@propagate_inbounds function contravariant(F, metric::SMatrix{1,1})
    return (
        F[1] * metric[1, 1],
    )
end

Base.@propagate_inbounds function contravariant(F, metric::SMatrix{2,2})
    return (
        F[1] * metric[1, 1] + F[2] * metric[2, 1],
        F[1] * metric[1, 2] + F[2] * metric[2, 2],
    )
end

Base.@propagate_inbounds function contravariant(F, metric::SMatrix{3,3})
    return (
        F[1] * metric[1, 1] + F[2] * metric[2, 1] + F[3] * metric[3, 1],
        F[1] * metric[1, 2] + F[2] * metric[2, 2] + F[3] * metric[3, 2],
        F[1] * metric[1, 3] + F[2] * metric[2, 3] + F[3] * metric[3, 3],
    )
end

