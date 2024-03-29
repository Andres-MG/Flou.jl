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

struct CartesianMesh{ND,RT,FV,EV,MT} <: AbstractMesh{ND,RT}
    nelements::NTuple{ND,Int}
    Δx::NTuple{ND,RT}
    nodes::Vector{SVector{ND,RT}}
    faces::FV
    elements::EV
    intfaces::Vector{Int}
    bdfaces::Vector{Vector{Int}}
    bdnames::Vector{String}
    bdmap::Dict{Int,Int}
    periodic::Dict{Int,Int}
    mappings::MT
end

nregions(::CartesianMesh) = 1
regions(m::CartesianMesh) = ones(Int, nelements(m))
region(::CartesianMesh, i) = 1

nelements(m::CartesianMesh, _) = nelements(m)
eachelement(m::CartesianMesh, _) = Base.OneTo(nelements(m))
nelements_dir(m::CartesianMesh, dir) = m.nelements[dir]

function CartesianMesh{ND,RT}(start, finish, nxyz) where {ND,RT<:Real}
    1 <= ND <= 3 || throw(ArgumentError(
        "The mesh can only have 1, 2 or 3 dimensions."
    ))
    length(start) == ND || throw(ArgumentError(
        "The `start` point must have $(ND) coordinates."
    ))
    length(finish) == ND || throw(ArgumentError(
        "The `finish` point must have $(ND) coordinates."
    ))
    length(nxyz) == ND || throw(ArgumentError(
        "The number of elements in all directions must be given."
    ))
    all(start .< finish) || throw(ArgumentError(
        "All components of `start` must be lower than those of `finish`."
    ))

    # Construct nodes
    xyz = range.(start, finish, (nxyz .+ 1))
    if ND == 1
        nodes = vec([SVector(RT(px)) for px in xyz])
        mappings = (SegmentLinearMapping(), PointMapping())

    elseif ND == 2
        nodes = vec([SVector(RT(px), RT(py)) for px in xyz[1], py in xyz[2]])
        mappings = (QuadLinearMapping(), SegmentLinearMapping())

    else # ND == 3
        nodes = vec([
            SVector(RT(px), RT(py), RT(pz))
            for px in xyz[1], py in xyz[2], pz in xyz[3]
        ])
        mappings = (HexLinearMapping(), QuadLinearMapping())
    end

    # Connectivities
    elements = _cartesian_element_connectivities(Val(ND), nxyz)
    elements = MeshElementVector(elements)

    intfaces, bdfaces, faces = _cartesian_face_connectivities(Val(ND), nxyz)
    faces = MeshFaceVector(faces)

    # Boundary indices map
    bdnames = [string(i) for i in 1:2ND]
    bdmap = Dict((i, i) for i in 1:2ND)

    return CartesianMesh(
            Tuple(nxyz),
            Tuple(RT.(finish .- start) ./ nxyz),
            nodes,
            faces,
            elements,
            intfaces,
            bdfaces,
            bdnames,
            bdmap,
            Dict{Int,Int}(),
            mappings,
    )
end

function Base.show(io::IO, ::MIME"text/plain", m::CartesianMesh{ND,RT}) where {ND,RT}
    @nospecialize
    # Header
    println(io, ND, "D CartesianMesh{", RT, "}:")
    dirs = ("x", "y", "z")

    # Box limits
    lims = (first(m.nodes), last(m.nodes))
    print(io, " Domain: x ∈ [", lims[1][1], ", ", lims[2][1], "]")
    for idim in 2:ND
        print(io, ", ", dirs[idim], " ∈ [", lims[1][idim], ", ", lims[2][idim], "]")
    end
    println(io)

    # Number of elements
    print(io, " Number of elements: x -> ", nelements_dir(m, 1))
    for idim in 2:ND
        print(io, ", ", dirs[idim], " -> ", nelements_dir(m, idim))
    end

    # Periodic boundaries
    if !isempty(m.periodic)
        println(io)
        print(io, " Periodic boundaries: ", join(m.periodic, ", "))
    end
    return nothing
end

function apply_periodicBCs!(mesh::CartesianMesh, BCs::Pair{String,String}...)
    # Mapped boundaries
    intbcs = Dict{Int,Int}()

    # Check pairs
    for bc in BCs
        bd1 = tryparse(Int, bc.first)
        bd2 = tryparse(Int, bc.second)
        !isnothing(bd1) && !isnothing(bd2) || throw(ArgumentError(
            "Boundary IDs must be integers."
        ))
        1 <= bd1 <= nboundaries(mesh) && 1 <= bd2 <= nboundaries(mesh) || throw(
            ArgumentError(
                "Boundary IDs must be between 1 and $(nboundaries(mesh))."
            )
        )
        bd2 % 2 == 0 && bd1 + 1 == bd2 || throw(ArgumentError(
            "Boundaries $(bc.first) and $(bc.second) cannot be made periodic."
        ))
        intbcs[bd1] = bd2
    end

    return _apply_periodicBCs!(mesh, intbcs)
end

function _cartesian_element_connectivities(::Val{1}, nx)
    enodeinds = Vector{Vector{Int}}(undef, nx)
    faceinds = Vector{Vector{Int}}(undef, nx)
    facepos = [Vector{Int}(undef, 2) for _ in 1:nx]
    for ielem in 1:nx
        enodeinds[ielem] = [ielem, ielem + 1]
        faceinds[ielem] = [ielem, ielem + 1]
        facepos[ielem] = [
            (ielem == 1) ? 1 : 2,
            1,
        ]
    end
    return [MeshElement(enodeinds[i], faceinds[i], facepos[i], 1) for i in 1:nx]
end

function _cartesian_element_connectivities(::Val{2}, nxy)
    nx, ny = nxy
    npx = nx + 1
    nelements = nx * ny
    enodeinds = Vector{Vector{Int}}(undef, nelements)
    faceinds = Vector{Vector{Int}}(undef, nelements)
    facepos = Vector{Vector{Int}}(undef, nelements)
    for j in 1:ny, i in 1:nx
        ielem = (j-1)*nx + i
        enodeinds[ielem] = [
            (j-1)*npx + i,
            (j-1)*npx + i + 1,
            (j-1)*npx + i + 1 + npx,
            (j-1)*npx + i + npx,
        ]
        faceinds[ielem] = [
            (j-1)*npx + i,
            (j-1)*npx + i + 1,
            npx*ny + (j-1)*nx + i,
            npx*ny + (j-1)*nx + i + nx,

        ]
        facepos[ielem] = [
            (i == 1) ? 1 : 2,
            1,
            (j == 1) ? 1 : 2,
            1,
        ]
    end
    return [MeshElement(enodeinds[i], faceinds[i], facepos[i], 1) for i in 1:nelements]
end

function _cartesian_element_connectivities(::Val{3}, nxyz)
    nx, ny, nz = nxyz
    npx, npy, _ = nxyz .+ 1
    nelements = nx * ny * nz
    enodeinds = Vector{Vector{Int}}(undef, nelements)
    faceinds = Vector{Vector{Int}}(undef, nelements)
    facepos = Vector{Vector{Int}}(undef, nelements)
    for k in 1:nz, j in 1:ny, i in 1:nx
        ielem = (k-1)*nx*ny + (j-1)*nx + i
        enodeinds[ielem] = [
            (k-1)*npx*npy + (j-1)*npx + i,
            (k-1)*npx*npy + (j-1)*npx + i + 1,
            (k-1)*npx*npy + (j-1)*npx + i + 1 + npx,
            (k-1)*npx*npy + (j-1)*npx + i + npx,
            npx*npy + (k-1)*npx*npy + (j-1)*npx + i,
            npx*npy + (k-1)*npx*npy + (j-1)*npx + i + 1,
            npx*npy + (k-1)*npx*npy + (j-1)*npx + i + 1 + npx,
            npx*npy + (k-1)*npx*npy + (j-1)*npx + i + npx,
        ]
        faceinds[ielem] = [
            (k-1)*npx*ny + (j-1)*npx + i,
            (k-1)*npx*ny + (j-1)*npx + i + 1,
            npx*ny*nz + (k-1)*nx*npy + (j-1)*nx + i,
            npx*ny*nz + (k-1)*nx*npy + (j-1)*nx + i + nx,
            npx*ny*nz + nx*npy*nz + (k-1)*nx*ny + (j-1)*nx + i,
            npx*ny*nz + nx*npy*nz + (k-1)*nx*ny + (j-1)*nx + i + nx*ny,
        ]
        facepos[ielem] = [
            (i == 1) ? 1 : 2,
            1,
            (j == 1) ? 1 : 2,
            1,
            (k == 1) ? 1 : 2,
            1,
        ]
    end
    return [MeshElement(enodeinds[i], faceinds[i], facepos[i], 1) for i in 1:nelements]
end

function _cartesian_face_connectivities(::Val{1}, nx)
    nfaces = nx + 1
    nbdfaces = 2
    nintfaces = nfaces - nbdfaces

    fnodeinds = Vector{Vector{Int}}(undef, nfaces)
    eleminds = Vector{Vector{Int}}(undef, nfaces)
    elempos = Vector{Vector{Int}}(undef, nfaces)
    intfaces = Int[]; sizehint!(intfaces, nintfaces)
    bdfaces = [Int[], Int[]]

    for iface in 1:nfaces
        if iface == 1
            push!(bdfaces[1], iface)
            eleminds[iface] = [1, 0]
            elempos[iface] = [1, 0]
        elseif iface == nfaces
            push!(bdfaces[2], iface)
            eleminds[iface] = [nx, 0]
            elempos[iface] = [2, 0]
        else
            push!(intfaces, iface)
            eleminds[iface] = [iface - 1, iface]
            elempos[iface] = [2, 1]
        end
        fnodeinds[iface] = [iface]
    end

    return (
        intfaces,
        bdfaces,
        [
            MeshFace(
                fnodeinds[i],
                eleminds[i],
                elempos[i],
                UInt8(0),
                2,
            )
            for i in 1:nfaces
        ],
    )

end

function _cartesian_face_connectivities(::Val{2}, nxy)
    nx, ny = nxy
    npx, npy = nxy .+ 1
    nfaces = npx*ny + npy*nx
    nbdfaces = 2*nx + 2*ny
    nintfaces = nfaces - nbdfaces

    fnodeinds = Vector{Vector{Int}}(undef, nfaces)
    eleminds = Vector{Vector{Int}}(undef, nfaces)
    elempos = Vector{Vector{Int}}(undef, nfaces)
    intfaces = Int[]; sizehint!(intfaces, nintfaces)
    bdfaces = [Int[] for _ in 1:4]

    # Vertical faces
    for j in 1:ny, i in 1:npx
        iface = (j-1)*npx + i
        if i == 1
            push!(bdfaces[1], iface)
            eleminds[iface] = [
                (j-1)*nx + 1,
                0,
            ]
            elempos[iface] = [1, 0]
        elseif i == npx
            push!(bdfaces[2], iface)
            eleminds[iface] = [
                (j-1)*nx + nx,
                0,
            ]
            elempos[iface] = [2, 0]
        else
            push!(intfaces, iface)
            eleminds[iface] = [
                (j-1)*nx + i - 1,
                (j-1)*nx + i,
            ]
            elempos[iface] = [2, 1]
        end
        fnodeinds[iface] = [
            (j-1)*npx + i,
            (j-1)*npx + i + npx,
        ]
    end

    # Horizontal faces
    for j in 1:npy, i in 1:nx
        iface = npx*ny + (j-1)*nx + i
        if j == 1
            push!(bdfaces[3], iface)
            eleminds[iface] = [
                i,
                0,
            ]
            elempos[iface] = [3, 0]
        elseif j == npy
            push!(bdfaces[4], iface)
            eleminds[iface] = [
                (ny-1)*nx + i,
                0,
            ]
            elempos[iface] = [4, 0]
        else
            push!(intfaces, iface)
            eleminds[iface] = [
                (j-2)*nx + i,
                (j-2)*nx + i + nx,
            ]
            elempos[iface] = [4, 3]
        end
        fnodeinds[iface] = [
            (j-1)*npx + i,
            (j-1)*npx + i + 1,
        ]
    end

    return (
        intfaces,
        bdfaces,
        [
            MeshFace(
                fnodeinds[i],
                eleminds[i],
                elempos[i],
                UInt8(0),
                2,
            )
            for i in 1:nfaces
        ],
    )
end

function _cartesian_face_connectivities(::Val{3}, nxyz)
    nx, ny, nz = nxyz
    npx, npy, npz = nxyz .+ 1
    nfaces = npx*ny*nz + npy*nx*nz + npz*nx*ny
    nbdfaces = 2*nx + 2*ny + 2*nz
    nintfaces = nfaces - nbdfaces

    fnodeinds = Vector{Vector{Int}}(undef, nfaces)
    eleminds = Vector{Vector{Int}}(undef, nfaces)
    elempos = Vector{Vector{Int}}(undef, nfaces)
    intfaces = Int[]; sizehint!(intfaces, nintfaces)
    bdfaces = [Int[] for _ in 1:6]

    # X faces
    for k in 1:nz, j in 1:ny, i in 1:npx
        iface = (k-1)*npx*ny + (j-1)*npx + i
        if i == 1
            push!(bdfaces[1], iface)
            eleminds[iface] = [
                (k-1)*nx*ny + (j-1)*nx + 1,
                0,
            ]
            elempos[iface] = [1, 0]
        elseif i == npx
            push!(bdfaces[2], iface)
            eleminds[iface] = [
                (k-1)*nx*ny + (j-1)*nx + nx,
                0,
            ]
            elempos[iface] = [2, 0]
        else
            push!(intfaces, iface)
            eleminds[iface] = [
                (k-1)*nx*ny + (j-1)*nx + i - 1,
                (k-1)*nx*ny + (j-1)*nx + i,
            ]
            elempos[iface] = [2, 1]
        end
        fnodeinds[iface] = [
            (k-1)*npx*npy + (j-1)*npx + i,
            (k-1)*npx*npy + (j-1)*npx + i + npx,
            npx*npy + (k-1)*npx*npy + (j-1)*npx + i + npx,
            npx*npy + (k-1)*npx*npy + (j-1)*npx + i,
        ]
    end

    # Y faces
    for k in 1:nz, j in 1:npy, i in 1:nx
        iface = npx*ny*nz + (k-1)*nx*npy + (j-1)*nx + i
        if j == 1
            push!(bdfaces[3], iface)
            eleminds[iface] = [
                (k-1)*nx*ny + i,
                0,
            ]
            elempos[iface] = [3, 0]
        elseif j == npy
            push!(bdfaces[4], iface)
            eleminds[iface] = [
                (k-1)*nx*ny + (ny-1)*nx + i,
                0,
            ]
            elempos[iface] = [4, 0]
        else
            push!(intfaces, iface)
            eleminds[iface] = [
            (k-1)*nx*ny + (j-2)*nx + i,
            (k-1)*nx*ny + (j-2)*nx + i + nx,
            ]
            elempos[iface] = [4, 3]
        end
        fnodeinds[iface] = [
            (k-1)*npx*npy + (j-1)*npx + i,
            (k-1)*npx*npy + (j-1)*npx + i + 1,
            npx*npy + (k-1)*npx*npy + (j-1)*npx + i + 1,
            npx*npy + (k-1)*npx*npy + (j-1)*npx + i,
        ]
    end

    # Z faces
    for k in 1:npz, j in 1:ny, i in 1:nx
        iface = npx*ny*nz + npy*nx*nz + (k-1)*nx*ny + (j-1)*nx + i
        if k == 1
            push!(bdfaces[5], iface)
            eleminds[iface] = [
                (j-1)*nx + i,
                0,
            ]
            elempos[iface] = [5, 0]
        elseif k == npz
            push!(bdfaces[6], iface)
            eleminds[iface] = [
                (nz-1)*nx*ny + (j-1)*nx + i,
                0,
            ]
            elempos[iface] = [6, 0]
        else
            push!(intfaces, iface)
            eleminds[iface] = [
                (k-2)*nx*ny + (j-1)*nx + i,
                (k-2)*nx*ny + (j-1)*nx + i + nx*ny,
            ]
            elempos[iface] = [6, 5]
        end
        fnodeinds[iface] = [
            (k-1)*npx*npy + (j-1)*npx + i,
            (k-1)*npx*npy + (j-1)*npx + i + 1,
            npx + (k-1)*npx*npy + (j-1)*npx + i + 1,
            npx + (k-1)*npx*npy + (j-1)*npx + i,
        ]
    end

    return (
        intfaces,
        bdfaces,
        [
            MeshFace(
                fnodeinds[i],
                eleminds[i],
                elempos[i],
                UInt8(0),
                2,
            )
            for i in 1:nfaces
        ],
    )
end
