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

"""
    StepMesh{RT<:Real}(start, finish, offset, height, nxy)

A mesh representing a domain with the shape of an L.

<pre>
          4
  ┌---------------┐
  | (2)     (3)   | 2
1 |     ┌---------┘
  | (1) | 5  6
  └-----┘
     3
</pre>
"""
struct StepMesh{RT,FV,EV,MT} <: AbstractMesh{2,RT}
    nelements::NTuple{3,Tuple{Int,Int}}
    Δx::NTuple{3,Tuple{RT,RT}}
    nodes::Vector{SVector{2,RT}}
    faces::FV
    elements::EV
    intfaces::Vector{Int}
    bdfaces::Vector{Vector{Int}}
    periodic::Dict{Int,Int}
    bdnames::Vector{String}
    bdmap::Dict{Int,Int}
    regionmap::Vector{Int}
    mappings::MT
    step_offset::RT
    step_height::RT
end

nregions(::StepMesh) = 3
regions(m::StepMesh) = m.regionmap
region(m::StepMesh, i) = regions(m)[i]

nelements(m::StepMesh, i) = prod(m.nelements[i])
eachelement(m::StepMesh, i) = Base.OneTo(nelements(m, i))

function StepMesh{RT}(start, finish, offset, height, nxy) where {RT}
    length(start) == 2 || throw(ArgumentError(
        "The `start` point must have 2 coordinates."
    ))
    length(finish) == 2 || throw(ArgumentError(
        "The `finish` point must have 2 coordinates."
    ))
    length(nxy) == 3 || throw(ArgumentError(
        "A `StepMesh`` must have three regions."
    ))
    nxy[1][1] == nxy[2][1] && nxy[2][2] == nxy[3][2] || throw(ArgumentError(
        "The resulting `StepMesh` must be conforming. Check the number of elements."
    ))
    all(start .< finish) || throw(ArgumentError(
        "All components of `start` must be lower than those of `finish`."
    ))
    height > 0 || throw(ArgumentError(
        "The step must have a positive height."
    ))
    offset > 0 || throw(ArgumentError(
        "The step offset must be positive."
    ))

    x0 = start
    xf = (x0[1] + offset, x0[2] + height)
    mesh1 = CartesianMesh{2,RT}(x0, xf, nxy[1])

    x0 = (start[1], start[2] + height)
    xf = (start[1] + offset, finish[2])
    mesh2 = CartesianMesh{2,RT}(x0, xf, nxy[2])

    x0 = (start[1] + offset, start[2] + height)
    xf = finish
    mesh3 = CartesianMesh{2,RT}(x0, xf, nxy[3])

    return _step_merge_meshes(mesh1, mesh2, mesh3, offset, height)
end

function Base.show(io::IO, ::MIME"text/plain", m::StepMesh{RT}) where {RT}
    @nospecialize
    # Header
    println(io, "2D StepMesh{", RT, "}:")

    # Box limits
    lims = (first(m.nodes), last(m.nodes))
    print(io, " Domain: x ∈ [", lims[1][1], ", ", lims[2][1], "],")
    println(io, " y ∈ [", lims[1][2], ", ", lims[2][2], "]")

    # Step size
    print(io, " Step of height ", m.step_height)
    println(io, " with an offset of ", m.step_offset)

    # Number of elements
    print(io, " Number of elements: ", nelements(m))

    return nothing
end

function _step_merge_meshes(
    mesh1::CartesianMesh{ND,RT},
    mesh2::CartesianMesh{ND,RT},
    mesh3::CartesianMesh{ND,RT},
    offset,
    height,
) where {
    ND,
    RT,
}
    nx1, ny1 = nelements_dir(mesh1, 1), nelements_dir(mesh1, 2)
    nx2, ny2 = nelements_dir(mesh2, 1), nelements_dir(mesh2, 2)
    nx3, ny3 = nelements_dir(mesh3, 1), nelements_dir(mesh3, 2)

    # Maps
    nodemap = ntuple(_ -> Dict{Int,Int}(), 3)
    facemap = ntuple(_ -> Dict{Int,Int}(), 3)
    elemmap = ntuple(_ -> Dict{Int,Int}(), 3)
    bdmap = ntuple(_ -> Dict{Int,Int}(), 3)

    # Region 1
    map(i -> nodemap[1][i] = i, eachvertex(mesh1))
    map(i -> facemap[1][i] = i, eachface(mesh1))
    map(i -> elemmap[1][i] = i, eachelement(mesh1))
    nnodes1 = length(nodemap[1])
    nfaces1 = length(facemap[1])
    nelems1 = length(elemmap[1])

    # Region 2
    li1 = LinearIndices((nx1 + 1, ny1 + 1))
    li2 = LinearIndices((nx2 + 1, ny2 + 1))
    for (i1, i2) in zip(li1[:, end], li2[:, 1])
        nodemap[2][i2] = nodemap[1][i1]
    end
    for (i0, i) in enumerate(li2[:, 2:end])
        nodemap[2][i] = nnodes1 + i0
    end

    nvfaces1 = (nx1 + 1) * ny1
    nvfaces2 = (nx2 + 1) * ny2
    for i in 1:nvfaces2
        facemap[2][i] = nfaces1 + i
    end
    li1 = LinearIndices((nx1, ny1 + 1))
    li2 = LinearIndices((nx2, ny2 + 1))
    for (i1, i2) in zip(li1[:, end], li2[:, 1])
        facemap[2][nvfaces2 + i2] = facemap[1][nvfaces1 + i1]
        bdmap[2][nvfaces2 + i2] = mesh2.faces[nvfaces2 + i2].eleminds[1]
    end
    for (i0, i) in enumerate(li2[:, 2:end])
        facemap[2][nvfaces2 + i] = nfaces1 + nvfaces2 + i0
    end

    map(i -> elemmap[2][i] = nelems1 + i, eachelement(mesh2))

    nnodes2 = length(nodemap[2]) - (nx2 + 1)
    nfaces2 = length(facemap[2]) - nx2
    nelems2 = length(elemmap[2])

    # Region 3
    li2 = LinearIndices((nx2 + 1, ny2 + 1))
    li3 = LinearIndices((nx3 + 1, ny3 + 1))
    for (i2, i3) in zip(li2[end, :], li3[1, :])
        nodemap[3][i3] = nodemap[2][i2]
    end
    for (i0, i) in enumerate(li3[2:end, :])
        nodemap[3][i] = nnodes1 + nnodes2 + i0
    end

    li2 = LinearIndices((nx2 + 1, ny2))
    li3 = LinearIndices((nx3 + 1, ny3))
    for (i2, i3) in zip(li2[end, :], li3[1, :])
        facemap[3][i3] = facemap[2][i2]
        bdmap[3][i3] = mesh3.faces[i3].eleminds[1]
    end
    cnt = 0
    for (i0, i) in enumerate(li3[2:end, :])
        cnt += 1
        facemap[3][i] = nfaces1 + nfaces2 + i0
    end
    nvfaces = prod(size(li3))
    for (i0, i) in enumerate((nvfaces + 1):nfaces(mesh3))
        facemap[3][i] = nfaces1 + nfaces2 + cnt + i0
    end

    map(i -> elemmap[3][i] = nelems1 + nelems2 + i, eachelement(mesh3))

    nelems3 = length(elemmap[3])

    # Indices of entities to be deleted
    nodes2del = ntuple(_ -> Int[], 3)
    faces2del = ntuple(_ -> Int[], 3)
    for iface in eachbdface(mesh2, 3)
        append!(nodes2del[2], mesh2.faces[iface].nodeinds)
    end
    append!(faces2del[2], eachbdface(mesh2, 3))
    unique!(nodes2del[2])

    for iface in eachbdface(mesh3, 1)
        append!(nodes2del[3], mesh3.faces[iface].nodeinds)
    end
    append!(faces2del[3], eachbdface(mesh3, 1))
    unique!(nodes2del[3])

    # Update connectivities
    for (i, elem) in enumerate(mesh2.elements)
        for (inode, node) in enumerate(elem.nodeinds)
            mesh2.elements[i].nodeinds[inode] = nodemap[2][node]
        end
        for (iface, face) in enumerate(elem.faceinds)
            mesh2.elements[i].faceinds[iface] = facemap[2][face]
        end
    end
    for (i, face) in enumerate(mesh2.faces)
        for (inode, node) in enumerate(face.nodeinds)
            mesh2.faces[i].nodeinds[inode] = nodemap[2][node]
        end
        for (ielem, elem) in enumerate(face.eleminds)
            if elem == 0 continue end
            mesh2.faces[i].eleminds[ielem] = elemmap[2][elem]
        end
    end

    for (i, elem) in enumerate(mesh3.elements)
        for (inode, node) in enumerate(elem.nodeinds)
            mesh3.elements[i].nodeinds[inode] = nodemap[3][node]
        end
        for (iface, face) in enumerate(elem.faceinds)
            mesh3.elements[i].faceinds[iface] = facemap[3][face]
        end
    end
    for (i, face) in enumerate(mesh3.faces)
        for (inode, node) in enumerate(face.nodeinds)
            mesh3.faces[i].nodeinds[inode] = nodemap[3][node]
        end
        for (ielem, elem) in enumerate(face.eleminds)
            if elem == 0 continue end
            mesh3.faces[i].eleminds[ielem] = elemmap[3][elem]
        end
    end

    # Final list of nodes, faces and elements
    deleteat!(mesh2.nodes, nodes2del[2])
    deleteat!(mesh3.nodes, nodes2del[3])
    nodes = mesh1.nodes
    append!(nodes, mesh2.nodes)
    append!(nodes, mesh3.nodes)

    deleteat!(mesh2.faces, faces2del[2])
    deleteat!(mesh3.faces, faces2del[3])

    faces = vcat(
        [copyface(mesh1, i) for i in eachface(mesh1)],
        [copyface(mesh2, i) for i in eachface(mesh2)],
        [copyface(mesh3, i) for i in eachface(mesh3)],
    )
    faces = MeshFaceVector(faces)

    elements = vcat(
        [copyelement(mesh1, i) for i in eachelement(mesh1)],
        [copyelement(mesh2, i) for i in eachelement(mesh2)],
        [copyelement(mesh3, i) for i in eachelement(mesh3)],
    )
    elements = MeshElementVector(elements)

    for i in eachbdface(mesh2, 3)
        iface = facemap[2][i]
        ielem = elemmap[2][bdmap[2][i]]
        faces[iface].eleminds[2] = ielem
        faces[iface].elempos[2] = 3
        elements[ielem].facepos[3] = 2
    end
    for i in eachbdface(mesh3, 1)
        iface = facemap[3][i]
        ielem = elemmap[3][bdmap[3][i]]
        faces[iface].eleminds[2] = ielem
        faces[iface].elempos[2] = 1
        elements[ielem].facepos[1] = 2
    end

    # List of interior and boundary faces
    intfaces = Int[]
    sizehint!(intfaces, nintfaces(mesh1) + nintfaces(mesh2) + nintfaces(mesh3))
    for i in eachintface(mesh1)
        push!(intfaces, facemap[1][i])
    end
    for i in eachbdface(mesh1, 4)
        push!(intfaces, facemap[1][i])
    end
    for i in eachintface(mesh2)
        push!(intfaces, facemap[2][i])
    end
    for i in eachbdface(mesh2, 2)
        push!(intfaces, facemap[2][i])
    end
    for i in eachintface(mesh3)
        push!(intfaces, facemap[3][i])
    end
    sort!(intfaces)

    bdfaces = [Int[] for _ in 1:6]
    for i in eachbdface(mesh1, 1)
        push!(bdfaces[1], facemap[1][i])
    end
    for i in eachbdface(mesh2, 1)
        push!(bdfaces[1], facemap[2][i])
    end
    for i in eachbdface(mesh3, 2)
        push!(bdfaces[2], facemap[3][i])
    end
    for i in eachbdface(mesh1, 3)
        push!(bdfaces[3], facemap[1][i])
    end
    for i in eachbdface(mesh2, 4)
        push!(bdfaces[4], facemap[2][i])
    end
    for i in eachbdface(mesh3, 4)
        push!(bdfaces[4], facemap[3][i])
    end
    for i in eachbdface(mesh1, 2)
        push!(bdfaces[5], facemap[1][i])
    end
    for i in eachbdface(mesh3, 3)
        push!(bdfaces[6], facemap[3][i])
    end
    sort!.(bdfaces)

    bdnames = [string(i) for i in 1:6]
    bdmap = Dict((i, i) for i in 1:6)
    regionmap = fill(1, nelems1)
    append!(regionmap, fill(2, nelems2))
    append!(regionmap, fill(3, nelems3))
    mappings = mesh1.mappings

    return StepMesh(
        ((nx1, ny1), (nx2, ny2), (nx3, ny3)),
        (mesh1.Δx, mesh2.Δx, mesh3.Δx),
        nodes,
        faces,
        elements,
        intfaces,
        bdfaces,
        Dict{Int,Int}(),
        bdnames,
        bdmap,
        regionmap,
        mappings,
        offset,
        height,
    )
end
