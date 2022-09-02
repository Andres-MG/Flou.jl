struct MeshElement
    nodeinds::Vector{Int}
    faceinds::Vector{Int}
    facepos::Vector{Int}
    mapind::Int
end

struct MeshFace
    nodeinds::Vector{Int}
    eleminds::Vector{Int}
    elempos::Vector{Int}
    orientation::UInt8
    mapind::Int
end

abstract type AbstractMesh{ND,RT<:Real} end

get_spatialdim(::AbstractMesh{ND}) where {ND} = ND

nelements(m::AbstractMesh) = length(m.elements)
nboundaries(m::AbstractMesh) = length(m.bdfaces)
nfaces(m::AbstractMesh) = length(m.faces)
nintfaces(m::AbstractMesh) = length(m.intfaces)
nbdfaces(m::AbstractMesh) = sum(length.(m.bdfaces))
nbdfaces(m::AbstractMesh, i) = length(m.bdfaces[i])
nperiodic(m::AbstractMesh) = length(m.periodic)
nvertices(m::AbstractMesh) = length(m.nodes)
nvertices(e::MeshElement) = length(e.nodeinds)
nvertices(f::MeshFace) = length(f.nodeinds)
nmappings(m::AbstractMesh) = length(m.mappings)
function nregions end

get_elements(m::AbstractMesh) = LazyRows(m.elements)
get_faces(m::AbstractMesh) = LazyRows(m.faces)
get_intfaces(m::AbstractMesh) = LazyRows(m.faces[m.intfaces])
get_bdfaces(m::AbstractMesh) = LazyRows(m.faces[vcat(m.bdfaces...)])
get_bdfaces(m::AbstractMesh, i) = LazyRows(m.faces[m.bdfaces[i]])
get_periodic(m::AbstractMesh) = m.periodic
get_vertices(m::AbstractMesh) = m.nodes
get_mappings(m::AbstractMesh) = m.mappings

get_element(m::AbstractMesh, i) = LazyRow(m.elements, i)
get_face(m::AbstractMesh, i) = LazyRow(m.faces, i)
get_intface(m::AbstractMesh, i) = LazyRow(m.faces, m.intfaces[i])
get_bdface(m::AbstractMesh, ib, i) = LazyRow(m.faces, m.bdfaces[ib][i])
get_periodic(m::AbstractMesh, i) = m.periodic[i]
get_vertex(m::AbstractMesh, i) = m.nodes[i]
get_mapping(m::AbstractMesh, i) = m.mappings[i]
function get_region end

eachelement(m::AbstractMesh) = Base.OneTo(nelements(m))
eachboundary(m::AbstractMesh) = Base.OneTo(nboundaries(m))
eachface(m::AbstractMesh) = Base.OneTo(nfaces(m))
eachintface(m::AbstractMesh) = m.intfaces
eachbdface(m::AbstractMesh, i) = m.bdfaces[i]
eachbdface(m::AbstractMesh) = vcat(m.bdfaces...)
eachvertex(m::AbstractMesh) = Base.OneTo(nvertices(m))
eachmapping(m::AbstractMesh) = Base.OneTo(nmappings(m))
function eachregion end

function apply_periodicBCs! end

function _apply_periodicBCs!(mesh::AbstractMesh, BCs::Dict{Int,Int})
    faces2del = Int[]
    for bc in BCs
        # Unpack
        bd1, bd2 = bc

        # Checks
        bd1 in keys(mesh.bdmap) && bd2 in keys(mesh.bdmap) || throw(
            ArgumentError("Boundaries $(bc.first) and $(bc.second) are already periodic.")
        )

        # Update connectivities
        bdind = mesh.bdmap[bd1] => mesh.bdmap[bd2]
        for (if1, if2) in zip(eachbdface(mesh, bdind.first), eachbdface(mesh, bdind.second))
            face1 = get_face(mesh, if1)
            face2 = get_face(mesh, if2)
            elmind = face2.eleminds[1]
            elmpos = face2.elempos[1]
            face1.eleminds[2] = elmind
            face1.elempos[2] = elmpos
            elem = get_element(mesh, elmind)
            elem.faceinds[elmpos] = if1
            elem.facepos[elmpos] = 2
        end

        # Modify face indices and faces to delete
        append!(mesh.intfaces, mesh.bdfaces[bdind.first])
        append!(faces2del, mesh.bdfaces[bdind.second])
        deleteat!(mesh.bdfaces, bdind)

        # Boundary mappings
        delete!(mesh.bdmap, bd1)
        delete!(mesh.bdmap, bd2)
        for i in keys(mesh.bdmap)
            if mesh.bdmap[i] > bdind.second
                mesh.bdmap[i] -= 2
            end
        end
        push!(mesh.periodic, bd1 => bd2)
    end

    # Update face indices
    sort!(mesh.intfaces)
    sort!(faces2del)
    for i in eachindex(get_intfaces(mesh))
        Δ = findfirst(>(mesh.intfaces[i]), faces2del)
        if isnothing(Δ)
            Δ = length(faces2del)
        else
            Δ -= 1
        end
        mesh.intfaces[i] -= Δ
    end
    for ib in eachboundary(mesh)
        for i in eachindex(get_bdfaces(mesh, ib))
            Δ = findfirst(>(mesh.bdfaces[ib][i]), faces2del)
            if isnothing(Δ)
                Δ = length(faces2del)
            else
                Δ -= 1
            end
            mesh.bdfaces[ib][i] -= Δ
        end
    end

    # Delete duplicated faces
    StructArrays.foreachfield(x -> deleteat!(x, faces2del), mesh.faces)

    # Update indices in element connectivities
    for iface in eachface(mesh)
        f = get_face(mesh, iface)
        # Master element
        elmind = f.eleminds[1]
        elmpos = f.elempos[1]
        get_element(mesh, elmind).faceinds[elmpos] = iface
        # Slave element
        elmind = f.eleminds[2]
        if elmind != 0
            elmpos = f.elempos[2]
            get_element(mesh, elmind).faceinds[elmpos] = iface
        end
    end
    return nothing
end

abstract type AbstractMapping end

struct PointMapping <: AbstractMapping end
struct SegmentLinearMapping <: AbstractMapping end
struct QuadLinearMapping <: AbstractMapping end
struct HexLinearMapping <: AbstractMapping end

function phys_coords(ξ, mesh::AbstractMesh, ie)
    elem = get_element(mesh, ie)
    mapping = get_mapping(mesh, elem.mapind)
    nodes = get_vertices(mesh)[elem.nodeinds]
    return phys_coords(ξ, nodes, mapping)
end

function face_phys_coords(ξ, mesh::AbstractMesh, iface)
    face = get_face(mesh, iface)
    mapping = get_mapping(mesh, face.mapind)
    nodes = get_vertices(mesh)[face.nodeinds]
    return phys_coords(ξ, nodes, mapping)
end

function map_basis(ξ, mesh::AbstractMesh, ie)
    elem = get_element(mesh, ie)
    mapping = get_mapping(mesh, elem.mapind)
    nodes = get_vertices(mesh)[elem.nodeinds]
    return map_basis(ξ, nodes, mapping)
end

function map_dual_basis(main, mesh::AbstractMesh, ie)
    elem = get_element(mesh, ie)
    mapping = get_mapping(mesh, elem.mapind)
    return map_dual_basis(main, mapping)
end

function phys_coords(::AbstractVector, nodes::AbstractVector, ::PointMapping)
    return SVector{length(nodes[1])}(nodes[1])
end

function map_basis(ξ::AbstractVector, ::AbstractVector, ::PointMapping)
    return (SVector(zero(typeof(ξ))),)
end

function map_dual_basis(main, ::PointMapping)
    return main
end

function phys_coords(ξ::AbstractVector, nodes::AbstractVector, ::SegmentLinearMapping)
    ξrel = (ξ[1] + 1) / 2
    x = nodes[1] .* (1 - ξrel) .+ nodes[2] .* ξrel
    return SVector(x)
end

function map_basis(::AbstractVector, nodes::AbstractVector, ::SegmentLinearMapping)
    dxdξ = (nodes[2] .- nodes[1]) ./ 2
    return (SVector(dxdξ),)
end

function map_dual_basis(main, ::SegmentLinearMapping)
    rt = eltype(main[1])
    return (SVector(one(rt)),)
end

function phys_coords(ξ::AbstractVector, nodes::AbstractVector, ::QuadLinearMapping)
    ξrel, ηrel = (ξ .+ 1) ./ 2
    return SVector(
        nodes[1] .* (1 - ξrel) .* (1 - ηrel) .+
        nodes[2] .* ξrel .* (1 - ηrel) .+
        nodes[3] .* ξrel .* ηrel .+
        nodes[4] .* (1 - ξrel) .* ηrel
    )
end

function map_basis(ξ::AbstractVector, nodes::AbstractVector, ::QuadLinearMapping)
    ξrel, ηrel = (ξ .+ 1) ./ 2
    dxdξ = SVector(
        (nodes[2] .- nodes[1]) ./ 2 .* (1 - ηrel) .+
        (nodes[3] .- nodes[4]) ./ 2 .* ηrel
    )
    dxdη = SVector(
        (nodes[4] .- nodes[1]) ./ 2 .* (1 - ξrel) .+
        (nodes[3] .- nodes[2]) ./ 2 .* ξrel
    )
    return (dxdξ, dxdη)
end

function map_dual_basis(main, ::QuadLinearMapping)
    return (
        SVector(main[2][2], -main[2][1]),
        SVector(-main[1][2], main[1][1]),
    )
end

function phys_coords(ξ::AbstractVector, nodes::AbstractVector, ::HexLinearMapping)
    ξrel, ηrel, ζrel = (ξ .+ 1) ./ 2
    return SVector(
        nodes[1] .* (1 - ξrel) .* (1 - ηrel) .* (1 - ζrel) .+
        nodes[2] .* ξrel .* (1 - ηrel) .* (1 - ζrel) .+
        nodes[3] .* ξrel .* ηrel .* (1 - ζrel) .+
        nodes[4] .* (1 - ξrel) .* ηrel .* (1 - ζrel) .+
        nodes[5] .* (1 - ξrel) .* (1 - ηrel) .* ζrel .+
        nodes[6] .* ξrel .* (1 - ηrel) .* ζrel .+
        nodes[7] .* ξrel .* ηrel .* ζrel .+
        nodes[8] .* (1 - ξrel) .* ηrel .* ζrel
    )
end

function map_basis(ξ::AbstractVector, nodes::AbstractVector, ::HexLinearMapping)
    ξrel, ηrel, ζrel = (ξ .+ 1) ./ 2
    dxdξ = SVector(
        (1 - ζrel) .* (
            (nodes[2] .- nodes[1]) ./ 2 .* (1 - ηrel) .+
            (nodes[3] .- nodes[4]) ./ 2 .* ηrel) .+
        ζrel .* (
            (nodes[6] .- nodes[5]) ./ 2 .* (1 - ηrel) .+
            (nodes[7] .- nodes[8]) ./ 2 .* ηrel)
    )
    dxdη = SVector(
        (1 - ξrel) .* (
            (nodes[4] .- nodes[1]) ./ 2 .* (1 - ζrel) .+
            (nodes[8] .- nodes[5]) ./ 2 .* ζrel) .+
        ξrel .* (
            (nodes[3] .- nodes[2]) ./ 2 .* (1 - ζrel) .+
            (nodes[7] .- nodes[6]) ./ 2 .* ζrel)
    )
    dxdζ = SVector(
        (1 - ηrel) .* (
            (nodes[5] .- nodes[1]) ./ 2 .* (1 - ξrel) .+
            (nodes[6] .- nodes[2]) ./ 2 .* ξrel) .+
        ηrel .* (
            (nodes[8] .- nodes[4]) ./ 2 .* (1 - ξrel) .+
            (nodes[7] .- nodes[3]) ./ 2 .* ξrel)
    )
    return (dxdξ, dxdη, dxdζ)
end

function map_dual_basis(main, ::HexLinearMapping)
    return (
        cross(main[2], main[3]),
        cross(main[3], main[1]),
        cross(main[1], main[2]),
    )
end

include("CartesianMesh.jl")
include("StepMesh.jl")
include("GmshMesh.jl")
