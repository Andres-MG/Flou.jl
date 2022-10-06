#==========================================================================================#
#                                 Element & face vectors                                   #

struct MeshElement{V,M}
    nodeinds::V
    faceinds::V
    facepos::V
    mapind::M
end

struct MeshElementVector <: AbstractVector{MeshElement}
    _nodeoffsets::Vector{Int}
    _faceoffsets::Vector{Int}
    nodeinds::Vector{Int}
    faceinds::Vector{Int}
    facepos::Vector{Int}
    mapind::Vector{Int}
end

function MeshElementVector(elements::AbstractVector{<:MeshElement})
    nodeoffset = vcat(0, cumsum(length(e.nodeinds) for e in elements))
    faceoffset = vcat(0, cumsum(length(e.faceinds) for e in elements))
    return MeshElementVector(
        nodeoffset,
        faceoffset,
        reduce(vcat, (e.nodeinds for e in elements)),
        reduce(vcat, (e.faceinds for e in elements)),
        reduce(vcat, (e.facepos for e in elements)),
        [e.mapind for e in elements],
    )
end

Base.length(s::MeshElementVector) = length(s._nodeoffsets) - 1
Base.size(s::MeshElementVector) = (length(s),)

@inline function Base.getindex(ev::MeshElementVector, i::Integer)
    @boundscheck 1 <= i <= length(ev._nodeoffsets) - 1 || throw(BoundsError(ev, i))
    @inbounds begin
        in1 = ev._nodeoffsets[i] + 1
        in2 = ev._nodeoffsets[i + 1]
        if1 = ev._faceoffsets[i] + 1
        if2 = ev._faceoffsets[i + 1]
        return MeshElement(
            view(ev.nodeinds, in1:in2),
            view(ev.faceinds, if1:if2),
            view(ev.facepos, if1:if2),
            view(ev.mapind, i),
        )
    end
end

@inline function copy_element(ev::MeshElementVector, i::Integer)
    @boundscheck 1 <= i <= length(ev._nodeoffsets) - 1 || throw(BoundsError(ev, i))
    @inbounds begin
        in1 = ev._nodeoffsets[i] + 1
        in2 = ev._nodeoffsets[i + 1]
        if1 = ev._faceoffsets[i] + 1
        if2 = ev._faceoffsets[i + 1]
        return MeshElement(
            ev.nodeinds[in1:in2],
            ev.faceinds[if1:if2],
            ev.facepos[if1:if2],
            ev.mapind[i],
        )
    end
end

struct MeshFace{V,O,M}
    nodeinds::V
    eleminds::V
    elempos::V
    orientation::O
    mapind::M
end

struct MeshFaceVector <: AbstractVector{MeshFace}
    _nodeoffsets::Vector{Int}
    _elemoffsets::Vector{Int}
    nodeinds::Vector{Int}
    eleminds::Vector{Int}
    elempos::Vector{Int}
    orientation::Vector{UInt8}
    mapind::Vector{Int}
end

function MeshFaceVector(faces::AbstractVector{<:MeshFace})
    nodeoffset = vcat(0, cumsum(length(f.nodeinds) for f in faces))
    elemoffset = vcat(0, cumsum(length(f.eleminds) for f in faces))
    return MeshFaceVector(
        nodeoffset,
        elemoffset,
        reduce(vcat, (f.nodeinds for f in faces)),
        reduce(vcat, (f.eleminds for f in faces)),
        reduce(vcat, (f.elempos for f in faces)),
        [f.orientation for f in faces],
        [f.mapind for f in faces],
    )
end

Base.length(s::MeshFaceVector) = length(s._nodeoffsets) - 1
Base.size(s::MeshFaceVector) = (length(s),)

@inline function Base.getindex(fv::MeshFaceVector, i::Integer)
    @boundscheck 1 <= i <= length(fv._nodeoffsets) - 1 || throw(BoundsError(fv, i))
    @inbounds begin
        in1 = fv._nodeoffsets[i] + 1
        in2 = fv._nodeoffsets[i + 1]
        ie1 = fv._elemoffsets[i] + 1
        ie2 = fv._elemoffsets[i + 1]
        return MeshFace(
            view(fv.nodeinds, in1:in2),
            view(fv.eleminds, ie1:ie2),
            view(fv.elempos, ie1:ie2),
            view(fv.orientation, i),
            view(fv.mapind, i),
        )
    end
end

@inline function copy_face(fv::MeshFaceVector, i::Integer)
    @boundscheck 1 <= i <= length(fv._nodeoffsets) - 1 || throw(BoundsError(fv, i))
    @inbounds begin
        in1 = fv._nodeoffsets[i] + 1
        in2 = fv._nodeoffsets[i + 1]
        ie1 = fv._elemoffsets[i] + 1
        ie2 = fv._elemoffsets[i + 1]
        return MeshFace(
            fv.nodeinds[in1:in2],
            fv.eleminds[ie1:ie2],
            fv.elempos[ie1:ie2],
            fv.orientation[i],
            fv.mapind[i],
        )
    end
end

function Base.deleteat!(fv::MeshFaceVector, i::Integer)
    # Delete the face
    i1 = fv._nodeoffsets[i] + 1
    i2 = fv._nodeoffsets[i + 1]
    deleteat!(fv.nodeinds, i1:i2)

    i1 = fv._elemoffsets[i] + 1
    i2 = fv._elemoffsets[i + 1]
    deleteat!(fv.eleminds, i1:i2)
    deleteat!(fv.elempos, i1:i2)

    deleteat!(fv.orientation, i)
    deleteat!(fv.mapind, i)

    # Update the offsets
    Δ = fv._nodeoffsets[i + 1] - fv._nodeoffsets[i]
    deleteat!(fv._nodeoffsets, i + 1)
    fv._nodeoffsets[(i + 1):end] .-= Δ

    Δ = fv._elemoffsets[i + 1] - fv._elemoffsets[i]
    deleteat!(fv._elemoffsets, i + 1)
    fv._elemoffsets[(i + 1):end] .-= Δ

    return nothing
end

# i must be sorted!!
@inbounds function Base.deleteat!(fv::MeshFaceVector, i::AbstractVecOrTuple)
    for ipos in eachindex(i)
        iface = i[ipos] - (ipos - 1)
        deleteat!(fv, iface)
    end
    return nothing
end

nelements(ev::MeshElementVector) = length(ev)
nfaces(fv::MeshFaceVector) = length(fv)

#==========================================================================================#
#                                 Generic mesh interface                                   #

abstract type AbstractMesh{ND,RT<:Real} end

get_spatialdim(::AbstractMesh{ND}) where {ND} = ND

nelements(m::AbstractMesh) = nelements(m.elements)
nboundaries(m::AbstractMesh) = length(m.bdfaces)
nfaces(m::AbstractMesh) = nfaces(m.faces)
nintfaces(m::AbstractMesh) = length(m.intfaces)
nbdfaces(m::AbstractMesh) = sum(length.(m.bdfaces))
nbdfaces(m::AbstractMesh, i) = length(m.bdfaces[i])
nperiodic(m::AbstractMesh) = length(m.periodic)
nvertices(m::AbstractMesh) = length(m.nodes)
nvertices(e::MeshElement) = length(e.nodeinds)
nvertices(f::MeshFace) = length(f.nodeinds)
nmappings(m::AbstractMesh) = length(m.mappings)
function nregions end

get_intfaces(m::AbstractMesh) = LazyVector(m.faces, m.intfaces)
get_bdfaces(m::AbstractMesh) = LazyVector(m.faces, vcat(m.bdfaces...))
get_bdfaces(m::AbstractMesh, i) = LazyVector(m.faces, m.bdfaces[i])
get_periodic(m::AbstractMesh) = m.periodic
get_mappings(m::AbstractMesh) = m.mappings

copy_element(m::AbstractMesh, i::Integer) = copy_element(m.elements, i)
copy_face(m::AbstractMesh, i::Integer) = copy_face(m.faces, i)

get_intface(m::AbstractMesh, i) = m.faces[m.intfaces[i]]
get_bdface(m::AbstractMesh, ib, i) = m.faces[m.bdfaces[ib][i]]
get_periodic(m::AbstractMesh, i) = m.periodic[i]
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
eachregion(m::AbstractMesh) = Base.OneTo(nregions(m))

function apply_periodicBCs! end

function _apply_periodicBCs!(mesh::AbstractMesh, BCs::Dict{Int,Int})
    faces2del = Int[]
    for bc in BCs
        # Unpack
        bd1, bd2 = bc

        # Checks
        bd1 in keys(mesh.bdmap) && bd2 in keys(mesh.bdmap) || throw(
            ArgumentError("Boundaries $(bc.first) and $(bc.second) cannot be periodic.")
        )

        # Update connectivities
        bdind = mesh.bdmap[bd1] => mesh.bdmap[bd2]
        for (if1, if2) in zip(eachbdface(mesh, bdind.first), eachbdface(mesh, bdind.second))
            face1 = mesh.faces[if1]
            face2 = mesh.faces[if2]
            elmind = face2.eleminds[1]
            elmpos = face2.elempos[1]
            face1.eleminds[2] = elmind
            face1.elempos[2] = elmpos
            elem = mesh.elements[elmind]
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
                mesh.bdmap[i] -= 1
            end
            if mesh.bdmap[i] > bdind.first
                mesh.bdmap[i] -= 1
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
    deleteat!(mesh.faces, faces2del)

    # Update indices in element connectivities
    for iface in eachface(mesh)
        f = mesh.faces[iface]
        # Master element
        elmind = f.eleminds[1]
        elmpos = f.elempos[1]
        mesh.elements[elmind].faceinds[elmpos] = iface
        # Slave element
        elmind = f.eleminds[2]
        if elmind != 0
            elmpos = f.elempos[2]
            mesh.elements[elmind].faceinds[elmpos] = iface
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
    elem = mesh.elements[ie]
    mapping = get_mapping(mesh, elem.mapind[])
    nodes = mesh.nodes[elem.nodeinds]
    return phys_coords(ξ, nodes, mapping)
end

function face_phys_coords(ξ, mesh::AbstractMesh, iface)
    face = mesh.faces[iface]
    mapping = get_mapping(mesh, face.mapind[])
    nodes = mesh.nodes[face.nodeinds]
    return phys_coords(ξ, nodes, mapping)
end

function map_basis(ξ, mesh::AbstractMesh, ie)
    elem = mesh.elements[ie]
    mapping = get_mapping(mesh, elem.mapind[])
    nodes = mesh.nodes[elem.nodeinds]
    return map_basis(ξ, nodes, mapping)
end

function map_dual_basis(main, mesh::AbstractMesh, ie)
    elem = mesh.elements[ie]
    mapping = get_mapping(mesh, elem.mapind[])
    return map_dual_basis(main, mapping)
end

function map_jacobian(main, mesh::AbstractMesh, ie)
    elem = mesh.elements[ie]
    mapping = get_mapping(mesh, elem.mapind[])
    return map_jacobian(main, mapping)
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

function map_jacobian(main, ::PointMapping)
    return one(eltype(main))
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

function map_jacobian(main, ::SegmentLinearMapping)
    return main[1][1]
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

function map_jacobian(main, ::QuadLinearMapping)
    return main[1][1] * main[2][2] - main[1][2] * main[2][1]
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

function map_jacobian(main, ::HexLinearMapping)
    return dot(main[1], cross(main[2], main[3]))
end

include("CartesianMesh.jl")
include("StepMesh.jl")
include("GmshMesh.jl")