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

spatialdim(::AbstractMesh{ND}) where {ND} = ND

nelements(m::AbstractMesh) = length(m.elements)
nboundaries(m::AbstractMesh) = length(m.bdfaces)
nfaces(m::AbstractMesh) = length(m.faces)
nintfaces(m::AbstractMesh) = length(m.intfaces)
nbdfaces(m::AbstractMesh) = sum(length.(m.bdfaces))
nbdfaces(m::AbstractMesh, i) = length(m.bdfaces[i])
nperiodic(m::AbstractMesh) = length(m.periodic)
nvertices(m::AbstractMesh) = length(m.nodes)
nvertices(e::MeshElement) = length(e.nodeinds)
nvertices(f::MeshFace) = lengt(f.nodeinds)

elements(m::AbstractMesh) = LazyRows(m.elements)
faces(m::AbstractMesh) = LazyRows(m.faces)
intfaces(m::AbstractMesh) = LazyRows(m.faces[m.intfaces])
bdfaces(m::AbstractMesh) = LazyRows(m.faces[vcat(m.bdfaces...)])
bdfaces(m::AbstractMesh, i) = LazyRows(m.faces[m.bdfaces[i]])
periodic(m::AbstractMesh) = m.periodic
vertices(m::AbstractMesh) = m.nodes

element(m::AbstractMesh, i) = LazyRow(m.elements, i)
face(m::AbstractMesh, i) = LazyRow(m.faces, i)
intface(m::AbstractMesh, i) = LazyRow(m.faces, m.intfaces[i])
bdface(m::AbstractMesh, ib, i) = LazyRow(m.faces, m.bdfaces[ib][i])
periodic(m::AbstractMesh, i) = m.periodic[i]
vertex(m::AbstractMesh, i) = m.nodes[i]

eachelement(m::AbstractMesh) = Base.OneTo(nelements(m))
eachboundary(m::AbstractMesh) = Base.OneTo(nboundaries(m))
eachface(m::AbstractMesh) = Base.OneTo(nfaces(m))
eachintface(m::AbstractMesh) = m.intfaces
eachbdface(m::AbstractMesh, i) = m.bdfaces[i]
eachbdface(m::AbstractMesh) = vcat(m.bdfaces...)
eachvertex(m::AbstractMesh) = Base.OneTo(nvertices(m))

function apply_periodicBCs! end

abstract type AbstractMapping end

struct PointMapping <: AbstractMapping end
struct SegmentMapping <: AbstractMapping end
struct TriLinearMapping <: AbstractMapping end
struct QuadLinearMapping <: AbstractMapping end
struct HexLinearMapping <: AbstractMapping end

function coords(ξ, mesh::AbstractMesh, ie)
    elem = element(mesh, ie)
    mapping = mesh.mappings[elem.mapind]
    nodes = vertices(mesh)[elem.nodeinds]
    return coords(ξ, nodes, mapping)
end

function coords(ξ::AbstractVector, nodes::AbstractVector, ::PointMapping)
    return SVector{length(nodes[1])}(nodes[1])
end

function coords(ξ::AbstractVector, nodes::AbstractVector, ::SegmentMapping)
    ξrel = (ξ[1] + 1) / 2
    x = nodes[1] .* (1 - ξrel) .+ nodes[2] .* ξrel
    return x |> SVector{length(nodes[1])}
end

function coords(ξ::AbstractVector, nodes::AbstractVector, ::QuadLinearMapping)
    ξrel = SVector{2}((ξ .+ 1) ./ 2)
    x = nodes[1] .* (1 - ξrel[1]) .* (1 - ξrel[2]) .+
        nodes[2] .* ξrel[1] .* (1 - ξrel[2]) .+
        nodes[3] .* ξrel[1] .* ξrel[2] .+
        nodes[4] .* (1 - ξrel[1]) .* ξrel[2]
    return x |> SVector{length(nodes[1])}
end

function coords(ξ::AbstractVector, nodes::AbstractVector, ::TriLinearMapping)
    error("Not implemented yet!")
end

function coords(ξ::AbstractVector, nodes::AbstractVector, ::HexLinearMapping)
    error("Not implemented yet!")
end

include("CartesianMesh.jl")
