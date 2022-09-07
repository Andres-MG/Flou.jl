struct UnstructuredMesh{ND,RT,FV,EV,MT} <: AbstractMesh{ND,RT}
    nelements::Int
    nodes::Vector{SVector{ND,RT}}
    faces::FV
    elements::EV
    intfaces::Vector{Int}
    bdfaces::Vector{Vector{Int}}
    bdnames::Vector{String}
    bdmap::Dict{Int,Int}
    regionmap::Vector{Int}
    periodic::Dict{Int,Int}
    mappings::MT
end

nregions(::UnstructuredMesh) = 1
get_region(m::UnstructuredMesh, i) = m.regionmap[i]
eachregion(m::UnstructuredMesh) = Base.OneTo(nregions(m))

nelements(m::UnstructuredMesh, i) = count(prod(m.nelements[i]))

function UnstructuredMesh{ND,RT}(filename) where {ND,RT}
    gmsh.initialize()
    gmsh.open(filename)
    mesh = UnstructuredMesh{ND,RT}()
    gmsh.finalize()
    return mesh
end

function UnstructuredMesh{ND,RT}() where {ND,RT}
    # Get nodes
    _, _nodes, _ = gmsh.model.mesh.get_nodes()
    numnodes = length(_nodes) ÷ 3
    nodes = Vector{SVector{ND,RT}}(undef, numnodes)
    cnt = 0
    for i in 1:numnodes
        nodes[i] = SVector{ND}(RT.(_nodes[(cnt + 1):(cnt + ND)]))
        cnt += 3
    end

    # Get elements
    etypes, etags, enodes = gmsh.model.mesh.get_elements(ND)

    length(etypes) == 1 || throw(ArgumentError("Hybrid meshes are not supported."))
    etype = etypes[1]
    etags = etags[1]
    enodes = enodes[1]
    numelements = length(etags)

    # Get properties
    typename, _, _, nnodes, _, _ = gmsh.model.mesh.get_element_properties(etype)

    # Check the type
    if ND == 1
        occursin("Line", typename) ||
            throw(ArgumentError("In 1D, all elements must be lines."))
    elseif ND == 2
        occursin("Quadrilateral", typename) ||
            throw(ArgumentError("In 2D, all elements must be quadrilaterals."))
    else # ND == 3
        occursin("Hexahedron", typename) ||
            throw(ArgumentError("In 3D, all elements must be hexahedrons."))
    end

    # Get the corresponding node IDs
    elemnodes = Vector{Vector{UInt}}(undef, numelements)
    regionmap = Vector{Int}(undef, numelements)
    cnt = 0
    for i in 1:numelements
        elemnodes[i] = enodes[(cnt + 1):(cnt + nnodes)]
        regionmap[i] = 1
        cnt += nnodes
    end

    # Get faces
    faceinds, eleminds, facenodes, orientation = if ND == 1
        _facemap_1d(etype, numelements)
    elseif ND == 2
        _facemap_2d(etype, numelements)
    else # ND == 3
        _facemap_3d(etype, numelements)
    end

    numfaces = length(eleminds)
    intfaces = Int[]
    for (i, face) in eleminds
        if face[2] != 0
            push!(intfaces, i)
        end
    end
    sort!(intfaces)

    # Boundaries
    bdnames = String[]
    bdfaces = Vector{Int}[]
    phys = gmsh.model.get_physical_groups(ND - 1)
    for (i, group) in enumerate(phys)
        push!(bdnames, gmsh.model.get_physical_name(group[1], group[2]))
        push!(bdfaces, UInt[])
        entities = gmsh.model.get_entities_for_physical_group(group[1], group[2])
        for entity in entities
            _, ftags, _ = gmsh.model.mesh.get_elements(group[1], entity)
            sort!(ftags[1])
            append!(bdfaces[i], ftags[1])
        end
    end
    bdmap = Dict((i, i) for i in 1:length(bdfaces))

    # Construct elements and faces
    pos = Vector{UInt}(undef, 2ND)
    elements = Vector{MeshElement}(undef, numelements)
    for i in 1:numelements
        for (j, iface) in enumerate(faceinds[i])
            pos[j] = findfirst(==(i), eleminds[iface])
        end
        elements[i] = MeshElement(elemnodes[i], faceinds[i], pos, 1)
    end

    pos = Vector{UInt}(undef, 2)
    faces = Vector{MeshFace}(undef, numfaces)
    for i in 1:numfaces
        for (j, ielem) in enumerate(eleminds[i])
            if ielem == 0
                pos[j] = 0
            else
                pos[j] = findfirst(==(i), faceinds[ielem])
            end
        end
        faces[i] = MeshFace(facenodes[i], eleminds[i], pos, orientation[i], 2)
    end

    mappings = if ND == 1
        (SegmentLinearMapping(), PointMapping())
    elseif ND == 2
        (QuadLinearMapping(), SegmentLinearMapping())
    else # ND == 3
        (HexLinearMapping(), QuadLinearMapping())
    end

    return UnstructuredMesh(
        numelements,
        nodes,
        StructVector(faces),
        StructVector(elements),
        intfaces,
        bdfaces,
        bdnames,
        bdmap,
        regionmap,
        Dict{Int,Int}(),
        mappings,
    )
end

function Base.show(io::IO, ::MIME"text/plain", m::UnstructuredMesh{ND,RT}) where {ND,RT}
    @nospecialize
    # Header
    println(io, ND, "D UnstructuredMesh{", RT, "}:")

    # Box limits
    dirs = ("x", "y", "z")
    lims = (
        mapreduce(x -> x[1], min, get_vertices(m)),
        mapreduce(x -> x[1], max, get_vertices(m))
    )
    print(io, " Bounding box: x ∈ [", lims[1], ", ", lims[2], "]")
    for idim in 2:ND
        lims = (
            mapreduce(x -> x[idim], min, get_vertices(m)),
            mapreduce(x -> x[idim], max, get_vertices(m))
        )
        print(io, ", ", dirs[idim], " ∈ [", lims[1], ", ", lims[2], "]")
    end
    println(io)

    # Number of elements
    println(io, " Number of elements: ", nelements(m))

    # Number of faces
    print(io, " Number of faces: ", nfaces(m))

    # Periodic boundaries
    return nothing
end

function apply_periodicBCs!(mesh::UnstructuredMesh, BCs::Pair{String,String}...)
    # Mapped boundaries
    intbcs = Dict{Int,Int}()

    # Check pairs
    for bc in BCs
        bd1 = findfirst(==(bc.first), mesh.bdnames)
        bd2 = findfirst(==(bc.second), mesh.bdnames)
        !isnothing(bd1) && !isnothing(bd2) || throw(ArgumentError(
            "Boundaries $(bc.first) and $(bc.second) do not exist."
        ))
        intbcs[bd1] = bd2
    end

    return _apply_periodicBCs!(mesh, intbcs)
end

function _facemap_1d(_, nelems)
    _, _, enodes = gmsh.model.mesh.get_elements(1)
    enodes = enodes[1]

    # Nodes for each element
    element2node = Dict{UInt,Vector{UInt}}()
    cnt::UInt = 0
    for i in 1:nelems
        element2node[i] = [
            enodes[cnt + 1],
            enodes[cnt + 2],
        ]
        cnt += 2
    end

    # Elements, nodes and orientations for each node
    node2element = Dict{UInt,Vector{UInt}}()
    node2node = Dict{UInt,Vector{UInt}}()
    orientations = zeros(Int32, nelems + 1)
    for (i, enode) in enumerate(enodes)
        ielem = (i - 1) ÷ 2 + 1
        if enode in keys(node2element)
            node2element[enode][2] = ielem
        else
            node2element[enode] = [ielem, 0]
            node2node[enode] = [enode]
        end
    end

    return (element2node, node2element, node2node, orientations)
end

function _facemap_2d(etype, nelems)
    gmsh.model.mesh.create_edges()
    ntags = gmsh.model.mesh.get_element_edge_nodes(etype)
    etags, _ = gmsh.model.mesh.get_edges(ntags)

    # Edges for each element
    element2edge = Dict{UInt,Vector{UInt}}()
    cnt::UInt = 0
    for i in 1:nelems
        element2edge[i] = [
            etags[cnt + 4],
            etags[cnt + 2],
            etags[cnt + 1],
            etags[cnt + 3],
        ]
        cnt += 4
    end

    # Elements and nodes for each edge
    nodemap = (
        UInt.([1, 2]),
        UInt.([1, 2]),
        UInt.([2, 1]),
        UInt.([2, 1]),
    )
    edge2element = Dict{UInt,Vector{UInt}}()
    edge2node = Dict{UInt,Vector{UInt}}()
    e2n_second = Dict{UInt,Vector{UInt}}()
    cnt = 0
    for (i, etag) in enumerate(etags)
        ielem = (i - 1) ÷ 4 + 1
        pos = (i - 1) % 4 + 1
        if etag in keys(edge2element)
            edge2element[etag][2] = ielem
            e2n_second[etag] = ntags[cnt .+ nodemap[pos]]
        else
            edge2element[etag] = [ielem, 0]
            edge2node[etag] = ntags[cnt .+ nodemap[pos]]
        end
        cnt += 2
    end

    # Orientations from the API are not useful
    orientations = zeros(Int32, length(edge2node))
    for (iedge, nodes2) in e2n_second
        node = edge2node[iedge][1]
        pos = findfirst(==(node), nodes2)
        if pos == 1
            orientations[iedge] = 0
        else # pos == 2
            orientations[iedge] = 1
        end
    end

    return (element2edge, edge2element, edge2node, orientations)
end

function _facemap_3d(etype, nelems)
    gmsh.model.mesh.create_faces()
    ntags = gmsh.model.mesh.get_element_face_nodes(etype, 4)
    ftags, _ = gmsh.model.mesh.get_faces(4, ntags)

    # Faces for each element
    element2face = Dict{UInt,Vector{UInt}}()
    cnt::UInt = 0
    for i in 1:nelems
        element2face[i] = [
            ftags[cnt + 3],
            ftags[cnt + 4],
            ftags[cnt + 2],
            ftags[cnt + 5],
            ftags[cnt + 1],
            ftags[cnt + 6],
        ]
        cnt += 6
    end

    # Elements and nodes for each face
    nodemap = (
        UInt.([1, 4, 3, 2]),
        UInt.([1, 2, 3, 4]),
        UInt.([1, 4, 3, 2]),
        UInt.([1, 2, 3, 4]),
        UInt.([2, 1, 4, 3]),
        UInt.([1, 2, 3, 4]),
    )
    face2element = Dict{UInt,Vector{UInt}}()
    face2node = Dict{UInt,Vector{UInt}}()
    f2n_second = Dict{UInt,Vector{UInt}}()
    cnt = 0
    for (i, ftag) in enumerate(ftags)
        ielem = (i - 1) ÷ 6 + 1
        pos = (i - 1) % 6 + 1
        if ftag in keys(face2element)
            face2element[ftag][2] = ielem
            f2n_second[ftag] = ntags[cnt .+ nodemap[pos]]
        else
            face2element[ftag] = [ielem, 0]
            face2node[ftag] = ntags[cnt .+ nodemap[pos]]
        end
        cnt += 4
    end

    # Orientations are not implemented in GMSH yet
    # https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/src/common/gmsh.cpp#L3485
    orientations = zeros(Int32, length(face2node))
    for (iface, nodes2) in f2n_second
        n1 = face2node[iface][1]
        n2 = face2node[iface][2]
        pos1 = findfirst(==(n1), nodes2)
        pos2 = findfirst(==(n2), nodes2)
        if pos1 == 1
            if pos2 == 2
                orientations[iface] = 0
            else # pos2 == 4
                orientations[iface] = 4
            end
        elseif pos1 == 2
            if pos2 == 3
                orientations[iface] = 1
            else # pos2 == 1
                orientations[iface] = 5
            end
        elseif pos1 == 3
            if pos2 == 4
                orientations[iface] = 2
            else # pos2 == 2
                orientations[iface] = 6
            end
        else # pos1 == 4
            if pos2 == 1
                orientations[iface] = 3
            else # pos2 == 3
                orientations[iface] = 7
            end
        end
    end

    return (element2face, face2element, face2node, orientations)
end

