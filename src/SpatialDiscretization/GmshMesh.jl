struct UnstructuredMesh{ND,RT,FV,EV,MT} <: AbstractMesh{ND,RT}
    nelements::Int
    nodes::Vector{SVector{ND,RT}}
    faces::FV
    elements::EV
    intfaces::Vector{Int}
    bdfaces::Vector{Vector{Int}}
    bdmap::Dict{Int,Int}
    regionmap::Vector{Int}
    periodic::Dict{Int,Int}
    mappings::MT
end

function UnstructuredMesh{ND,RT}() where {ND,RT}
    # Get nodes
    _, nodes, _ = gmsh.model.mesh.getNodes()
    nodes = [SVector(RT.(node[1:ND])) for node in eachcol(reshape(nodes, (3, :)))]

    # Get elements
    etypes, etags, enodes = gmsh.model.mesh.getElements(ND)

    length(etypes) == 1 || throw(ArgumentError("Hybrid meshes are not supported."))
    etype = etypes[1]
    etags = etags[1]
    enodes = enodes[1]
    nelements = length(etags)
    nodeinds = Vector{Int}[]; sizehint!(nodeinds, nelements)
    regionmap = Int[]; sizehint!(regionmap, nelements)

    # Get properties
    typename, _, _, nnodes, _, _ = gmsh.model.mesh.getElementProperties(etype)

    # Check the type
    if ND == 1
        lowercase(typename) == "line" ||
            throw(ArgumentError("In 1D, all elements must be lines."))
    elseif ND == 2
        lowercase(typename) == "quadrangle" ||
            throw(ArgumentError("In 2D, all elements must be quadrangles."))
    else # ND == 3
        lowercase(typename) == "hexaedron" ||
            throw(ArgumentError("In 3D, all elements must be hexaedrons."))
    end

    # Get the corresponding node IDs
    cnt = 0
    for _ in eachindex(etags)
        push!(nodeinds, enodes[(cnt + 1):(cnt + nnodes)])
        push!(regionmap, 1)
        cnt += nnodes
    end

    # Construct gmsh faces
    gmsh.model.mesh.createFaces()
    fnodes, _, _ = gmsh.model.mesh.getNodes(ND - 1)
    ftags, orientations = gmsh.model.mesh.getFaces(ND^2, fnodes)

    # TODO
end

function UnstructuredMesh{ND,RT}(filename) where {ND,RT}
    gmsh.initialize()
    gmsh.open(filename)
    mesh = UnstructuredMesh{ND,RT}()
    gmsh.finalize()
    return mesh
end

