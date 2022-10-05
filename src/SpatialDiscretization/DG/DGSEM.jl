struct DGSEM{EQ,RT,NF,MT,ST,G,R,BCT,FT} <: DiscontinuousGalerkin{EQ,RT}
    equation::EQ
    riemannsolver::NF
    mesh::MT
    stdvec::ST
    dofhandler::DofHandler
    geometry::G
    Qf::FaceStateVector{RT,R}
    Fn::FaceStateVector{RT,R}
    bcs::BCT
    source!::FT
end

function DGSEM(
    mesh::AbstractMesh{ND,RT},
    stdvec,
    equation,
    bcs,
    riemannsolver,
    source=nothing,
) where {
    ND,
    RT,
}
    # Standard elements
    (isa(stdvec, AbstractStdRegion) || length(stdvec) == nregions(mesh)) || throw(
        ArgumentError(
            "The mesh has $(nregions(mesh)) region(s) but " *
            "$(length(stdvec)) different standard regions were given."
        )
    )
    if isa(stdvec, AbstractStdRegion)
        _stdvec = ntuple(_ -> stdvec, nregions(mesh))
    else
        _stdvec = tuple(stdvec)
    end

    elem2std = [get_region(mesh, ie) for ie in 1:nelements(mesh)]

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
    dofhandler = DofHandler(mesh, _stdvec, elem2std)

    # Physical elements
    subgrid = requires_subgrid.(equation.operators) |> any
    geometry = Geometry(_stdvec, dofhandler, mesh, subgrid)

    # Faces storage
    Qf = FaceStateVector{RT}(undef, nvariables(equation), dofhandler)
    Fn = FaceStateVector{RT}(undef, nvariables(equation), dofhandler)

    # Source term
    sourceterm = if isnothing(source)
        (dQ, Q, x, t) -> nothing
    else
        source
    end

    return DGSEM(
        equation,
        riemannsolver,
        mesh,
        _stdvec,
        dofhandler,
        geometry,
        Qf,
        Fn,
        _bcs,
        sourceterm,
    )
end

nvariables(dg::DGSEM) = nvariables(dg.equation)
nelements(dg::DGSEM) = nelements(dg.dofhandler)
nfaces(dg::DGSEM) = nfaces(dg.dofhandler)
ndofs(dg::DGSEM) = ndofs(dg.dofhandler)

eachvariable(dg::DGSEM) = eachvariable(dg.equation)
eachelement(dg::DGSEM) = eachelement(dg.dofhandler)
eachface(dg::DGSEM) = eachface(dg.dofhandler)
eachdof(dg::DGSEM) = eachdof(dg.dofhandler)

@inline function get_std(dg::DGSEM, ie)
    return @inbounds dg.stdvec[get_stdid(dg.dofhandler, ie)]
end

function Base.show(io::IO, m::MIME"text/plain", dg::DGSEM{EQ,RT}) where {EQ,RT}
    @nospecialize

    # Header
    println(io, "=========================================================================")
    println(io, "DGSEM{", RT, "} spatial discretization")

    # Equation
    println(io, "\nEquation:")
    println(io, "---------")
    show(io, m, dg.equation)

    # Riemann solver
    println(io, "\n\nRiemann solver:")
    println(io, "---------------")
    show(io, m, dg.riemannsolver)

    # Standard regions
    println(io, "\n\nStandard regions:")
    println(io, "-----------------")
    for std in dg.stdvec
        show(io, m, std)
        println(io, "\n")
    end

    # Mesh
    println(io, "Mesh:")
    println(io, "-----")
    show(io, m, dg.mesh)
    print(io, "\n=========================================================================")
end
