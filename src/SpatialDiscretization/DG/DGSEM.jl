struct DGSEM{EQ,RT,NF,MT,ST,PE,PF,BCT,FT} <: DiscontinuousGalerkin{EQ,RT}
    equation::EQ
    riemannsolver::NF
    mesh::MT
    stdvec::ST
    dofhandler::DofHandlerDG
    physelem::PE
    physface::PF
    Qf::MortarStateVector{RT}
    Fn::MortarStateVector{RT}
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
            "$(length(stdvec)) quadratures were given."
        )
    )
    _stdvec = tuple(stdvec)

    # Dof handler
    if length(_stdvec) == 1
        dofhandler = DofHandlerDG([nelements(mesh)])
    else
        nelems = (nelements(mesh, i) for i in eachregion(mesh))
        dofhandler = DofHandlerDG(nelems)
    end

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

    # Physical elements
    subgrid = requires_subgrid.(equation.operators) |> any
    physelements, physfaces = compute_metric_terms(_stdvec, dofhandler, mesh, subgrid)

    # Faces storage
    Qf = MortarStateVector{RT}(undef, mesh, _stdvec, dofhandler, nvariables(equation))
    Fn = MortarStateVector{RT}(undef, mesh, _stdvec, dofhandler, nvariables(equation))

    # Source term
    sourceterm = if isnothing(source)
        (dQ, Q, x, t, region) -> nothing
    else
        source
    end

    return DGSEM(
        equation,
        riemannsolver,
        mesh,
        _stdvec,
        dofhandler,
        physelements,
        physfaces,
        Qf,
        Fn,
        _bcs,
        sourceterm,
    )
end

function ndofs(dg::DGSEM)
    n = 0
    for i in eachregion(dg.dofhandler)
        n += nelements(dg.dofhandler, i) * ndofs(dg.stdvec[i])
    end
    return n
end

function Base.show(io::IO, m::MIME"text/plain", dg::DGSEM{EQ,RT}) where {EQ,RT}
    @nospecialize

    # Header
    println(io, "=========================================================================")
    println(io, "DGSEM{", RT, "} spatial discretization with ",
        nregions(dg.dofhandler), " region(s):")

    # Equation
    println(io, "\nEquation:")
    println(io, "---------")
    show(io, m, dg.equation)

    # TODO: numerical flux

    # Standard regions
    println(io, "\n\nStandard regions:")
    println(io, "-----------------")
    for i in eachregion(dg.dofhandler)
        show(io, m, dg.stdvec[i])
        println(io, "")
    end

    # Mesh
    println(io, "\nMesh:")
    println(io, "-----")
    show(io, m, dg.mesh)
    print(io, "\n=========================================================================")
end
