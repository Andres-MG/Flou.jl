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
    nq,
    qtype,
    equation,
    bcs,
    riemannsolver,
    source=nothing,
) where {
    ND,
    RT,
}
    # Standard element
    std = if ND == 1
        StdSegment{RT}(nq, qtype)
    elseif ND == 2
        StdQuad{RT}(nq, qtype)
    elseif ND == 3
        error("Not implemented yet!")
    end
    stdvec = (std,)  # TODO: single-region discretization
    dofhandler = DofHandlerDG([nelements(mesh)])

    # Boundary conditions
    nbounds = nboundaries(mesh)
    length(bcs) == nbounds ||
        throw(ArgumentError("The number of BCs does not match the number of boundaries."))
    _bcs = Vector{Any}(undef, nbounds)
    for (key, values) in bcs
        i = mesh.bdmap[key]
        _bcs[i] = values
    end
    _bcs = Tuple(_bcs)

    # Physical elements
    subgrid = requires_subgrid.(equation.operators) |> any
    physelements, physfaces = compute_metric_terms(stdvec, dofhandler, mesh, subgrid)

    # Faces storage
    Qf = MortarStateVector{RT}(undef, mesh, stdvec, dofhandler, nvariables(equation))
    Fn = MortarStateVector{RT}(undef, mesh, stdvec, dofhandler, nvariables(equation))

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
        stdvec,
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
