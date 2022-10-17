struct DGSEM{RT,MT,ST,G,O,C<:DGcache{RT},BCT,FT} <: DiscontinuousGalerkin{RT}
    mesh::MT
    stdvec::ST
    dofhandler::DofHandler
    geometry::G
    operators::O
    cache::C
    bcs::BCT
    source!::FT
end

function DGSEM(
    mesh::AbstractMesh{ND,RT},
    stdvec,
    equation,
    operators,
    bcs,
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
        _stdvec = Tuple(stdvec)
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

    if operators isa AbstractOperator
        _operators = (operators,)
    else
        _operators = Tuple(operators)
    end

    # Physical elements
    subgrid = false
    for std in _stdvec
        subgrid |= any(requires_subgrid.(_operators, Ref(std)))
    end
    geometry = Geometry(_stdvec, dofhandler, mesh, subgrid)

    # Equation cache
    cache = construct_cache(:dgsem, RT, dofhandler, equation)

    # Source term
    sourceterm = if isnothing(source)
        (_, _, _, _) -> nothing
    else
        source
    end

    return DGSEM(
        mesh,
        _stdvec,
        dofhandler,
        geometry,
        _operators,
        cache,
        _bcs,
        sourceterm,
    )
end

get_spatialdim(dg::DGSEM) = get_spatialdim(dg.mesh)

nelements(dg::DGSEM) = nelements(dg.dofhandler)
nfaces(dg::DGSEM) = nfaces(dg.dofhandler)
ndofs(dg::DGSEM) = ndofs(dg.dofhandler)

eachelement(dg::DGSEM) = eachelement(dg.dofhandler)
eachface(dg::DGSEM) = eachface(dg.dofhandler)
eachdof(dg::DGSEM) = eachdof(dg.dofhandler)

@inline function get_std(dg::DGSEM, ie)
    return @inbounds dg.stdvec[get_stdid(dg.dofhandler, ie)]
end

function Base.show(io::IO, m::MIME"text/plain", dg::DGSEM{RT}) where {RT}
    @nospecialize

    # Header
    println(io, "=========================================================================")
    println(io, "DGSEM{", RT, "} spatial discretization")

    # Mesh
    println(io, "\nMesh:")
    println(io, "-----")
    show(io, m, dg.mesh)

    # Standard regions
    println(io, "\n\nStandard regions:")
    println(io, "-----------------")
    for std in dg.stdvec
        show(io, m, std)
        println(io, "\n")
    end

    # Operators
    println(io, "Operators:")
    println(io, "----------")
    for op in dg.operators
        show(io, m, op)
        print(io, "\n")
    end

    print(io, "=========================================================================")

    return nothing
end

function get_max_dt(q::AbstractMatrix, dg::DGSEM, eq::AbstractEquation, cfl::Real)
    Q = StateVector(q, dg.dofhandler)
    Δt = typemax(eltype(q))

    for ie in eachelement(dg)
        std = get_std(dg, ie)
        d = get_spatialdim(dg)
        Δx = dg.geometry.elements[ie].volume[] / ndofs(std)
        Δx = d == 1 ? Δx : (d == 2 ? sqrt(Δx) : cbrt(Δx))

        @inbounds for i in eachdof(std)
            Δt = min(Δt, get_max_dt(view(Q[ie], i, :), Δx, cfl, eq))
        end
    end

    return Δt
end

