struct DGSEM{RT,MT,ST,G,O,C<:DGcache{RT},BCT,FT} <: DiscontinuousGalerkin{RT}
    mesh::MT
    std::ST
    dofhandler::DofHandler
    geometry::G
    operators::O
    cache::C
    bcs::BCT
    source!::FT
end

function DGSEM(
    mesh::AbstractMesh{ND,RT},
    std,
    equation,
    operators,
    bcs,
    source=nothing,
) where {
    ND,
    RT,
}
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
    dofhandler = DofHandler(mesh, std)

    if operators isa AbstractOperator
        _operators = (operators,)
    else
        _operators = Tuple(operators)
    end

    # Physical elements
    subgrid = any(requires_subgrid.(_operators, Ref(std)))
    geometry = Geometry(std, dofhandler, mesh, subgrid)

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
        std,
        dofhandler,
        geometry,
        _operators,
        cache,
        _bcs,
        sourceterm,
    )
end

datatype(::DGSEM{RT}) where {RT} = RT
get_spatialdim(dg::DGSEM) = get_spatialdim(dg.mesh)

nelements(dg::DGSEM) = nelements(dg.dofhandler)
nfaces(dg::DGSEM) = nfaces(dg.dofhandler)
ndofs(dg::DGSEM) = ndofs(dg.dofhandler)

eachelement(dg::DGSEM) = eachelement(dg.dofhandler)
eachface(dg::DGSEM) = eachface(dg.dofhandler)
eachdof(dg::DGSEM) = eachdof(dg.dofhandler)

function Base.show(io::IO, m::MIME"text/plain", dg::DGSEM)
    @nospecialize

    rt = datatype(dg)

    # Header
    println(io, "=========================================================================")
    println(io, "DGSEM{", rt, "} spatial discretization")

    # Mesh
    println(io, "\nMesh:")
    println(io, "-----")
    show(io, m, dg.mesh)

    # Standard regions
    println(io, "\n\nStandard region:")
    println(io, "----------------")
    show(io, m, dg.std)

    # Operators
    println(io, "\n\nOperators:")
    println(io, "----------")
    for op in dg.operators
        show(io, m, op)
        print(io, "\n")
    end

    print(io, "=========================================================================")

    return nothing
end

function get_max_dt(
    q::Union{AbstractVector{<:SVector},AbstractMatrix{<:Number}},
    dg::DGSEM,
    eq::AbstractEquation,
    cfl::Real,
)
    Q = StateVector{nvariables(eq)}(q, dg.dofhandler)
    Δt = typemax(datatype(Q))
    std = dg.std
    d = get_spatialdim(dg)
    n = ndofs(std)

    for ie in eachelement(dg)
        Δx = dg.geometry.elements[ie].volume[] / n
        Δx = d == 1 ? Δx : (d == 2 ? sqrt(Δx) : cbrt(Δx))
        @inbounds for i in eachdof(std)
            Δt = min(Δt, get_max_dt(Q[ie][i], Δx, cfl, eq))
        end
    end

    return Δt
end

