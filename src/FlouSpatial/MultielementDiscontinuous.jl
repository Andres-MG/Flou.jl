struct MultielementDisc{ND,RT,MT<:AbstractMesh{ND,RT},ST,G,O,C,BCT,M,FT} <:
        AbstractSpatialDiscretization{ND,RT}
    mesh::MT
    std::ST
    dofhandler::DofHandler
    geometry::G
    operators::O
    cache::C
    bcs::BCT
    mass::M
    source!::FT
end

function MultielementDisc(
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
    cache = construct_cache(RT, dofhandler, equation)

    # Mass matrix (include overintegration matrix)
    mass = BlockDiagonal(
        [factorize(std.Pover * Diagonal(elem.jac)) for elem in geometry.elements]
    )

    # Source term
    sourceterm = if isnothing(source)
        (_, _, _, _) -> nothing
    else
        source
    end

    return MultielementDisc(
        mesh,
        std,
        dofhandler,
        geometry,
        _operators,
        cache,
        _bcs,
        mass,
        sourceterm,
    )
end

FlouCommon.nelements(disc::MultielementDisc) = nelements(disc.dofhandler)
FlouCommon.nfaces(disc::MultielementDisc) = nfaces(disc.dofhandler)
ndofs(disc::MultielementDisc) = ndofs(disc.dofhandler)

FlouCommon.eachelement(disc::MultielementDisc) = eachelement(disc.dofhandler)
FlouCommon.eachface(disc::MultielementDisc) = eachface(disc.dofhandler)
eachdof(disc::MultielementDisc) = eachdof(disc.dofhandler)

function Base.show(io::IO, m::MIME"text/plain", disc::MultielementDisc)
    @nospecialize

    # Header
    println(io, "=========================================================================")
    println("Discontinuous multi-element spatial discretization")

    # Mesh
    println(io, "\nMesh:")
    println(io, "-----")
    show(io, m, disc.mesh)

    # Standard regions
    println(io, "\n\nStandard region:")
    println(io, "----------------")
    show(io, m, disc.std)

    # Operators
    println(io, "\n\nOperators:")
    println(io, "----------")
    for op in disc.operators
        show(io, m, op)
        print(io, "\n")
    end

    print(io, "=========================================================================")

    return nothing
end

function apply_massmatrix!(dQ, disc::MultielementDisc)
    @flouthreads for ie in eachelement(disc)
        ldiv!(blocks(disc.mass)[ie], dQ.elements[ie].dofs)
    end
    return nothing
end

function apply_sourceterm!(dQ, Q, disc::MultielementDisc, time)
    (; geometry, source!) = disc
    @flouthreads for i in eachdof(disc)
        @inbounds x = geometry.elements.coords[i]
        @inbounds source!(dQ.dofs[i], Q.dofs[i], x, time)
    end
    return nothing
end

function integrate(_f::AbstractVector, disc::MultielementDisc)
    f = GlobalStateVector(_f, disc.dofhandler)
    return integrate(f, disc, 1)
end

function integrate(f::StateVector, disc::MultielementDisc, ivar::Integer=1)
    (; geometry) = disc
    integral = zero(datatype(f))
    @flouthreads for ie in eachelement(disc)
        @inbounds integral += integrate(f.elements[ie].vars[ivar], geometry.elements[ie])
    end
    return integral
end

function FlouCommon.get_max_dt(q, disc::MultielementDisc, eq::AbstractEquation, cfl)
    Q = GlobalStateVector{nvariables(eq)}(q, disc.dofhandler)
    Δt = typemax(datatype(Q))
    (; std, geometry) = disc
    d = spatialdim(disc)
    n = ndofs(std)

    for ie in eachelement(disc)
        Δx = geometry.elements[ie].volume / n
        Δx = d == 1 ? Δx : (d == 2 ? sqrt(Δx) : cbrt(Δx))
        @inbounds for i in eachdof(std)
            @inbounds Δt = min(Δt, get_max_dt(Q.elements[ie].dofs[i], Δx, cfl, eq))
        end
    end

    return Δt
end

include("IO.jl")
include("Interfaces.jl")

# Operators
include("Equations/OpDivergence.jl")
include("Equations/OpGradient.jl")

# Equations
include("Equations/Hyperbolic.jl")
include("Equations/LinearAdvection.jl")
include("Equations/Burgers.jl")
include("Equations/KPP.jl")
include("Equations/Euler.jl")

include("Equations/Gradient.jl")
