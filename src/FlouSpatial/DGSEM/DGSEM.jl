# Standard regions
include("StdRegions.jl")

struct DGSEM{ND,RT,MT<:AbstractMesh{ND,RT},ST,G,O,C,BCT,M,FT} <: HighOrderElements{ND,RT}
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
    subgrid = any(dg_requires_subgrid.(_operators, Ref(std)))
    geometry = Geometry(std, dofhandler, mesh, subgrid)

    # Equation cache
    cache = construct_cache(:dgsem, RT, dofhandler, equation)

    # Mass matrix
    mass = BlockDiagonal([
        massmatrix(std, elem.jac) |> factorize for elem in geometry.elements
    ])

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
        mass,
        sourceterm,
    )
end

FlouCommon.nelements(dg::DGSEM) = nelements(dg.dofhandler)
FlouCommon.nfaces(dg::DGSEM) = nfaces(dg.dofhandler)
ndofs(dg::DGSEM) = ndofs(dg.dofhandler)

FlouCommon.eachelement(dg::DGSEM) = eachelement(dg.dofhandler)
FlouCommon.eachface(dg::DGSEM) = eachface(dg.dofhandler)
eachdof(dg::DGSEM) = eachdof(dg.dofhandler)

function Base.show(io::IO, m::MIME"text/plain", dg::DGSEM)
    @nospecialize

    # Header
    println(io, "=========================================================================")
    println(io, "DGSEM spatial discretization")

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

function project2faces!(Qf, Q, dg::DGSEM)
    # Unpack
    (; mesh, std) = dg

    @flouthreads for ie in eachelement(dg)
        iface = mesh.elements[ie].faceinds
        facepos = mesh.elements[ie].facepos
        @inbounds for (s, (face, pos)) in enumerate(zip(iface, facepos))
            mul!(Qf.face[face][pos], std.l[s], Q.element[ie])
        end
    end
    return nothing
end

function apply_massmatrix!(dQ, dg::DGSEM)
    @flouthreads for ie in eachelement(dg)
        ldiv!(blocks(dg.mass)[ie], dQ.element[ie])
    end
    return nothing
end

function FlouCommon.get_max_dt(
    q::Union{AbstractVector{<:SVector},AbstractMatrix{<:Number}},
    dg::DGSEM,
    eq::AbstractEquation,
    cfl::Real,
)
    Q = StateVector{nvariables(eq)}(q, dg.dofhandler)
    Δt = typemax(eltype(Q))
    std = dg.std
    d = spatialdim(dg)
    n = ndofs(std)

    for ie in eachelement(dg)
        Δx = dg.geometry.elements[ie].volume[] / n
        Δx = d == 1 ? Δx : (d == 2 ? sqrt(Δx) : cbrt(Δx))
        @inbounds for i in eachdof(std)
            Δt = min(Δt, get_max_dt(Q.element[ie][i], Δx, cfl, eq))
        end
    end

    return Δt
end

# Operators
include("OpDivergence.jl")
include("OpGradient.jl")

# Equations
include("Hyperbolic.jl")
include("LinearAdvection.jl")
include("Burgers.jl")
include("KPP.jl")
include("Euler.jl")

include("Gradient.jl")
