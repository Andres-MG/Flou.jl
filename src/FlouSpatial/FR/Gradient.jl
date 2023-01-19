struct GradientFRcache{NV,RT,DQ,DF} <: AbstractCache
    Qf::FaceStateVector{NV,RT,DQ}
    Fn::FaceBlockVector{NV,RT,DF}
end

function construct_cache(
    disctype::Symbol,
    realtype::Type,
    dofhandler::DofHandler,
    eq::GradientEquation,
)
    if disctype == :fr
        Qf = FaceStateVector{nvariables(eq),realtype}(undef, dofhandler)
        Fn = FaceBlockVector{nvariables(eq),realtype}(undef, ndims(eq), dofhandler)
        return GradientFRcache(Qf, Fn)
    else
        @error "Unknown discretization type $(disctype)"
    end
end

function FlouCommon.rhs!(
    G::BlockVector,
    Q::StateVector,
    p::EquationConfig{<:FR,<:GradientEquation},
    _::Real,
)
    # Unpack
    (; disc, equation) = p
    cache = disc.cache

    # Restart gradients
    fill!(G, zero(eltype(G)))

    # Volume flux
    volume_contribution!(G, Q, disc, equation, disc.operators[1])

    # Project Q to faces
    project2faces!(cache.Qf, Q, disc)

    # Boundary conditions
    applyBCs!(cache.Qf, disc, equation, time)

    # Interface fluxes
    interface_fluxes!(cache.Fn, cache.Qf, disc, equation, disc.operators[1].numflux)

    # Surface contribution
    surface_contribution!(G, Q, cache.Fn, disc, equation, disc.operators[1])

    # Apply mass matrix
    apply_massmatrix!(G, disc)

    return nothing
end

function volume_contribution!(G, Q, dg, equation::GradientEquation, operator)
    @flouthreads for ie in eachelement(dg)
        @inbounds volume_contribution!(
            G.element[ie], Q.element[ie],
            ie, dg.std, dg, equation, operator,
        )
    end
    return nothing
end

function surface_contribution!(G, Q, Fn, dg, equation::GradientEquation, operator)
    @flouthreads for ie in eachelement(dg)
        @inbounds surface_contribution!(
            G.element[ie], Q.element[ie],
            Fn, ie, dg.std, dg, equation, operator,
        )
    end
    return nothing
end

function rotate2face(Qf, _, ::GradientEquation{ND,NV}) where {ND,NV}
    return SVector{NV}(Qf)
end

function rotate2phys(Qrot, _, ::GradientEquation{ND,NV}) where {ND,NV}
    return ntuple(d -> SVector{NV}(
            Qrot[d][v] for v in 1:NV
        ), ND)
end

function numericalflux(Ql, Qr, n, eq::GradientEquation, ::StdAverageNumericalFlux)
    nd = spatialdim(eq)
    nv = nvariables(eq)
    return ntuple(d -> SVector{nv}(
        (Ql[v] + Qr[v]) * n[d] / 2 for v in 1:nv
    ), nd)
end
