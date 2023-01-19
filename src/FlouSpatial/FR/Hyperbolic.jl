struct HyperbolicDGcache{NV,RT,D} <: AbstractCache
    Qf::FaceStateVector{NV,RT,D}
    Fn::FaceStateVector{NV,RT,D}
end

function construct_cache(
    disctype::Symbol,
    realtype::Type,
    dofhandler::DofHandler,
    equation::HyperbolicEquation,
)
    if disctype == :fr
        Qf = FaceStateVector{nvariables(equation),realtype}(undef, dofhandler)
        Fn = FaceStateVector{nvariables(equation),realtype}(undef, dofhandler)
        return HyperbolicDGcache(Qf, Fn)
    else
        @error "Unknown discretization type $(disctype)"
    end
end

function FlouCommon.rhs!(
    dQ::StateVector,
    Q::StateVector,
    p::EquationConfig{<:FR,<:HyperbolicEquation},
    time::Real,
)
    # Unpack
    (; disc, equation) = p
    cache = disc.cache

    # Restart time derivative
    fill!(dQ, zero(eltype(dQ)))

    # Volume flux
    volume_contribution!(dQ, Q, disc, equation, disc.operators[1])

    # Project Q to faces
    project2faces!(cache.Qf, Q, disc)

    # Boundary conditions
    applyBCs!(cache.Qf, disc, equation, time)

    # Interface fluxes
    interface_fluxes!(cache.Fn, cache.Qf, disc, equation, disc.operators[1].numflux)

    # Surface contribution
    surface_contribution!(dQ, Q, cache.Fn, disc, equation, disc.operators[1])

    # Apply mass matrix
    apply_massmatrix!(dQ, disc)

    # Add source term
    apply_sourceterm!(dQ, Q, disc, time)

    return nothing
end

function volume_contribution!(dQ, Q, dg, equation::HyperbolicEquation, operator)
    @flouthreads for ie in eachelement(dg)
        @inbounds volume_contribution!(
            dQ.element[ie], Q.element[ie],
            ie, dg.std, dg, equation, operator,
        )
    end
    return nothing
end

function surface_contribution!(dQ, Q, Fn, dg, equation::HyperbolicEquation, operator)
    @flouthreads for ie in eachelement(dg)
        @inbounds surface_contribution!(
            dQ.element[ie], Q.element[ie],
            Fn, ie, dg.std, dg, equation, operator,
        )
    end
    return nothing
end

