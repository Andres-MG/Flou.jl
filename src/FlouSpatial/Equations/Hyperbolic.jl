struct HyperbolicCache{NV,RT} <: AbstractCache
    Qf::FaceStateVector{NV,RT}
    Fn::FaceStateVector{NV,RT}
end

function construct_cache(
    ftype::Type,
    dofhandler::DofHandler,
    equation::HyperbolicEquation,
)
    Qf = FaceStateVector{nvariables(equation)}(undef, dofhandler, ftype)
    Fn = FaceStateVector{nvariables(equation)}(undef, dofhandler, ftype)
    return HyperbolicCache(Qf, Fn)
end

function FlouCommon.rhs!(
    _dQ::Matrix,
    _Q::Matrix,
    p::EquationConfig{<:MultielementDisc,<:HyperbolicEquation},
    time::Real,
)
    # Unpack
    (; disc, equation) = p
    cache = disc.cache

    dQ = GlobalStateVector{nvariables(equation)}(_dQ, disc.dofhandler)
    Q = GlobalStateVector{nvariables(equation)}(_Q, disc.dofhandler)

    # Restart time derivative
    fill!(dQ, zero(datatype(dQ)))

    # Project Q to faces
    project2faces!(cache.Qf, Q, disc)

    # Volume flux
    volume_contribution!(dQ, Q, disc, equation, disc.operators[1])

    # Boundary conditions
    applyBCs!(cache.Qf, disc, equation, time)

    # Interface fluxes
    interface_fluxes!(cache.Fn, cache.Qf, disc, equation, disc.operators[1].numflux)

    # Surface contribution
    surface_contribution!(dQ, Q, disc, equation, disc.operators[1])

    # Apply mass matrix
    apply_massmatrix!(dQ, disc)

    # Add source term
    apply_sourceterm!(dQ, Q, disc, time)

    return nothing
end

function volume_contribution!(dQ, Q, disc, equation::HyperbolicEquation, operator)
    @flouthreads for ie in eachelement(disc)
        @inbounds volume_contribution!(dQ, Q, ie, disc.std, disc, equation, operator)
    end

    return nothing
end

function surface_contribution!(dQ, Q, disc, equation::HyperbolicEquation, operator)
    @flouthreads for ie in eachelement(disc)
        @inbounds surface_contribution!(dQ, Q, ie, disc.std, disc, equation, operator)
    end
    return nothing
end
