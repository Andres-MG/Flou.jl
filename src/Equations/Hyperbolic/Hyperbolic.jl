#==========================================================================================#
#                                           DGSEM                                          #

struct HyperbolicDGcache{NV,RT,D,T} <: DGcache{RT}
    Qf::FaceStateVector{NV,RT,D,T}
    Fn::FaceStateVector{NV,RT,D,T}
end

function construct_cache(disctype, realtype, dofhandler, equation::HyperbolicEquation)
    if disctype == :dgsem
        Qf = FaceStateVector{nvariables(equation),realtype}(undef, dofhandler)
        Fn = FaceStateVector{nvariables(equation),realtype}(undef, dofhandler)
        return HyperbolicDGcache(Qf, Fn)
    else
        @error "Unknown discretization type $(disctype)"
    end
end

function rhs!(_dQ, _Q, p::Tuple{<:DGSEM,<:HyperbolicEquation}, time)
    # Unpack
    dg, equation = p

    # Wrap solution and its derivative
    Q = StateVector{nvariables(equation)}(_Q, dg.dofhandler)
    dQ = StateVector{nvariables(equation)}(_dQ, dg.dofhandler)

    # Restart time derivative
    fill!(dQ, zero(datatype(dQ)))

    # Volume flux
    volume_contribution!(dQ, Q, dg, equation, dg.operators[1])

    # Project Q to faces
    project2faces!(dg.cache.Qf, Q, dg)

    # Boundary conditions
    applyBCs!(dg.cache.Qf, dg, equation, time)

    # Interface fluxes
    interface_fluxes!(dg.cache.Fn, dg.cache.Qf, dg, equation, dg.operators[1].numflux)

    # Surface contribution
    surface_contribution!(dQ, Q, dg.cache.Fn, dg, equation, dg.operators[1])

    # Apply mass matrix
    apply_massmatrix!(dQ, dg)

    # Add source term
    apply_sourceterm!(dQ, Q, dg, time)

    return nothing
end

function volume_contribution!(dQ, Q, dg, equation::HyperbolicEquation, operator)
    @flouthreads for ie in eachelement(dg)
        volume_contribution!(dQ[ie], Q[ie], ie, dg.std, dg, equation, operator)
    end
    return nothing
end

function surface_contribution!(dQ, Q, Fn, dg, equation::HyperbolicEquation, operator)
    @flouthreads for ie in eachelement(dg)
        surface_contribution!(dQ[ie], Q[ie], Fn, ie, dg.std, dg, equation, operator)
    end
    return nothing
end

