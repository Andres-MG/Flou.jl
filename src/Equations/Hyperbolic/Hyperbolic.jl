#==========================================================================================#
#                                           DGSEM                                          #

struct HyperbolicDGcache{RT,R} <: DGcache{RT}
    Qf::FaceStateVector{RT,R}
    Fn::FaceStateVector{RT,R}
end

function construct_cache(disctype, realtype, dofhandler, equation::HyperbolicEquation)
    if disctype == :dgsem
        Qf = FaceStateVector{realtype}(undef, nvariables(equation), dofhandler)
        Fn = FaceStateVector{realtype}(undef, nvariables(equation), dofhandler)
        return HyperbolicDGcache(Qf, Fn)
    else
        @error "Unknown discretization type $(disctype)."
    end
end

function rhs!(_dQ, _Q, p::Tuple{<:DGSEM,<:HyperbolicEquation}, time)
    # Unpack
    dg, equation = p

    # Wrap solution and its derivative
    Q = StateVector(_Q, dg.dofhandler)
    dQ = StateVector(_dQ, dg.dofhandler)

    # Restart time derivative
    fill!(dQ, zero(eltype(dQ)))

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

function volume_contribution!(dQ, Q, dg, equation, operator)
    @flouthreads for ie in eachelement(dg)
        std = get_std(dg, ie)
        volume_contribution!(dQ[ie], Q[ie], ie, std, dg, equation, operator)
    end
    return nothing
end

function surface_contribution!(dQ, Q, Fn, dg, equation, operator)
    @flouthreads for ie in eachelement(dg)
        std = get_std(dg, ie)
        surface_contribution!(dQ[ie], Q[ie], Fn, ie, std, dg, equation, operator)
    end
    return nothing
end

