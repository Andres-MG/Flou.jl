#==========================================================================================#
#                                           DGSEM                                          #

struct HyperbolicDGcache{NV,RT,D} <: DGcache{RT}
    Qf::FaceStateVector{NV,RT,D}
    Fn::FaceStateVector{NV,RT,D}
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

function rhs!(dQ, Q, p::Tuple{<:DGSEM,<:HyperbolicEquation}, time)
    # Unpack
    dg, equation = p

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

