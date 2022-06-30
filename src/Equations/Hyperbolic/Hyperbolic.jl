#==========================================================================================#
#                                           DGSEM                                          #

function rhs!(_dQ, _Q, dg::DGSEM{<:HyperbolicEquation}, time)
    # Unpacking
    (; equation) = dg

    # Wrap solution and its derivative
    Q = StateVector(_Q, dg.dofhandler, dg.stdvec, nvariables(equation))
    dQ = StateVector(_dQ, dg.dofhandler, dg.stdvec, nvariables(equation))

    # Restart time derivative
    fill!(dQ, zero(eltype(dQ)))

    # Volume flux
    volume_contribution!(dQ, Q, dg, equation.operators[1])

    # Project Q to faces
    project2faces!(dg.Qf, Q, dg)

    # Boundary conditions
    applyBCs!(dg.Qf, dg, time)

    # Interface fluxes
    interface_fluxes!(dg.Fn, dg.Qf, dg, dg.riemannsolver)

    # Surface contribution
    surface_contribution!(dQ, dg.Fn, dg, equation.operators[1])

    # Apply mass matrix
    apply_massmatrix!(dQ, dg)

    # Add source term
    apply_sourceterm!(dQ, Q, dg, time)

    return nothing
end

function volume_contribution!(dQ, Q, dg, operator)
    for ireg in eachregion(dg.dofhandler)
        std = dg.stdvec[ireg]
        @flouthreads for ieloc in eachelement(dg.dofhandler)
            ie = reg2loc(dg.dofhandler, ireg, ieloc)
            volume_contribution!(view(dQ[ireg], :, :, ieloc), Q, ie, std, dg, operator)
        end
    end
end

function surface_contribution!(dQ, Fn, dg, operator)
    for ireg in eachregion(dg.dofhandler)
        std = dg.stdvec[ireg]
        @flouthreads for ieloc in eachelement(dg.dofhandler, ireg)
            ie = reg2loc(dg.dofhandler, ireg, ieloc)
            surface_contribution!(view(dQ[ireg], :, :, ieloc), Fn, ie, std, dg, operator)
        end
    end
    return nothing
end
