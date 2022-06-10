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
