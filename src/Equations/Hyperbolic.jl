abstract type AbstractHyperbolicEquation{NV} <: AbstractEquation{NV} end

requires_subgrid(e::AbstractHyperbolicEquation) = requires_subgrid(e.div_operator)

#==========================================================================================#
#                                           DGSEM                                          #

function evaluate!(_dQ, _Q, disc::DGSEM{<:AbstractHyperbolicEquation}, time)
    # Unpacking
    (; mesh, physelem, physface, dofhandler, stdvec, equation) = disc

    # Wrap solution and its derivative
    Q = StateVector(_Q, dofhandler, stdvec, nvariables(equation))
    dQ = StateVector(_dQ, dofhandler, stdvec, nvariables(equation))

    # Restart time derivative
    fill!(dQ, zero(eltype(dQ)))

    # Volume flux
    volume_contribution!(dQ, Q, dofhandler, stdvec, physelem, equation)

    # Project Q to faces
    project2faces!(disc.Qf, Q, mesh, dofhandler, stdvec, equation)

    # Boundary conditions
    applyBCs!(disc.Qf, mesh, physface, time, equation, disc.BCs)

    # Interface fluxes
    interface_fluxes!(
        disc.Fn,
        disc.Qf,
        mesh,
        dofhandler,
        stdvec,
        physface,
        equation,
        disc.riemannsolver,
    )

    # Surface contribution
    surface_contribution!(dQ, disc.Fn, mesh, dofhandler, stdvec, equation)

    # Apply mass matrix
    apply_massmatrix!(dQ, dofhandler, physelem)

    # Add source term
    apply_sourceterm!(dQ, Q, disc.source!, dofhandler, physelem, time)

    return nothing
end

function volume_contribution!(dQ, Q, dh, stdvec, physelem, eq)
    volume_div_operator!(dQ, Q, dh, stdvec, physelem, eq, eq.div_operator)
    return nothing
end

function surface_contribution!(dQ, Fn, mesh, dh, stdvec, eq)
    surface_div_operator!(dQ, Fn, mesh, dh, stdvec, eq, eq.div_operator)
    return nothing
end

#==========================================================================================#
#                                   Hyperbolic equations                                   #

include("LinearAdvection.jl")
include("Burgers.jl")
include("Euler.jl")
include("KPP.jl")
