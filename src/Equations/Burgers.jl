struct BurgersEquation{NV,DV} <: AbstractHyperbolicEquation{NV}
    div_operator::DV
end

function BurgersEquation(div_operator)
    BurgersEquation{1,typeof(div_operator)}(div_operator)
end

function Base.show(io::IO, m::MIME"text/plain", eq::BurgersEquation)
    @nospecialize
    print(io, eq |> typeof, ":")
    print(io, " Advection operator: "); show(io, m, eq.div_operator)
end

function variablenames(::BurgersEquation; unicode=false)
    return if unicode
        (:u,)
    else
        (:u,)
    end
end

function volumeflux!(F, Q, ::BurgersEquation)
    F[1, 1] = Q[1]^2 / 2
    return nothing
end

function rotate2face!(Qrot, Qf, n, t, b, ::BurgersEquation)
    Qrot[1] = Qf[1]
    return nothing
end

function rotate2phys!(Qf, Qrot, n, t, b, ::BurgersEquation)
    Qf[1] = Qrot[1]
    return nothing
end

function numericalflux!(Fn, Ql, Qr, n, ::BurgersEquation, ::StdAverageNumericalFlux)
    Fn[1] = (Ql[1]^2 + Qr[1]^2) / 4
    return nothing
end

function numericalflux!(Fn, Ql, Qr, n, eq::BurgersEquation, nf::LxFNumericalFlux)
    # Average
    numericalflux!(Fn, Ql, Qr, n, eq, nf.avg)

    # Dissipation
    λ = max(abs(Ql[1]), abs(Qr[1]))
    Fn[1] += λ * (Ql[1] - Qr[1]) / 2 * nf.intensity
    return nothing
end
