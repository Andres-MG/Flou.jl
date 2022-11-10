struct BurgersEquation <: HyperbolicEquation{1,1} end

function Base.show(io::IO, ::MIME"text/plain", eq::BurgersEquation)
    @nospecialize
    print(io, "1D Burgers equation")
end

function variablenames(::BurgersEquation; unicode=false)
    return if unicode
        ("u",)
    else
        ("u",)
    end
end

function volumeflux(Q, ::BurgersEquation)
    return (SVector{1}(Q[1]^2 / 2),)
end

function rotate2face(Qf, _, ::BurgersEquation)
    return SVector(Qf[1])
end

function rotate2phys(Qrot, _, ::BurgersEquation)
    return SVector(Qrot[1])
end

function get_max_dt(Q, Δx::Real, cfl::Real, ::BurgersEquation)
    return cfl * Δx / Q[1]
end

function numericalflux(Ql, Qr, n, ::BurgersEquation, ::StdAverageNumericalFlux)
    return SVector((Ql[1]^2 + Qr[1]^2) / 4 * n[1])
end

function numericalflux(Ql, Qr, n, eq::BurgersEquation, nf::LxFNumericalFlux)
    # Average
    Fn = numericalflux(Ql, Qr, n, eq, nf.avg)

    # Dissipation
    λ = max(abs(Ql[1]), abs(Qr[1]))
    return SVector(Fn[1] + λ * (Ql[1] - Qr[1]) / 2 * nf.intensity)
end
