struct LinearAdvection{ND,NV,RT} <: HyperbolicEquation{NV}
    a::SVector{ND,RT}
end

function LinearAdvection(velocity::RT...) where {RT}
    ndim = length(velocity)
    1 <= ndim <= 3 || throw(ArgumentError(
        "Linear advection is implemented in 1D, 2D and 3D."
    ))
    LinearAdvection{ndim,1,RT}(SVector{ndim,RT}(velocity...))
end

function Base.show(io::IO, ::MIME"text/plain", eq::LinearAdvection{ND}) where {ND}
    @nospecialize
    println(io, ND, "D linear advection equation:")
    print(io, " Advection velocity: ", eq.a)
end

function variablenames(::LinearAdvection; unicode=false)
    return if unicode
        ("u",)
    else
        ("u",)
    end
end

function volumeflux(Q, eq::LinearAdvection{ND}) where {ND}
    return SMatrix{ND,1}(a * Q[1] for a in eq.a)
end

function rotate2face(Qf, _, ::LinearAdvection)
    return SVector(Qf[1])
end

function rotate2phys(Qrot, _, ::LinearAdvection)
    return SVector(Qrot[1])
end

function get_max_dt(_, Δx::Real, cfl::Real, eq::LinearAdvection)
    return cfl * Δx / norm(eq.a)
end

function numericalflux(
    Ql,
    Qr,
    n,
    eq::LinearAdvection{ND},
    ::StdAverageNumericalFlux,
) where {
    ND,
}
    an = dot(eq.a, n)
    return SVector(an * (Ql[1] + Qr[1]) / 2)
end

function numericalflux(
    Ql,
    Qr,
    n,
    eq::LinearAdvection{ND},
    nf::LxFNumericalFlux,
) where {
    ND,
}
    # Average
    Fn = numericalflux(Ql, Qr, n, eq, nf.avg)

    # Dissipation
    an = dot(eq.a, n)
    return SVector(Fn[1] + abs(an) * (Ql[1] - Qr[1]) / 2 * nf.intensity)
end
