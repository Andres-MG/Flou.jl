struct LinearAdvection{ND,NV,RT,DV} <: HyperbolicEquation{NV}
    operators::Tuple{DV}
    a::SVector{ND,RT}
end

function LinearAdvection(div_operator, velocity::RT...) where {RT}
    ndim = length(velocity)
    1 <= ndim <= 3 || throw(ArgumentError(
        "Linear advection is implemented in 1D, 2D and 3D."
    ))
    LinearAdvection{ndim,1,RT,typeof(div_operator)}(
        (div_operator,),
        SVector{ndim,RT}(velocity...),
    )
end

function Base.show(io::IO, m::MIME"text/plain", eq::LinearAdvection)
    @nospecialize
    println(io, eq |> typeof, ":")
    print(io, " Advection operator: "); show(io, m, eq.operators[1]); println(io, "")
    print(io, " Advection velocity: ", eq.a)
end

function variablenames(::LinearAdvection; unicode=false)
    return if unicode
        (:u,)
    else
        (:u,)
    end
end

function volumeflux!(F, Q, eq::LinearAdvection{ND}) where {ND}
    for i in 1:ND
        F[i,1] = eq.a[i] * Q[1]
    end
    return nothing
end

function rotate2face!(Qrot, Qf, n, t, b, ::LinearAdvection)
    Qrot[1] = Qf[1]
    return nothing
end

function rotate2phys!(Qf, Qrot, n, t, b, ::LinearAdvection)
    Qf[1] = Qrot[1]
    return nothing
end

function numericalflux!(
    Fn,
    Ql,
    Qr,
    n,
    eq::LinearAdvection{ND},
    ::StdAverageNumericalFlux,
) where {
    ND,
}
    an = zero(eltype(eq.a))
    @inbounds for i in 1:ND
        an += eq.a[i] * n[i]
    end
    Fn[1] = an * (Ql[1] + Qr[1]) / 2
    return nothing
end

function numericalflux!(
    Fn,
    Ql,
    Qr,
    n,
    eq::LinearAdvection{ND},
    nf::LxFNumericalFlux,
) where {
    ND,
}
    # Average
    numericalflux!(Fn, Ql, Qr, n, eq, nf.avg)

    # Dissipation
    an = zero(eltype(eq.a))
    @inbounds for i in 1:ND
        an += eq.a[i] * n[i]
    end
    Fn[1] += abs(an) * (Ql[1] - Qr[1]) / 2 * nf.intensity
    return nothing
end
