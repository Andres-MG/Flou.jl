function rotate2face(Qf, _, ::LinearAdvection)
    return SVector(Qf[1])
end

function rotate2phys(Qrot, _, ::LinearAdvection)
    return SVector(Qrot[1])
end

function numericalflux(Ql, Qr, n, eq::LinearAdvection, ::StdAverageNumericalFlux)
    an = dot(eq.a, n)
    return SVector(an * (Ql[1] + Qr[1]) / 2)
end

function numericalflux(Ql, Qr, n, eq::LinearAdvection, nf::LxFNumericalFlux)
    # Average
    Fn = numericalflux(Ql, Qr, n, eq, nf.avg)

    # Dissipation
    an = dot(eq.a, n)
    return SVector(Fn[1] + abs(an) * (Ql[1] - Qr[1]) / 2 * nf.intensity)
end
