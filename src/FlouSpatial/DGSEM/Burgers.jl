function rotate2face(Qf, _, ::BurgersEquation)
    return SVector(Qf[1])
end

function rotate2phys(Qrot, _, ::BurgersEquation)
    return SVector(Qrot[1])
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
