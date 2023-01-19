function rotate2face(Qf, _, ::KPPEquation)
    return SVector(Qf[1])
end

function rotate2phys(Qrot, _, ::KPPEquation)
    return SVector(Qrot[1])
end

function numericalflux(Ql, Qr, n, ::KPPEquation, ::StdAverageNumericalFlux)
    Fl = sin(Ql[1]) * n[1] + cos(Ql[1]) * n[2]
    Fr = sin(Qr[1]) * n[1] + cos(Qr[1]) * n[2]
    return SVector((Fl + Fr) / 2)
end
