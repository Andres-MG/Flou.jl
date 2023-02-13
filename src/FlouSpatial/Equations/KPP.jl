function rotate2face(Qf, _, ::KPPEquation)
    return SVector(Qf[1])
end

function rotate2phys(Qrot, _, ::KPPEquation)
    return SVector(Qrot[1])
end

#==========================================================================================#
#                                     Numerical fluxes                                     #

function numericalflux(Ql, Qr, n, ::KPPEquation, ::StdAverage)
    Fl = sin(Ql[1]) * n[1] + cos(Ql[1]) * n[2]
    Fr = sin(Qr[1]) * n[1] + cos(Qr[1]) * n[2]
    return SVector((Fl + Fr) / 2)
end

#==========================================================================================#
#                                     Numerical fluxes                                     #

function twopointflux(Q1, Q2, Ja1, Ja2, eq::KPPEquation, avg::StdAverage)
    n = (Ja1 .+ Ja2) ./ 2
    return numericalflux(Q1, Q2, n, eq, avg)
end
