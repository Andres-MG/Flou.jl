function rotate2face(Qf, _, ::LinearAdvection)
    return SVector(Qf[1])
end

function rotate2phys(Qrot, _, ::LinearAdvection)
    return SVector(Qrot[1])
end

#==========================================================================================#
#                                     Numerical fluxes                                     #

function numericalflux(Ql, Qr, n, eq::LinearAdvection, ::StdAverage)
    an = dot(eq.a, n)
    return SVector(an * (Ql[1] + Qr[1]) / 2)
end

function numericalflux(Ql, Qr, n, eq::LinearAdvection, nf::LxF)
    # Average
    Fn = numericalflux(Ql, Qr, n, eq, nf.avg)

    # Dissipation
    an = dot(eq.a, n)
    return SVector(Fn[1] + abs(an) * (Ql[1] - Qr[1]) / 2 * nf.intensity)
end

#==========================================================================================#
#                                     Two-point fluxes                                     #

function twopointflux(Q1, Q2, Ja1, Ja2, eq::LinearAdvection, avg::StdAverage)
    n = (Ja1 .+ Ja2) ./ 2
    return numericalflux(Q1, Q2, n, eq, avg)
end
