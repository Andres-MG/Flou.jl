struct BurgersEquation{NV,DV} <: HyperbolicEquation{NV}
    operators::Tuple{DV}
end

function BurgersEquation(div_operator)
    BurgersEquation{1,typeof(div_operator)}((div_operator,))
end

function Base.show(io::IO, m::MIME"text/plain", eq::BurgersEquation)
    @nospecialize
    print(io, eq |> typeof, ":")
    print(io, " Advection operator: "); show(io, m, eq.operators[1])
end

function variablenames(::BurgersEquation; unicode=false)
    return if unicode
        (:u,)
    else
        (:u,)
    end
end

function volumeflux(Q, ::BurgersEquation)
    return SMatrix{1,1}(Q[1]^2 / 2)
end

function rotate2face(Qf, _, ::BurgersEquation)
    return SVector(Qf[1])
end

function rotate2phys(Qrot, _, ::BurgersEquation)
    return SVector(Qrot[1])
end

function numericalflux(Ql, Qr, _, ::BurgersEquation, ::StdAverageNumericalFlux)
    return SVector((Ql[1]^2 + Qr[1]^2) / 4)
end

function numericalflux(Ql, Qr, n, eq::BurgersEquation, nf::LxFNumericalFlux)
    # Average
    Fn = numericalflux(Ql, Qr, n, eq, nf.avg)

    # Dissipation
    λ = max(abs(Ql[1]), abs(Qr[1]))
    return SVector(Fn[1] + λ * (Ql[1] - Qr[1]) / 2 * nf.intensity)
end
