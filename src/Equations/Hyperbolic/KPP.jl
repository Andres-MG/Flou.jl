struct KPPEquation{NV,DV} <: HyperbolicEquation{NV}
    operators::Tuple{DV}
end

function KPPEquation(div_operator)
    KPPEquation{1,typeof(div_operator)}((div_operator,))
end

function Base.show(io::IO, m::MIME"text/plain", eq::KPPEquation)
    @nospecialize
    println(io, eq |> typeof, ":")
    print(io, " Divergence operator: "); show(io, m, eq.operators[1]); println(io, "")
end

function variablenames(::KPPEquation; unicode=false)
    return if unicode
        (:u,)
    else
        (:u,)
    end
end

function volumeflux!(F, Q, ::KPPEquation)
    F[1,1] = sin(Q[1])
    F[2,1] = cos(Q[1])
    return nothing
end

function rotate2face!(Qrot, Qf, n, t, b, ::KPPEquation)
    Qrot[1] = Qf[1]
    return nothing
end

function rotate2phys!(Qf, Qrot, n, t, b, ::KPPEquation)
    Qf[1] = Qrot[1]
    return nothing
end

function numericalflux!(Fn, Ql, Qr, n, ::KPPEquation, ::StdAverageNumericalFlux)
    Fl = sin(Ql[1]) * n[1] + cos(Ql[1]) * n[2]
    Fr = sin(Qr[1]) * n[1] + cos(Qr[1]) * n[2]
    Fn[1] = (Fl + Fr) / 2
    return nothing
end

function initial_whirl_KPP!(Q, disc)
    (; mesh, std, elemgeom) = disc
    for ie in eachelement(mesh)
        xy = coords(elemgeom, ie)
        for I in eachindex(std)
            x, y = xy[I]
            Q[I, 1, ie] = (x^2 + y^2) <= 1 ? 7π/2 : π/4
        end
    end
end
