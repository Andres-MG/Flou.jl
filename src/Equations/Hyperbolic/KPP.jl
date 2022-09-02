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

function volumeflux(Q, ::KPPEquation)
    return SMatrix{2,1}(sin(Q[1]), cos(Q[1]))
end

function rotate2face(Qf, _, _, _, ::KPPEquation)
    return SVector(Qf[1])
end

function rotate2phys(Qrot, _, _, _, ::KPPEquation)
    return SVector(Qrot[1])
end

function numericalflux(Ql, Qr, n, ::KPPEquation, ::StdAverageNumericalFlux)
    Fl = sin(Ql[1]) * n[1] + cos(Ql[1]) * n[2]
    Fr = sin(Qr[1]) * n[1] + cos(Qr[1]) * n[2]
    return SVector((Fl + Fr) / 2)
end

# TODO
# function initial_whirl_KPP!(Q, disc)
#     (; mesh, std, elemgeom) = disc
#     for ie in eachelement(mesh)
#         xy = coords(elemgeom, ie)
#         for I in eachindex(std)
#             x, y = xy[I]
#             Q[I, 1, ie] = (x^2 + y^2) <= 1 ? 7π/2 : π/4
#         end
#     end
# end
