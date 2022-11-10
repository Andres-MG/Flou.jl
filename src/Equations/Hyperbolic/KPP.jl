struct KPPEquation <: HyperbolicEquation{2,1} end

function Base.show(io::IO, ::MIME"text/plain", eq::KPPEquation)
    @nospecialize
    print(io, "2D KPP equation")
end

function variablenames(::KPPEquation; unicode=false)
    return if unicode
        ("u",)
    else
        ("u",)
    end
end

function volumeflux(Q, ::KPPEquation)
    return SVector{1}(
        SVector{2}(sin(Q[1]), cos(Q[1]))
    )
end

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
