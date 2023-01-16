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
