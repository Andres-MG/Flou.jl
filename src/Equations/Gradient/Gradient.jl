#==========================================================================================#
#                                           DGSEM                                          #

struct GradientDGcache{NV,RT,DQ,DF,TQ,TF} <: DGcache{RT}
    Qf::FaceStateVector{NV,RT,DQ,TQ}
    Fn::FaceBlockVector{NV,RT,DF,TF}
end

function construct_cache(disctype, realtype, dofhandler, eq::GradientEquation)
    if disctype == :dgsem
        Qf = FaceStateVector{nvariables(eq),realtype}(undef, dofhandler)
        Fn = FaceBlockVector{nvariables(eq),realtype}(undef, ndims(eq), dofhandler)
        return GradientDGcache(Qf, Fn)
    else
        @error "Unknown discretization type $(disctype)"
    end
end

function rhs!(_G, _Q, p::Tuple{<:DGSEM,<:GradientEquation}, _)
    # Unpack
    dg, equation = p

    # Wrap solution and its derivative
    Q = StateVector{nvariables(equation)}(_Q, dg.dofhandler)
    G = BlockVector{nvariables(equation)}(_G, dg.dofhandler)

    # Restart gradients
    fill!(G, zero(datatype(G)))

    # Volume flux
    volume_contribution!(G, Q, dg, equation, dg.operators[1])

    # Project Q to faces
    project2faces!(dg.cache.Qf, Q, dg)

    # Boundary conditions
    applyBCs!(dg.cache.Qf, dg, equation, time)

    # Interface fluxes
    interface_fluxes!(dg.cache.Fn, dg.cache.Qf, dg, equation, dg.operators[1].numflux)

    # Surface contribution
    surface_contribution!(G, Q, dg.cache.Fn, dg, equation, dg.operators[1])

    # Apply mass matrix
    apply_massmatrix!(G, dg)

    return nothing
end

function volume_contribution!(G, Q, dg, equation::GradientEquation, operator)
    @flouthreads for ie in eachelement(dg)
        volume_contribution!(G[ie], Q[ie], ie, dg.std, dg, equation, operator)
    end
    return nothing
end

function surface_contribution!(G, Q, Fn, dg, equation::GradientEquation, operator)
    @flouthreads for ie in eachelement(dg)
        surface_contribution!(G[ie], Q[ie], Fn, ie, dg.std, dg, equation, operator)
    end
    return nothing
end

#==========================================================================================#
#                                 Gradient struct interface                                #

function Base.show(io::IO, ::MIME"text/plain", eq::GradientEquation)
    @nospecialize
    nd = ndims(eq)
    nvars = nvariables(eq)
    print(io, nd, "D Gradient equation with ", nvars, " variables")
    return nothing
end

function variablenames(::GradientEquation{ND,NV}; unicode=false) where {ND,NV}
    names = if unicode
        ["∂u$(i)/∂x$(j)" for i in 1:NV, j in 1:ND]
    else
        ["u_$(i)$(j)" for i in 1:NV, j in 1:ND]
    end
    return names |> vec |> Tuple
end

function rotate2face(Qf, _, ::GradientEquation{ND,NV}) where {ND,NV}
    return SVector{NV}(Qf)
end

function rotate2phys(Qrot, _, ::GradientEquation{ND,NV}) where {ND,NV}
    return ntuple(d -> SVector{NV}(
            Qrot[d][v] for v in 1:NV
        ), ND)
end

#==========================================================================================#
#                                     Numerical fluxes                                     #

function numericalflux(
    Ql,
    Qr,
    n,
    ::GradientEquation{ND,NV},
    ::StdAverageNumericalFlux,
) where {
    ND,
    NV
}
    return ntuple(d -> SVector{NV}(
            (Ql[v] + Qr[v]) * n[d] / 2 for v in 1:NV
        ), ND)
end
