#==========================================================================================#
#                                           DGSEM                                          #

struct GradientDGcache{RT,Q,F} <: DGcache{RT}
    Qf::FaceStateVector{RT,Q}
    Fn::FaceBlockVector{RT,F}
end

function construct_cache(disctype, realtype, dofhandler, eq::GradientEquation)
    if disctype == :dgsem
        Qf = FaceStateVector{realtype}(undef, nvariables(eq), dofhandler)
        Fn = FaceBlockVector{realtype}(undef, nvariables(eq), ndims(eq), dofhandler)
        return GradientDGcache(Qf, Fn)
    else
        @error "Unknown discretization type $(disctype)"
    end
end

function rhs!(_G, _Q, p::Tuple{<:DGSEM,<:GradientEquation}, _)
    # Unpack
    dg, equation = p

    # Wrap solution and its derivative
    Q = StateVector(_Q, dg.dofhandler)
    G = BlockVector(_G, dg.dofhandler)

    # Restart gradients
    fill!(G, zero(eltype(G)))

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
        std = get_std(dg, ie)
        volume_contribution!(G[ie], Q[ie], ie, std, dg, equation, operator)
    end
    return nothing
end

function surface_contribution!(G, Q, Fn, dg, equation::GradientEquation, operator)
    @flouthreads for ie in eachelement(dg)
        std = get_std(dg, ie)
        surface_contribution!(G[ie], Q[ie], Fn, ie, std, dg, equation, operator)
    end
    return nothing
end

#==========================================================================================#
#                                 Gradient struct interface                                #

function Base.show(io::IO, ::MIME"text/plain", eq::GradientEquation{ND,NV}) where {ND,NV}
    @nospecialize
    print(io, ND, "D Gradient equation with ", NV, " variables")
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

function rotate2face(Qf::AbstractVector, _, ::GradientEquation{ND,NV}) where {ND,NV}
    return SVector{NV}(Qf)
end

function rotate2phys(Qrot::AbstractMatrix, _, ::GradientEquation{ND,NV}) where {ND,NV}
    return SMatrix{NV,ND}(Qrot)
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
    NV,
}
    return SMatrix{NV,ND}(
        (Ql[v] + Qr[v]) * n[d] / 2 for v in 1:NV, d in 1:ND
    )
end
