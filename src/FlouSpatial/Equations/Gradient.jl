struct GradientCache{NV,RT} <: AbstractCache
    Qf::FaceStateVector{NV,RT}
    Fn::FaceBlockVector{NV,RT}
end

function construct_cache(
    ftype::Type,
    dofhandler::DofHandler,
    equation::GradientEquation,
)
    Qf = FaceStateVector{nvariables(equation)}(undef, dofhandler, ftype)
    Fn = FaceBlockVector{nvariables(equation)}(undef, dofhandler, ndims(equation), ftype)
    return GradientCache(Qf, Fn)
end

function FlouCommon.rhs!(
    _G::Array{<:Any,3},
    _Q::Array{<:Any,3},
    p::EquationConfig{<:MultielementDisc,<:GradientEquation},
    _::Real,
)
    # Unpack
    (; disc, equation) = p
    cache = disc.cache

    G = GlobalBlockVector{nvariables(equation)}(_G)
    Q = GlobalStateVector{nvariables(equation)}(_Q)

    # Restart gradients
    fill!(G, zero(datatype(G)))

    # Volume flux
    volume_contribution!(G, Q, disc, equation, disc.operators[1])

    # Project Q to faces
    project2faces!(cache.Qf, Q, disc)

    # Boundary conditions
    applyBCs!(cache.Qf, disc, equation, time)

    # Interface fluxes
    interface_fluxes!(cache.Fn, cache.Qf, disc, equation, disc.operators[1].numflux)

    # Surface contribution
    surface_contribution!(G, Q, disc, equation, disc.operators[1])

    # Apply mass matrix
    apply_massmatrix!(G, disc)

    return nothing
end

function volume_contribution!(G, Q, disc, equation::GradientEquation, operator)
    @flouthreads for ie in eachelement(disc)
        @inbounds volume_contribution!(G, Q, ie, disc.std, disc, equation, operator)
    end
    return nothing
end

function surface_contribution!(G, Q, disc, equation::GradientEquation, operator)
    @flouthreads for ie in eachelement(disc)
        @inbounds surface_contribution!(G, Q, ie, disc.std, disc, equation, operator)
    end
    return nothing
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

function numericalflux(Ql, Qr, n, eq::GradientEquation, ::StdAverage)
    nd = spatialdim(eq)
    nv = nvariables(eq)
    return ntuple(d -> SVector{nv}(
        (Ql[v] + Qr[v]) * n[d] / 2 for v in 1:nv
    ), nd)
end

#==========================================================================================#
#                                     Two-point fluxes                                     #

function twopointflux(Q1, Q2, Ja1, Ja2, eq::GradientEquation, avg::StdAverage)
    n = (Ja1 .+ Ja2) ./ 2
    return numericalflux(Q1, Q2, n, eq, avg)
end
