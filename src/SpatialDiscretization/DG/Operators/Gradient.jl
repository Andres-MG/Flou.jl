abstract type AbstractGradOperator <: AbstractOperator end

function surface_contribution!(
    G,
    _,
    Fn,
    ielem,
    std::AbstractStdRegion,
    dg::DiscontinuousGalerkin,
    equation::AbstractEquation,
    ::AbstractGradOperator,
)
    # Unpack
    (; mesh) = dg

    rt = eltype(G)
    iface = mesh.elements[ielem].faceinds
    facepos = mesh.elements[ielem].facepos

    @inbounds for (s, (face, pos)) in enumerate(zip(iface, facepos))
        for dir in eachdirection(std)
            @views mul!(
                G[:, :, dir], std.lω[s], Fn[face][pos][:, :, dir], one(rt), one(rt)
            )
        end
    end
    return nothing
end

#==========================================================================================#
#                                  Weak gradient operator                                  #

struct WeakGradOperator{F<:AbstractNumericalFlux} <: AbstractGradOperator
    numflux::F
end

function volume_contribution!(
    G,
    Q,
    ielem,
    std::AbstractStdRegion,
    dg::DiscontinuousGalerkin,
    equation::AbstractEquation,
    ::WeakGradOperator,
)
    # Unpack
    (; geometry) = dg

    # Weak gradient operator
    d = std.cache.scalar[Threads.threadid()][1]
    Ja = geometry.elements[ielem].Ja
    @inbounds for dir in eachdirection(std)
        mul!(d, std.K[dir], Q)
        for v in eachvariable(equation), i in eachindex(std)
            @views G[i, v, :] .-= d[i, v] .* Ja[i][:, dir]
        end
    end
    return nothing
end

#==========================================================================================#
#                                Strong divergence operator                                #

struct StrongGradOperator{F<:AbstractNumericalFlux} <: AbstractGradOperator
    numflux::F
end

function volume_contribution!(
    G,
    Q,
    ielem,
    std::AbstractStdRegion,
    dg::DiscontinuousGalerkin,
    equation::AbstractEquation,
    ::StrongGradOperator,
)
    # Unpack
    (; geometry) = dg

    # Strong gradient operator
    d = std.cache.scalar[Threads.threadid()][1]
    Ja = geometry.elements[ielem].Ja
    @inbounds for dir in eachdirection(std)
        mul!(d, std.Ks[dir], Q)
        for v in eachvariable(equation), i in eachindex(std)
            @views G[i, v, :] .-= d[i, v] .* Ja[i][dir, :]
        end
    end
    return nothing
end

