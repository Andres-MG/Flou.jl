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
    (; mesh, std) = dg

    rt = eltype(first(G))
    iface = mesh.elements[ielem].faceinds
    facepos = mesh.elements[ielem].facepos

    @inbounds for (s, (face, pos)) in enumerate(zip(iface, facepos))
        mul!(G, std.lω[s], Fn[face][pos], one(rt), one(rt))
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
    ::AbstractEquation,
    ::WeakGradOperator,
)
    # Unpack
    (; geometry) = dg

    # Weak gradient operator
    d = std.cache.state[Threads.threadid()][1]
    Ja = geometry.elements[ielem].Ja
    @inbounds for dir in eachdirection(std)
        mul!(d, std.K[dir], Q)
        for i in eachindex(std), innerdir in eachdirection(std)
            G[i, innerdir] -= d[i] * Ja[i][innerdir, dir]
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
    ::AbstractEquation,
    ::StrongGradOperator,
)
    # Unpack
    (; geometry) = dg

    # Strong gradient operator
    d = std.cache.state[Threads.threadid()][1]
    Ja = geometry.elements[ielem].Ja
    @inbounds for dir in eachdirection(std)
        mul!(d, std.Ks[dir], Q)
        for i in eachindex(std), innerdir in eachdirection(std)
            G[i, innerdir] -= d[i] * Ja[i][dir, innerdir]
        end
    end
    return nothing
end

