abstract type AbstractGradOperator <: AbstractOperator end

function surface_contribution!(
    G,
    _,
    Fn,
    ielem,
    std::AbstractStdRegion,
    fr::FR,
    equation::AbstractEquation,
    ::AbstractGradOperator,
)
    # Unpack
    (; mesh) = fr

    rt = datatype(G)
    iface = mesh.elements[ielem].faceinds
    facepos = mesh.elements[ielem].facepos

    @inbounds for (s, (face, pos)) in enumerate(zip(iface, facepos))
        mul!(G, std.∂g[s], Fn.face[face][pos], one(rt), one(rt))
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
    fr::FR,
    ::AbstractEquation,
    ::WeakGradOperator,
)
    # Unpack
    (; geometry) = fr

    # Weak gradient operator
    d = std.cache.state[Threads.threadid()][1]
    Ja = geometry.elements[ielem].Ja
    @inbounds for dir in eachdirection(std)
        mul!(d, std.Dw[dir], Q)
        for i in eachdof(std), innerdir in eachdirection(std)
            G[i, innerdir] += d[i] * Ja[i][innerdir, dir]
        end
    end
    return nothing
end
