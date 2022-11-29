abstract type AbstractDivOperator <: AbstractOperator end

function surface_contribution!(
    dQ,
    _,
    Fn,
    ielem,
    std::AbstractStdRegion,
    dg::DiscontinuousGalerkin,
    ::AbstractEquation,
    ::AbstractDivOperator,
)
    # Unpack
    (; mesh) = dg

    rt = datatype(dQ)
    iface = mesh.elements[ielem].faceinds
    facepos = mesh.elements[ielem].facepos

    @inbounds for (s, (face, pos)) in enumerate(zip(iface, facepos))
        mul!(dQ, std.lω[s], Fn.face[face][pos], -one(rt), one(rt))
    end
    return nothing
end

#==========================================================================================#
#                                 Weak divergence operator                                 #

struct WeakDivOperator{F<:AbstractNumericalFlux} <: AbstractDivOperator
    numflux::F
end

function volume_contribution!(
    dQ,
    Q,
    ielem,
    std::AbstractStdRegion,
    dg::DiscontinuousGalerkin,
    equation::AbstractEquation,
    ::WeakDivOperator,
)
    # Unpack
    (; geometry) = dg

    # Volume fluxes
    F̃ = std.cache.block[Threads.threadid()][1]
    Ja = geometry.elements[ielem].Ja
    @inbounds for i in eachindex(std)
        F = volumeflux(Q[i], equation)
        F̃[i, :] .= contravariant(F, Ja[i])
    end

    # Weak derivative
    rt = datatype(dQ)
    @inbounds for s in eachindex(std.K)
        mul!(dQ, std.K[s], view(F̃, :, s), one(rt), one(rt))
    end
    return nothing
end

#==========================================================================================#
#                                Strong divergence operator                                #

struct StrongDivOperator{F<:AbstractNumericalFlux} <: AbstractDivOperator
    numflux::F
end

function volume_contribution!(
    dQ,
    Q,
    ielem,
    std::AbstractStdRegion,
    dg::DiscontinuousGalerkin,
    equation::AbstractEquation,
    ::StrongDivOperator,
)
    # Unpack
    (; geometry) = dg

    # Volume fluxes
    F̃ = std.cache.block[Threads.threadid()][1]
    Ja = geometry.elements[ielem].Ja
    @inbounds for i in eachindex(std)
        F = volumeflux(Q[i], equation)
        F̃[i, :] .= contravariant(F, Ja[i])
    end

    # Strong derivative
    rt = datatype(dQ)
    @inbounds for s in eachindex(std.K)
        mul!(dQ, std.Ks[s], view(F̃, :, s), one(rt), one(rt))
    end
    return nothing
end

#==========================================================================================#
#                                 Split divergence operator                                #

"""
    SplitDivOperator(twopointflux)

Split-divergence operator, only implemented for tensor-product elements. The two-point flux
represents the splitting strategy.
"""
struct SplitDivOperator{
    T<:AbstractNumericalFlux,
    F<:AbstractNumericalFlux,
} <: AbstractDivOperator
    tpflux::T
    numflux::F
end

function requires_subgrid(::SplitDivOperator, ::AbstractStdRegion{QT}) where {QT}
    return QT == GaussQuadrature
end

function twopointflux end

Base.@propagate_inbounds function _split_gauss_deriv_1d!(
    dQ, Fnl, Fnr, Q, W, Fli, Fri, Ja, frames, Js,
    std, l, r, ω, equation, op, idir,
)
    # Precompute two-point and entropy-projected fluxes
    for i in eachindex(std, idir)
        W[i] = vars_cons2entropy(Q[i], equation)
    end
    Wl = (l * W) |> transpose
    Wr = (r * W) |> transpose
    Ql = vars_entropy2cons(Wl, equation)
    Qr = vars_entropy2cons(Wr, equation)

    nl = frames[1].n * Js[1]
    nr = frames[end].n * Js[end]

    for i in eachindex(std, idir)
        Fli[i] = twopointflux(
            Q[i],
            Ql,
            view(Ja[i], :, idir),
            nl,
            equation,
            op.tpflux,
        )
        Fri[i] = twopointflux(
            Q[i],
            Qr,
            view(Ja[i], :, idir),
            nr,
            equation,
            op.tpflux,
        )
    end

    # Complementary grid fluxes
    l_Fli = l * Fli
    r_Fri = r * Fri
    for i in eachindex(std, idir)
        dQ[i] += l[i] * ω * (Fli[i] - l_Fli - Fnl) -
                 r[i] * ω * (Fri[i] - r_Fri + Fnr)
    end
    return nothing
end

function surface_contribution!(
    dQ,
    Q,
    Fn,
    ielem,
    std::AbstractStdRegion{<:GaussQuadrature,ND},
    dg::DiscontinuousGalerkin,
    equation::AbstractEquation,
    op::SplitDivOperator,
) where {
    ND
}
    # Unpack
    (; mesh, geometry) = dg

    rt = datatype(Q)
    iface = mesh.elements[ielem].faceinds
    facepos = mesh.elements[ielem].facepos

    Qr = reshape(Q, size(std))
    dQr = reshape(dQ, size(std))
    Ja = reshape(geometry.elements[ielem].Ja, size(std))
    frames = geometry.subgrids[ielem].frames[]
    Js = geometry.subgrids[ielem].J[]

    tid = Threads.threadid()

    if ND == 1
        W = std.cache.state[tid][1]
        Fli = std.cache.state[tid][2]
        Fri = std.cache.state[tid][3]
        l = std.l[1]
        r = std.l[2]
        _split_gauss_deriv_1d!(
            dQ, Fn.face[iface[1]][facepos[1]][1], Fn.face[iface[2]][facepos[2]][1],
            Q, W, Fli, Fri, Ja, frames[1], Js[1], std, l, r, one(rt), equation, op, 1,
        )

    elseif ND == 2
        ω = std.face.ω
        W = std.face.cache.state[tid][1]
        Fli = std.face.cache.state[tid][2]
        Fri = std.face.cache.state[tid][3]
        l = std.face.l[1]
        r = std.face.l[2]

        # X direction
        @inbounds for j in eachindex(std, 2)
            @views _split_gauss_deriv_1d!(
                dQr[:, j],
                Fn.face[iface[1]][facepos[1]][j], Fn.face[iface[2]][facepos[2]][j],
                Qr[:, j], W, Fli, Fri, Ja[:, j], frames[1][:, j], Js[1][:, j],
                std, l, r, ω[j], equation, op, 1,
            )
        end

        # Y direction
        @inbounds for i in eachindex(std, 1)
            @views _split_gauss_deriv_1d!(
                dQr[i, :],
                Fn.face[iface[3]][facepos[3]][i], Fn.face[iface[4]][facepos[4]][i],
                Qr[i, :], W, Fli, Fri, Ja[i, :], frames[2][i, :], Js[2][i, :],
                std, l, r, ω[i], equation, op, 2,
            )
        end

    else # ND == 3
        li = LinearIndices(std.face)
        ω = std.face.ω
        W = std.edge.cache.state[tid][1]
        Fli = std.edge.cache.state[tid][2]
        Fri = std.edge.cache.state[tid][3]
        l = std.edge.l[1]
        r = std.edge.l[2]

        # X direction
        @inbounds for k in eachindex(std, 3), j in eachindex(std, 2)
            ind = li[j, k]
            @views _split_gauss_deriv_1d!(
                dQr[:, j, k],
                Fn.face[iface[1]][facepos[1]][ind], Fn.face[iface[2]][facepos[2]][ind],
                Qr[:, j, k], W, Fli, Fri, Ja[:, j, k], frames[1][:, j, k],
                Js[1][:, j, k], std, l, r, ω[ind], equation, op, 1,
            )
        end

        # Y direction
        @inbounds for k in eachindex(std, 3), i in eachindex(std, 1)
            ind = li[i, k]
            @views _split_gauss_deriv_1d!(
                dQr[i, :, k],
                Fn.face[iface[3]][facepos[3]][ind], Fn.face[iface[4]][facepos[4]][ind],
                Qr[i, :, k], W, Fli, Fri, Ja[i, :, k], frames[2][i, :, k],
                Js[2][i, :, k], std, l, r, ω[ind], equation, op, 2,
            )
        end

        # Z direction
        @inbounds for j in eachindex(std, 2), i in eachindex(std, 1)
            ind = li[i, j]
            @views _split_gauss_deriv_1d!(
                dQr[i, j, :],
                Fn.face[iface[5]][facepos[5]][ind], Fn.face[iface[6]][facepos[6]][ind],
                Qr[i, j, :], W, Fli, Fri, Ja[i, j, :], frames[3][i, j, :],
                Js[3][i, j, :], std, l, r, ω[ind], equation, op, 3,
            )
        end
    end
    return nothing
end

Base.@propagate_inbounds function _split_flux_1d!(F♯, F̃, Q, Ja, std, equation, op, idir)
    for i in eachindex(std, idir)
        F♯[i, i] = F̃[i]
        for l in (i + 1):size(std, idir)
            F♯[l, i] = twopointflux(
                Q[i],
                Q[l],
                view(Ja[i], :, idir),
                view(Ja[l], :, idir),
                equation,
                op.tpflux,
            )
            F♯[i, l] = F♯[l, i]
        end
    end
    return nothing
end

function volume_contribution!(
    dQ,
    Q,
    ielem,
    std::AbstractStdRegion{QT,ND},
    dg::DiscontinuousGalerkin,
    equation::AbstractEquation,
    op::SplitDivOperator,
) where {
    QT,
    ND
}
    # Unpack
    (; geometry) = dg

    is_tensor_product(std) || throw(ArgumentError(
        "All the standard regions must be tensor-products."
    ))

    # Buffers
    tid = Threads.threadid()
    F̃ = std.cache.block[tid][1]
    F♯ = std.cache.sharp[tid]

    # Volume fluxes (diagonal of F♯ matrices)
    Ja = geometry.elements[ielem].Ja
    @inbounds for i in eachindex(std)
        F = volumeflux(Q[i], equation)
        F̃[i, :] .= contravariant(F, Ja[i])
    end

    # Two-point fluxes
    dQr = reshape(dQ, size(std))
    Qr = reshape(Q, size(std))
    Jar = reshape(Ja, size(std))
    F̃r = reshape(F̃, (size(std)..., ND))

    if ND == 1
        _split_flux_1d!(F♯[1], F̃, Q, Jar, std, equation, op, 1)
        for i in eachindex(std)
            dQ[i] += dot(view(std.K♯[1], i, :), view(F♯[1], :, i))
        end

    elseif ND == 2
        # X direction
        @inbounds for j in eachindex(std, 2)
            @views _split_flux_1d!(
                F♯[1], F̃r[:, j, 1], Qr[:, j], Jar[:, j],
                std, equation, op, 1,
            )
            ω = std.face.ω[j]
            for i in eachindex(std, 1)
                dQr[i, j] += ω * dot(view(std.K♯[1], i, :), view(F♯[1], :, i))
            end
        end

        # Y direction
        @inbounds for i in eachindex(std, 1)
            @views _split_flux_1d!(
                F♯[2], F̃r[i, :, 2], Qr[i, :], Jar[i, :],
                std, equation, op, 2,
            )
            ω = std.face.ω[i]
            for j in eachindex(std, 2)
                dQr[i, j] += ω * dot(view(std.K♯[2], j, :), view(F♯[2], :, j))
            end
        end

    else # ND == 3
        li = LinearIndices(std.face)

        # X direction
        @inbounds for k in eachindex(std, 3), j in eachindex(std, 2)
            @views _split_flux_1d!(
                F♯[1], F̃r[:, j, k, 1], Qr[:, j, k], Jar[:, j, k],
                std, equation, op, 1,
            )
            ω = std.face.ω[li[j, k]]
            for i in eachindex(std, 1)
                dQr[i, j, k] += ω * dot(view(std.K♯[1], i, :), view(F♯[1], :, i))
            end
        end

        # Y direction
        @inbounds for k in eachindex(std, 3), i in eachindex(std, 1)
            @views _split_flux_1d!(
                F♯[2], F̃r[i, :, k, 2], Qr[i, :, k], Jar[i, :, k],
                std, equation, op, 2,
            )
            ω = std.face.ω[li[i, k]]
            for j in eachindex(std, 2)
                dQr[i, j, k] += ω * dot(view(std.K♯[2], j, :), view(F♯[2], :, j))
            end
        end

        # Z direction
        @inbounds for j in eachindex(std, 2), i in eachindex(std, 1)
            @views _split_flux_1d!(
                F♯[3], F̃r[i, j, :, 3], Qr[i, j, :], Jar[i, j, :],
                std, equation, op, 3,
            )
            ω = std.face.ω[li[i, j]]
            for k in eachindex(std, 3)
                dQr[i, j, k] += ω * dot(view(std.K♯[3], k, :), view(F♯[3], :, k))
            end
        end
    end

    return nothing
end

#==========================================================================================#
#                               Telescopic divergence operator                             #

"""
    SSFVDivOperator(twopointflux, fvflux, blending)

Split-divergence operator in telescopic form. Only implemented for tensor-product elements.

The telescopic operator approach effectively turns the initial split formulation into a
sub-element finite volume scheme. The `twopointflux` and `fvflux` are both combined into
an entropy-stable interface flux that controls the volume dissipation through the
subcell interface Riemann solver.
"""
struct SSFVDivOperator{
    T<:AbstractNumericalFlux,
    F<:AbstractNumericalFlux,
    N<:AbstractNumericalFlux,
    RT,
} <: AbstractDivOperator
    tpflux::T
    fvflux::F
    blend::RT
    numflux::N
end

requires_subgrid(::SSFVDivOperator, _) = true

function _ssfv_compute_delta(b, c)
    # Fisher
    # δ = sqrt(b^2 + c)
    # δ = (δ - b) / δ

    # Ours
    δ = if b <= 0
        one(b)
    elseif b >= c
        zero(b)
    else
        (1 + cospi(b / c)) / 2
    end

    # Constant
    # return c

    # Limiting
    return max(δ, 0.5)
    # return δ
end

Base.@propagate_inbounds function _ssfv_gauss_flux_1d!(
    F̄c, F♯, F̃, Fnl, Fnr, Q, W, Fli, Fri, K♯mat, Ja, frames, Js,
    std, l, r, equation, op, idir,
)
    # Precompute two-point and entropy-projected fluxes
    for i in eachindex(std, idir)
        W[i] = vars_cons2entropy(Q[i], equation)
    end
    Wl = (l * W)'
    Wr = (r * W)'
    Ql = vars_entropy2cons(Wl, equation)
    Qr = vars_entropy2cons(Wr, equation)

    nl = frames[1].n .* Js[1]
    nr = frames[end].n .* Js[end]

    for i in eachindex(std, idir)
        Fli[i] = twopointflux(
            Q[i],
            Ql,
            view(Ja[i], :, idir),
            nl,
            equation,
            op.tpflux,
        )
        Fri[i] = twopointflux(
            Q[i],
            Qr,
            view(Ja[i], :, idir),
            nr,
            equation,
            op.tpflux,
        )
        F♯[i, i] = F̃[i]
        for l in (i + 1):size(std, idir)
            F♯[l, i] = twopointflux(
                Q[i],
                Q[l],
                view(Ja[i], :, idir),
                view(Ja[l], :, idir),
                equation,
                op.tpflux,
            )
            F♯[i, l] = F♯[l, i]
        end
    end

    # Complementary grid fluxes
    # The "mathematically correct" way to do this is to compute `F̄c[end]` also in the loop,
    # but this is a bit more accurate and efficient.
    F̄c[1] = -Fnl
    F̄c[end] = Fnr
    l_Fli = l * Fli
    r_Fri = r * Fri
    for i in 1:(size(std, idir) - 1)
        @views F̄c[i + 1] = F̄c[i] - dot(K♯mat[i, :], F♯[:, i]) -
                    l[i] * (Fli[i] - l_Fli - Fnl) +
                    r[i] * (Fri[i] - r_Fri + Fnr)
    end

    # Interior points
    for i in 2:size(std, idir)
        # FV fluxes
        Qln = rotate2face(Q[i - 1], frames[i], equation)
        Qrn = rotate2face(Q[i], frames[i], equation)
        Fn = numericalflux(Qln, Qrn, frames[i].n, equation, op.fvflux)
        F̄v = rotate2phys(Fn, frames[i], equation)
        F̄v *= Js[i]

        # Blending
        Wl = W[i - 1]
        Wr = W[i]
        b = dot(Wr - Wl, F̄c[i] - F̄v)
        δ = _ssfv_compute_delta(b, op.blend)
        F̄c[i] = (1 - δ) * F̄v + δ * F̄c[i]
    end
    return nothing
end

function surface_contribution!(
    dQ,
    Q,
    Fn,
    ielem,
    std::AbstractStdRegion{<:GaussQuadrature,ND},
    dg::DiscontinuousGalerkin,
    equation::AbstractEquation,
    op::SSFVDivOperator,
) where {
    ND
}
    # Unpack
    (; geometry) = dg

    # Buffers
    tid = Threads.threadid()
    F̃ = std.cache.block[tid][1]
    F♯ = std.cache.sharp[tid]

    # Volume fluxes (diagonal of F♯ matrices)
    Ja = geometry.elements[ielem].Ja
    @inbounds for i in eachindex(std)
        F = volumeflux(Q[i], equation)
        F̃[i, :] .= contravariant(F, Ja[i])
    end

    iface = dg.mesh.elements[ielem].faceinds
    facepos = dg.mesh.elements[ielem].facepos

    Qr = reshape(Q, size(std))
    dQr = reshape(dQ, size(std))
    Jar = reshape(Ja, size(std))
    F̃r = reshape(F̃, (size(std)..., ND))
    frames = geometry.subgrids[ielem].frames[]
    Js = geometry.subgrids[ielem].J[]

    if ND == 1
        F̄c = std.cache.subcell[tid][1]
        W = std.cache.state[tid][1]
        Fli = std.cache.state[tid][2]
        Fri = std.cache.state[tid][3]
        l = std.l[1]
        r = std.l[2]
        _ssfv_gauss_flux_1d!(
            F̄c, F♯[1], F̃,
            Fn.face[iface[1]][facepos[1]][1], Fn.face[iface[2]][facepos[2]][1],
            Q, W, Fli, Fri, std.K♯[1], Jar, frames[1], Js[1],
            std, l, r, equation, op, 1,
        )

        # Strong derivative
        @inbounds for i in eachindex(std)
            dQ[i] += F̄c[i] - F̄c[i + 1]
        end

    elseif ND == 2
        F̄c = std.cache.subcell[tid][1]
        ω = std.face.ω
        W = std.face.cache.state[tid][1]
        Fli = std.face.cache.state[tid][2]
        Fri = std.face.cache.state[tid][3]
        l = std.face.l[1]
        r = std.face.l[2]

        # X direction
        @inbounds for j in eachindex(std, 2)
            @views _ssfv_gauss_flux_1d!(
                F̄c, F♯[1], F̃r[:, j, 1],
                Fn.face[iface[1]][facepos[1]][j], Fn.face[iface[2]][facepos[2]][j],
                Qr[:, j], W, Fli, Fri, std.K♯[1],
                Jar[:, j], frames[1][:, j], Js[1][:, j],
                std, l, r, equation, op, 1,
            )

            # Strong derivative
            @inbounds for i in eachindex(std, 1)
                dQr[i, j] += (F̄c[i] - F̄c[i + 1]) * ω[j]
            end
        end

        # Y direction
        @inbounds for i in eachindex(std, 1)
            @views _ssfv_gauss_flux_1d!(
                F̄c, F♯[2], F̃r[i, :, 2],
                Fn.face[iface[3]][facepos[3]][i], Fn.face[iface[4]][facepos[4]][i],
                Qr[i, :], W, Fli, Fri, std.K♯[2],
                Jar[i, :], frames[2][i, :], Js[2][i, :],
                std, l, r, equation, op, 2,
            )

            # Strong derivative
            for j in eachindex(std, 2)
                dQr[i, j] += (F̄c[j] - F̄c[j + 1]) * ω[i]
            end
        end

    else # ND == 3
        F̄c = std.cache.subcell[tid][1]
        li = LinearIndices(std.face)
        ω = std.face.ω
        W = std.edge.cache.state[tid][1]
        Fli = std.edge.cache.state[tid][2]
        Fri = std.edge.cache.state[tid][3]
        l = std.edge.l[1]
        r = std.edge.l[2]

        # X direction
        @inbounds for k in eachindex(std, 3), j in eachindex(std, 2)
            ind = li[j, k]
            @views _ssfv_gauss_flux_1d!(
                F̄c, F♯[1], F̃r[:, j, k, 1],
                Fn.face[iface[1]][facepos[1]][ind], Fn.face[iface[2]][facepos[2]][ind],
                Qr[:, j, k], W, Fli, Fri, std.K♯[1], Jar[:, j, k],
                frames[1][:, j, k], Js[1][:, j, k],
                std, l, r, equation, op, 1,
            )

            # Strong derivative
            for i in eachindex(std, 1)
                dQr[i, j, k] += (F̄c[i] - F̄c[i + 1]) * ω[ind]
            end
        end

        # Y direction
        @inbounds for k in eachindex(std, 3), i in eachindex(std, 1)
            ind = li[i, k]
            @views _ssfv_gauss_flux_1d!(
                F̄c, F♯[2], F̃r[i, :, k, 2],
                Fn.face[iface[3]][facepos[3]][ind], Fn.face[iface[4]][facepos[4]][ind],
                Qr[i, :, k], W, Fli, Fri, std.K♯[2], Jar[i, :, k],
                frames[2][i, :, k], Js[2][i, :, k],
                std, l, r, equation, op, 2,
            )

            # Strong derivative
            for j in eachindex(std, 2)
                @views dQr[i, j, k] += (F̄c[j] - F̄c[j + 1]) * ω[ind]
            end
        end

        # Z direction
        @inbounds for j in eachindex(std, 2), i in eachindex(std, 1)
            ind = li[i, j]
            @views _ssfv_gauss_flux_1d!(
                F̄c, F♯[3], F̃r[i, j, :, 3],
                Fn.face[iface[5]][facepos[5]][ind], Fn.face[iface[6]][facepos[6]][ind],
                Qr[i, j, :], W, Fli, Fri, std.K♯[3], Jar[i, j, :],
                frames[i, j, :], Js[3][i, j, :],
                std, l, r, equation, op, 3,
            )

            # Strong derivative
            for k in eachindex(std, 3)
                @views dQr[i, j, k] += (F̄c[k] - F̄c[k + 1]) * ω[ind]
            end
        end
    end
    return nothing
end

Base.@propagate_inbounds function _ssfv_flux_1d!(
    F̄c, Q, Qmat, Ja, frames, Js, std, equation, op, idir,
)
    # Zero everything, boundaries will remain zero
    fill!(F̄c, zero(datatype(Q)))
    for i in 2:size(std, idir)

        # Two-point fluxes
        for k in i:size(std, idir), l in 1:(i - 1)
            F̄t = twopointflux(
                Q[l],
                Q[k],
                view(Ja[l], :, idir),
                view(Ja[k], :, idir),
                equation,
                op.tpflux,
            )
            F̄c[i] += 2Qmat[l, k] * F̄t
        end

        # FV fluxes
        Qln = rotate2face(Q[i - 1], frames[i], equation)
        Qrn = rotate2face(Q[i], frames[i], equation)
        Fn = numericalflux(Qln, Qrn, frames[i].n, equation, op.fvflux)
        F̄v = rotate2phys(Fn, frames[i], equation)
        F̄v *= Js[i]

        # Blending
        Wl = vars_cons2entropy(Q[i - 1], equation)
        Wr = vars_cons2entropy(Q[i], equation)
        b = dot(Wr - Wl, F̄c[i] - F̄v)
        δ = _ssfv_compute_delta(b, op.blend)
        F̄c[i] = (1 - δ) * F̄v + δ * F̄c[i]
    end
    return nothing
end

function volume_contribution!(
    dQ,
    Q,
    ielem,
    std::AbstractStdRegion{QT,ND},
    dg::DiscontinuousGalerkin,
    equation::AbstractEquation,
    op::SSFVDivOperator,
) where {
    QT,
    ND
}
    # Unpack
    (; geometry) = dg

    is_tensor_product(std) || throw(ArgumentError(
        "All the standard regions must be tensor-products."
    ))

    Qr = reshape(Q, size(std))
    dQr = reshape(dQ, size(std))
    Ja = reshape(geometry.elements[ielem].Ja, size(std))
    frames = geometry.subgrids[ielem].frames[]
    Js = geometry.subgrids[ielem].J[]

    tid = Threads.threadid()
    F̄ = std.cache.subcell[tid][1]

    if ND == 1
        _ssfv_flux_1d!(F̄, Q, std.Q[1], Ja, frames[1], Js[1], std, equation, op, 1)

        # Strong derivative
        @inbounds for i in eachindex(std)
            dQ[i] += F̄[i] - F̄[i + 1]
        end

    elseif ND == 2
        Qmat = std.face.Q[1]
        ω = std.face.ω

        # X direction
        @inbounds for j in eachindex(std, 2)
            @views _ssfv_flux_1d!(
                F̄, Qr[:, j], Qmat, Ja[:, j],
                frames[1][:, j], Js[1][:, j],
                std, equation, op, 1,
            )

            # Strong derivative
            for i in eachindex(std, 1)
                dQr[i, j] += (F̄[i] - F̄[i + 1]) * ω[j]
            end
        end

        # Y derivative
        @inbounds for i in eachindex(std, 1)
            @views _ssfv_flux_1d!(
                F̄, Qr[i, :], Qmat, Ja[i, :],
                frames[2][i, :], Js[2][i, :],
                std, equation, op, 2,
            )

            # Strong derivative
            for j in eachindex(std, 2)
                dQr[i, j] += (F̄[j] - F̄[j + 1]) * ω[i]
            end
        end

    else # ND == 3
        li = LinearIndices(std.face)
        Qmat = std.edge.Q[1]
        ω = std.face.ω

        # X direction
        @inbounds for k in eachindex(std, 3), j in eachindex(std, 2)
            @views _ssfv_flux_1d!(
                F̄, Qr[:, j, k], Qmat, Ja[:, j, k],
                frames[1][:, j, k], Js[1][:, j, k],
                std, equation, op, 1,
            )

            # Strong derivative
            ind = li[j, k]
            for i in eachindex(std, 1)
                dQr[i, j, k] += (F̄[i] - F̄[i + 1]) * ω[ind]
            end
        end

        # Y direction
        @inbounds for k in eachindex(std, 3), i in eachindex(std, 1)
            @views _ssfv_flux_1d!(
                F̄, Qr[i, :, k], Qmat, Ja[i, :, k],
                frames[2][i, :, k], Js[2][i, :, k],
                std, equation, op, 2,
            )

            # Strong derivative
            ind = li[i, k]
            for j in eachindex(std, 2)
                dQr[i, j, k] += (F̄[j] - F̄[j + 1]) * ω[ind]
            end
        end

        # Z direction
        @inbounds for j in eachindex(std, 2), i in eachindex(std, 1)
            @views _ssfv_flux_1d!(
                F̄, Qr[i, j, :], Qmat, Ja[i, j, :],
                frames[3][i, j, :], Js[3][i, j, :],
                std, equation, op, 3,
            )

            # Strong derivative
            ind = li[i, j]
            for k in eachindex(std, 3)
                dQr[i, j, k] += (F̄[k] - F̄[k + 1]) * ω[ind]
            end
        end
    end
    return nothing
end

function volume_contribution!(
    _,
    _,
    _,
    ::AbstractStdRegion{<:GaussQuadrature,ND},
    ::DiscontinuousGalerkin,
    ::AbstractEquation,
    ::SSFVDivOperator,
) where {
    ND
}
    # Do everything in the surface operator since we need the values of the Riemann problem
    return nothing
end
