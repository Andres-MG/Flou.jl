abstract type AbstractDivOperator <: AbstractOperator end

function surface_contribution!(
    dQ,
    Fn,
    ielem,
    std::AbstractStdRegion,
    dg::DiscontinuousGalerkin,
    ::AbstractDivOperator,
)
    # Unpack
    (; mesh) = dg

    iface = element(mesh, ielem).faceinds
    facepos = element(mesh, ielem).facepos

    @inbounds for (s, (face, pos)) in enumerate(zip(iface, facepos))
        mul!(
            dQ, std.lω[s], Fn[face][pos], -one(eltype(dQ)), one(eltype(dQ)),
        )
    end
    return nothing
end

#==========================================================================================#
#                                 Weak divergence operator                                 #

struct WeakDivOperator <: AbstractDivOperator end

function volume_contribution!(
    dQ,
    Q,
    ielem,
    std::AbstractStdRegion,
    dg::DiscontinuousGalerkin,
    ::WeakDivOperator,
)
    # Unpack
    (; dofhandler, physelem, equation) = dg

    ireg, ieloc = loc2reg(dofhandler, ielem)
    ndim = spatialdim(std)

    # Volume fluxes
    F̃ = MArray{Tuple{ndim,ndofs(std),nvariables(equation)},eltype(Q)}(undef)
    Ja = element(physelem, ielem).Ja
    @inbounds for i in eachindex(std)
        F = volumeflux(view(Q[ireg], i, :, ieloc), equation)
        for ivar in eachvariable(equation)
            F̃[:, i, ivar] = contravariant(view(F, :, ivar), Ja[i])
        end
    end

    # Weak derivative
    @inbounds for s in eachindex(std.K)
        mul!(
            dQ, std.K[s], view(F̃, s, :, :),
            one(eltype(dQ)), one(eltype(dQ)),
        )
    end
    return nothing
end

#==========================================================================================#
#                                Strong divergence operator                                #

struct StrongDivOperator <: AbstractDivOperator end

function volume_contribution!(
    dQ,
    Q,
    ielem,
    std::AbstractStdRegion,
    dg::DiscontinuousGalerkin,
    ::StrongDivOperator,
)
    # Unpack
    (; dofhandler, physelem, equation) = dg

    ireg, ieloc = loc2reg(dofhandler, ielem)
    ndim = spatialdim(std)

    # Volume fluxes
    F̃ = MArray{Tuple{ndim,ndofs(std),nvariables(equation)},eltype(Q)}(undef)
    Ja = element(physelem, ielem).Ja
    @inbounds for i in eachindex(std)
        F = volumeflux(view(Q[ireg], i, :, ieloc), equation)
        for ivar in eachvariable(equation)
            F̃[:, i, ivar] = contravariant(view(F, :, ivar), Ja[i])
        end
    end

    # Strong derivative
    @inbounds for s in eachindex(std.K)
        mul!(
            dQ, std.Ks[s], view(F̃, s, :, :),
            one(eltype(dQ)), one(eltype(dQ)),
        )
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
struct SplitDivOperator{T} <: AbstractDivOperator
    tpflux::T
end

function twopointflux end

function volume_contribution!(
    dQ,
    Q,
    ielem,
    std::AbstractStdRegion{ND,<:GaussLobattoQuadrature},
    dg::DiscontinuousGalerkin,
    op::SplitDivOperator,
) where {ND}
    # Unpack
    (; dofhandler, physelem, equation) = dg

    ireg, ieloc = loc2reg(dofhandler, ielem)
    is_tensor_product(std) || throw(ArgumentError(
        "All the standard regions must be tensor-products."
    ))

    # Buffers
    F♯ = [
        Array{eltype(Q),3}(undef, size(std, idir), ndofs(std), nvariables(equation))
        for idir in eachdirection(std)
    ]

    # Indexing
    ci = CartesianIndices(std)

    # Volume fluxes (diagonal of F♯ matrices)
    Ja = element(physelem, ielem).Ja
    @inbounds for i in eachindex(std)
        F = volumeflux(view(Q[ireg], i, :, ieloc), equation)
        for ivar in eachvariable(equation)
            F̃ = contravariant(view(F, :, ivar), Ja[i])
            for idir in eachdirection(std)
                F♯[idir][ci[i][idir], i, ivar] = F̃[idir]
            end
        end
    end

    # Two-point fluxes
    Qr = reshape(view(Q[ireg], :, :, ieloc), (size(std)..., nvariables(equation)))
    F♯r = [
        reshape(F♯[idir], (size(std, idir), size(std)..., nvariables(equation)))
        for idir in eachdirection(std)
    ]
    Jar = reshape(Ja, size(std))

    # 1D
    if ND == 1
        @inbounds for i in eachindex(std), l in (i + 1):ndofs(std)
            F♯r[1][l, i, :] = twopointflux(
                view(Qr, i, :),
                view(Qr, l, :),
                Jar[i],
                Jar[l],
                equation,
                op.tpflux,
            )
            @views copy!(F♯r[1][i, l, :], F♯r[1][l, i, :])
        end

    # 2D
    elseif ND == 2
        @inbounds for j in eachindex(std, 2), i in eachindex(std, 1)
            for l in (i + 1):size(std, 1)
                F♯r[1][l, i, j, :] = twopointflux(
                    view(Qr, i, j, :),
                    view(Qr, l, j, :),
                    view(Jar[i, j], :, 1),
                    view(Jar[l, j], :, 1),
                    equation,
                    op.tpflux,
                )
                @views copy!(F♯r[1][i, l, j, :], F♯r[1][l, i, j, :])
            end
            for l in (j + 1):size(std, 2)
                F♯r[2][l, i, j, :] = twopointflux(
                    view(Qr, i, j, :),
                    view(Qr, i, l, :),
                    view(Jar[i, j], :, 2),
                    view(Jar[i, l], :, 2),
                    equation,
                    op.tpflux,
                )
                @views copy!(F♯r[2][j, i, l, :], F♯r[2][l, i, j, :])
            end
        end

    # 3D
    else # ND == 3
        error("Not implemented yet!")
    end

    # Strong derivative
    @inbounds for v in eachvariable(equation)
        for i in eachindex(std)
            for (K♯, Fs) in zip(std.K♯, F♯)
                @views dQ[i, v] += dot(K♯[i, :], Fs[:, i, v])
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
struct SSFVDivOperator{T,F,RT} <: AbstractDivOperator
    tpflux::T
    fvflux::F
    blend::RT
end

requires_subgrid(::SSFVDivOperator) = true

function surface_contribution!(
    dQ,
    Fn,
    ielem,
    std::AbstractStdRegion,
    dg::DiscontinuousGalerkin,
    ::SSFVDivOperator,
)
    # Unpack
    (; mesh) = dg

    iface = element(mesh, ielem).faceinds
    facepos = element(mesh, ielem).facepos

    @inbounds for (s, (face, pos)) in enumerate(zip(iface, facepos))
        mul!(
            dQ, std.lωFV[s], Fn[face][pos], -one(eltype(dQ)), one(eltype(dQ)),
        )
    end
    return nothing
end

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

function _ssfv_flux_1d!(F̄c, Q, Qmat, Ja, ns, ts, bs, Js, std, equation, op, idir)
    # Boundaries
    fill!(@view(F̄c[1, :]), zero(eltype(Q)))
    fill!(@view(F̄c[end, :]), zero(eltype(Q)))

    # Interior points
    @inbounds for i in 2:size(std, idir)

        # Two-point fluxes
        fill!(view(F̄c, i, :), zero(eltype(Q)))
        for k in i:size(std, idir), l in 1:(i - 1)
            F̄t = twopointflux(
                view(Q, l, :),
                view(Q, k, :),
                view(Ja[l], :, idir),
                view(Ja[k], :, idir),
                equation,
                op.tpflux,
            )
            for v in eachvariable(equation)
                F̄c[i, v] += 2Qmat[l, k] * F̄t[v]
            end
        end

        # FV fluxes
        Qln = rotate2face(view(Q, i - 1, :), ns[i], ts[i], bs[i], equation)
        Qrn = rotate2face(view(Q, i, :), ns[i], ts[i], bs[i], equation)
        Fn = numericalflux(Qln, Qrn, ns[i], equation, op.fvflux)
        F̄v = rotate2phys(Fn, ns[i], ts[i], bs[i], equation) |> MVector
        F̄v .*= Js[i]

        # Blending
        Wl = vars_cons2entropy(view(Q, i - 1, :), equation)
        Wr = vars_cons2entropy(view(Q, i, :), equation)
        b = dot(Wr - Wl, view(F̄c, i, :) - F̄v)
        δ = _ssfv_compute_delta(b, op.blend)
        F̄c[i, :] = F̄v + δ .* (view(F̄c, i, :) - F̄v)
    end
end

function volume_contribution!(
    dQ,
    Q,
    ielem,
    std::AbstractStdRegion{ND,<:GaussLobattoQuadrature},
    dg::DiscontinuousGalerkin,
    op::SSFVDivOperator,
) where {ND}
    # Unpack
    (; dofhandler, physelem, equation) = dg

    ireg, ieloc = loc2reg(dofhandler, ielem)
    is_tensor_product(std) || throw(ArgumentError(
        "All the standard regions must be tensor-products."
    ))

    Ja = reshape(element(physelem, ielem).Ja, size(std))

    # 1D
    if ND == 1
        Qr = view(Q[ireg], :, :, ieloc)
        ns = elementgrid(physelem, ielem).n[1]
        ts = elementgrid(physelem, ielem).t[1]
        bs = elementgrid(physelem, ielem).b[1]
        Js = elementgrid(physelem, ielem).Jf[1]
        F̄ = MMatrix{(ndofs(std) + 1),nvariables(equation),eltype(Q)}(undef)
        _ssfv_flux_1d!(F̄, Qr, std.Q[1], Ja, ns, ts, bs, Js, std, equation, op, 1)

        # Strong derivative
        @inbounds for v in eachvariable(equation), i in eachindex(std)
            dQ[i, v] += F̄[i, v] - F̄[(i + 1), v]
        end

    # 2D
    elseif ND == 2
        Qr = reshape(view(Q[ireg], :, :, ieloc), (size(std)..., nvariables(equation)))
        dQr = reshape(dQ, (size(std)..., nvariables(equation)))

        # X direction
        ns = elementgrid(physelem, ielem).n[1]
        ts = elementgrid(physelem, ielem).t[1]
        bs = elementgrid(physelem, ielem).b[1]
        Js = elementgrid(physelem, ielem).Jf[1]
        Qmat = face(std, 3).Q[1]
        F̄ = MMatrix{size(std, 1) + 1,nvariables(equation),eltype(Q)}(undef)
        @inbounds for j in eachindex(std, 2)
            @views _ssfv_flux_1d!(
                F̄, Qr[:, j, :], Qmat, Ja[:, j], ns[:, j], ts[:, j], bs[:, j], Js[:, j],
                std, equation, op, 1,
            )

            # Strong derivative
            ωj = face(std, 1).ω[j]
            for v in eachvariable(equation), i in eachindex(std, 1)
                @views dQr[i, j, v] += (F̄[i, v] - F̄[(i + 1), v]) * ωj
            end
        end

        # Y derivative
        ns = elementgrid(physelem, ielem).n[2]
        ts = elementgrid(physelem, ielem).t[2]
        bs = elementgrid(physelem, ielem).b[2]
        Js = elementgrid(physelem, ielem).Jf[2]
        Qmat = face(std, 1).Q[1]
        F̄ = MMatrix{size(std, 2) + 1,nvariables(equation),eltype(Q)}(undef)
        @inbounds for i in eachindex(std, 1)
            @views _ssfv_flux_1d!(
                F̄, Qr[i, :, :], Qmat, Ja[i, :], ns[i, :], ts[i, :], bs[i, :], Js[i, :],
                std, equation, op, 2,
            )

            # Strong derivative
            ωi = face(std, 3).ω[i]
            for v in eachvariable(equation), j in eachindex(std, 2)
                @views dQr[i, j, v] += (F̄[j, v] - F̄[(j + 1), v]) * ωi
            end
        end

    # 3D
    else # ND == 3
        error("Not implemented yet!")
    end
    return nothing
end

function _ssfv_gauss_flux_1d!(
    F̄c, F♯, Fnl, Fnr, Q, W, Fli, Fri, K♯mat, Ja, ns, ts, bs, Js,
    std, l, r, equation, op, idir,
)
    # Precompute two-point and entropy-projected fluxes
    @inbounds for i in eachindex(std, idir)
        W[i, :] .= vars_cons2entropy(view(Q, i, :), equation)
    end
    Wl = (l * W)'
    Wr = (r * W)'
    Ql = vars_entropy2cons(Wl, equation)
    Qr = vars_entropy2cons(Wr, equation)

    nl = ns[1] .* Js[1]
    nr = ns[end] .* Js[end]
    Fl = transpose(nl) * volumeflux(Ql, equation)

    @inbounds for i in eachindex(std, idir)
        Fli[i, :] = twopointflux(
            view(Q, i, :),
            Ql,
            view(Ja[i], :, idir),
            nl,
            equation,
            op.tpflux,
        )
        Fri[i, :] = twopointflux(
            view(Q, i, :),
            Qr,
            view(Ja[i], :, idir),
            nr,
            equation,
            op.tpflux,
        )
        for l in (i + 1):size(std, idir)
            F♯[l, i, :] = twopointflux(
                view(Q, i, :),
                view(Q, l, :),
                view(Ja[i], :, idir),
                view(Ja[l], :, idir),
                equation,
                op.tpflux,
            )
            @views copy!(F♯[i, l, :], F♯[l, i, :])
        end
    end

    # Complementary grid fluxes
    F̄c[1, :] .= -Fl[1, :]
    @views for v in eachvariable(equation)
        l_Fli = l * Fli[:, v]
        r_Fri = r * Fri[:, v]
        for i in 1:(size(std, idir) - 1)
            F̄c[(i + 1), v] = F̄c[i, v] - dot(K♯mat[i, :], F♯[:, i, v]) -
                l[i] * (Fli[i, v] - l_Fli - Fnl[v]) +
                r[i] * (Fri[i, v] - r_Fri + Fnr[v])
            if i == 1
                F̄c[2, v] -= Fnl[v] - Fl[v]
            end
        end
    end

    # Boundaries
    fill!(@view(F̄c[1, :]), zero(eltype(Q)))
    fill!(@view(F̄c[end, :]), zero(eltype(Q)))

    # Interior points
    @inbounds for i in 2:size(std, idir)
        # FV fluxes
        Qln = rotate2face(view(Q, i - 1, :), ns[i], ts[i], bs[i], equation)
        Qrn = rotate2face(view(Q, i, :), ns[i], ts[i], bs[i], equation)
        Fn = numericalflux(Qln, Qrn, ns[i], equation, op.fvflux)
        F̄v = rotate2phys(Fn, ns[i], ts[i], bs[i], equation) |> MVector
        F̄v .*= Js[i]

        # Blending
        Wl = view(W, i - 1, :)
        Wr = view(W, i, :)
        b = dot(Wr - Wl, view(F̄c, i, :) - F̄v)
        δ = _ssfv_compute_delta(b, op.blend)
        F̄c[i, :] .= F̄v .+ δ .* (view(F̄c, i, :) .- F̄v)
    end
end

function volume_contribution!(
    dQ,
    Q,
    ielem,
    std::AbstractStdRegion{ND,<:GaussQuadrature},
    dg::DiscontinuousGalerkin,
    op::SSFVDivOperator,
) where {ND}
    # Unpack
    (; dofhandler, physelem, equation, Fn) = dg

    ireg, ieloc = loc2reg(dofhandler, ielem)
    is_tensor_product(std) || throw(ArgumentError(
        "All the standard regions must be tensor-products."
    ))

    if ND == 1
        F♯ = (
            Array{eltype(Q),3}(undef, size(std, 1), ndofs(std), nvariables(equation)),
        )
    elseif ND == 2
        F♯ = (
            Array{eltype(Q),3}(undef, size(std, 1), ndofs(std), nvariables(equation)),
            Array{eltype(Q),3}(undef, size(std, 2), ndofs(std), nvariables(equation)),
        )
    else # ND == 3
        F♯ = (
            Array{eltype(Q),3}(undef, size(std, 1), ndofs(std), nvariables(equation)),
            Array{eltype(Q),3}(undef, size(std, 2), ndofs(std), nvariables(equation)),
            Array{eltype(Q),3}(undef, size(std, 3), ndofs(std), nvariables(equation)),
        )
    end

    # Indexing
    ci = CartesianIndices(std)

    # Volume fluxes (diagonal of F♯ matrices)
    Ja = element(physelem, ielem).Ja
    @inbounds for i in eachindex(std)
        F = volumeflux(view(Q[ireg], i, :, ieloc), equation)
        for ivar in eachvariable(equation)
            F̃ = contravariant(view(F, :, ivar), Ja[i])
            for idir in eachdirection(std)
                F♯[idir][ci[i][idir], i, ivar] = F̃[idir]
            end
        end
    end

    Jar = reshape(Ja, size(std))

    # 1D
    if ND == 1
        iface = element(dg.mesh, ielem).faceinds
        facepos = element(dg.mesh, ielem).facepos

        Qr = view(Q[ireg], :, :, ieloc)
        ns = elementgrid(physelem, ielem).n[1]
        ts = elementgrid(physelem, ielem).t[1]
        bs = elementgrid(physelem, ielem).b[1]
        Js = elementgrid(physelem, ielem).Jf[1]

        F̄c = MMatrix{ndofs(std) + 1,nvariables(equation),eltype(Q)}(undef)
        W = MMatrix{ndofs(std),nvariables(equation),eltype(Q)}(undef)
        Fli = MMatrix{ndofs(std),nvariables(equation),eltype(Q)}(undef)
        Fri = MMatrix{ndofs(std),nvariables(equation),eltype(Q)}(undef)
        Fnl = view(Fn[iface[1]][facepos[1]], 1, :)
        Fnr = view(Fn[iface[2]][facepos[2]], 1, :)

        _ssfv_gauss_flux_1d!(
            F̄c, F♯[1], Fnl, Fnr, Qr, W, Fli, Fri, std.K♯[1], Jar, ns, ts, bs, Js,
            std, std.l[1], std.l[2], equation, op, 1,
        )

        # Strong derivative
        @inbounds for v in eachvariable(equation), i in eachindex(std)
            @views dQ[i, v] += F̄c[i, v] - F̄c[i + 1, v]
        end

    # 2D
    elseif ND == 2
        iface = element(dg.mesh, ielem).faceinds
        facepos = element(dg.mesh, ielem).facepos

        Qr = reshape(view(Q[ireg], :, :, ieloc), (size(std)..., nvariables(equation)))
        dQr = reshape(dQ, (size(std)..., nvariables(equation)))

        # X direction
        ns = elementgrid(physelem, ielem).n[1]
        ts = elementgrid(physelem, ielem).t[1]
        bs = elementgrid(physelem, ielem).b[1]
        Js = elementgrid(physelem, ielem).Jf[1]

        F̄c = MMatrix{size(std, 1) + 1,nvariables(equation),eltype(Q)}(undef)
        F♯r = reshape(F♯[1], (size(std, 1), size(std)..., nvariables(equation)))
        W = MMatrix{size(std, 1),nvariables(equation),eltype(Q)}(undef)
        Fli = MMatrix{size(std, 1),nvariables(equation),eltype(Q)}(undef)
        Fri = MMatrix{size(std, 1),nvariables(equation),eltype(Q)}(undef)
        K♯ = face(std, 3).K♯[1]
        l = face(std, 3).l[1]
        r = face(std, 3).l[2]

        @inbounds for j in eachindex(std, 2)
            Fnl = view(Fn[iface[1]][facepos[1]], j, :)
            Fnr = view(Fn[iface[2]][facepos[2]], j, :)
            @views _ssfv_gauss_flux_1d!(
            F̄c, F♯r[:, :, j, :], Fnl, Fnr, Qr[:, j, :], W, Fli, Fri, K♯,
                Jar[:, j], ns[:, j], ts[:, j], bs[:, j], Js[:, j],
                std, l, r, equation, op, 1,
            )

            # Strong derivative
            ωj = face(std, 1).ω[j]
            for v in eachvariable(equation), i in eachindex(std, 1)
                @views dQr[i, j, v] += (F̄c[i, v] - F̄c[(i + 1), v]) * ωj
            end
        end

        # Y direction
        ns = elementgrid(physelem, ielem).n[2]
        ts = elementgrid(physelem, ielem).t[2]
        bs = elementgrid(physelem, ielem).b[2]
        Js = elementgrid(physelem, ielem).Jf[2]

        F̄c = MMatrix{size(std, 2) + 1,nvariables(equation),eltype(Q)}(undef)
        F♯r = reshape(F♯[2], (size(std, 2), size(std)..., nvariables(equation)))
        W = MMatrix{size(std, 2),nvariables(equation),eltype(Q)}(undef)
        Fli = MMatrix{size(std, 2),nvariables(equation),eltype(Q)}(undef)
        Fri = MMatrix{size(std, 2),nvariables(equation),eltype(Q)}(undef)
        K♯ = face(std, 1).K♯[1]
        l = face(std, 1).l[1]
        r = face(std, 1).l[2]

        @inbounds for i in eachindex(std, 1)
            Fnl = view(Fn[iface[3]][facepos[3]], i, :)
            Fnr = view(Fn[iface[4]][facepos[4]], i, :)
            @views _ssfv_gauss_flux_1d!(
            F̄c, F♯r[:, i, :, :], Fnl, Fnr, Qr[i, :, :], W, Fli, Fri, K♯,
                Jar[i, :], ns[i, :], ts[i, :], bs[i, :], Js[i, :],
                std, l, r, equation, op, 2,
            )

            # Strong derivative
            ωi = face(std, 1).ω[i]
            for v in eachvariable(equation), j in eachindex(std, 2)
                @views dQr[i, j, v] += (F̄c[j, v] - F̄c[(j + 1), v]) * ωi
            end
        end

    # 3D
    else # ND == 3
        error("Not implemented yet!")
    end
    return nothing
end
