abstract type AbstractDivOperator <: AbstractOperator end

function surface_contribution!(
    dQ,
    _,
    Fn,
    ielem,
    std::AbstractStdRegion,
    dg::DiscontinuousGalerkin,
    ::AbstractDivOperator,
)
    # Unpack
    (; mesh) = dg

    iface = mesh.elements[ielem].faceinds
    facepos = mesh.elements[ielem].facepos

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
    (; geometry, equation) = dg

    ndim = get_spatialdim(std)

    # Volume fluxes
    F̃ = MArray{Tuple{ndim,ndofs(std),nvariables(equation)},eltype(Q)}(undef)
    Ja = geometry.elements[ielem].Ja
    @inbounds for i in eachindex(std)
        F = volumeflux(view(Q, i, :), equation)
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
    (; geometry, equation) = dg

    ndim = get_spatialdim(std)

    # Volume fluxes
    F̃ = MArray{Tuple{ndim,ndofs(std),nvariables(equation)},eltype(Q)}(undef)
    Ja = geometry.elements[ielem].Ja
    @inbounds for i in eachindex(std)
        F = volumeflux(view(Q, i, :), equation)
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

requires_subgrid(::SplitDivOperator) = true

function twopointflux end

function _split_gauss_deriv_1d!(
    dQ, Fnl, Fnr, Q, W, Fli, Fri, Ja, frames, Js,
    std, l, r, ω, equation, op, idir,
)
    # Precompute two-point and entropy-projected fluxes
    @inbounds for i in eachindex(std, idir)
        W[i, :] .= vars_cons2entropy(view(Q, i, :), equation)
    end
    Wl = (l * W)'
    Wr = (r * W)'
    Ql = vars_entropy2cons(Wl, equation)
    Qr = vars_entropy2cons(Wr, equation)

    nl = frames[1].n .* Js[1]
    nr = frames[end].n .* Js[end]

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
    end

    # Complementary grid fluxes
    @inbounds for v in eachvariable(equation)
        l_Fli = l * @view(Fli[:, v])
        r_Fri = r * @view(Fri[:, v])
        for i in eachindex(std, idir)
            dQ[i, v] += l[i] * ω * (Fli[i, v] - l_Fli - Fnl[v]) -
                        r[i] * ω * (Fri[i, v] - r_Fri + Fnr[v])
        end
    end
    return nothing
end

function surface_contribution!(
    dQ,
    Q,
    Fn,
    ielem,
    std::AbstractStdRegion{ND,<:GaussQuadrature},
    dg::DiscontinuousGalerkin,
    op::SplitDivOperator,
) where {ND}
    # Unpack
    (; mesh, geometry, equation) = dg

    rt = eltype(Q)
    iface = mesh.elements[ielem].faceinds
    facepos = mesh.elements[ielem].facepos

    Qr = reshape(Q, (size(std)..., nvariables(equation)))
    dQr = reshape(dQ, (size(std)..., nvariables(equation)))
    Ja = reshape(geometry.elements[ielem].Ja, size(std))
    frames = geometry.subgrids[ielem].frames[]
    Js = geometry.subgrids[ielem].J[]

    if ND == 1
        W = MMatrix{ndofs(std),nvariables(equation),rt}(undef)
        Fli = MMatrix{ndofs(std),nvariables(equation),rt}(undef)
        Fri = MMatrix{ndofs(std),nvariables(equation),rt}(undef)
        l = std.l[1]
        r = std.l[2]
        _split_gauss_deriv_1d!(
            dQ, view(Fn[iface[1]][facepos[1]], 1, :), view(Fn[iface[2]][facepos[2]], 1, :),
            Q, W, Fli, Fri, Ja, frames[1], Js[1], std, l, r, one(rt), equation, op, 1,
        )

    elseif ND == 2
        # X direction
        W = MMatrix{size(std, 1),nvariables(equation),rt}(undef)
        Fli = MMatrix{size(std, 1),nvariables(equation),rt}(undef)
        Fri = MMatrix{size(std, 1),nvariables(equation),rt}(undef)
        face = std.faces[3]
        ω = std.faces[1].ω
        @inbounds for j in eachindex(std, 2)
            @views _split_gauss_deriv_1d!(
                dQr[:, j, :],
                Fn[iface[1]][facepos[1]][j, :], Fn[iface[2]][facepos[2]][j, :],
                Qr[:, j, :], W, Fli, Fri, Ja[:, j], frames[1][:, j], Js[1][:, j],
                std, face.l[1], face.l[2], ω[j], equation, op, 1,
            )
        end

        # Y direction
        W = MMatrix{size(std, 2),nvariables(equation),rt}(undef)
        Fli = MMatrix{size(std, 2),nvariables(equation),rt}(undef)
        Fri = MMatrix{size(std, 2),nvariables(equation),rt}(undef)
        face = std.faces[1]
        ω = std.faces[3].ω
        @inbounds for i in eachindex(std, 1)
            @views _split_gauss_deriv_1d!(
                dQr[i, :, :],
                Fn[iface[3]][facepos[3]][i, :], Fn[iface[4]][facepos[4]][i, :],
                Qr[i, :, :], W, Fli, Fri, Ja[i, :], frames[2][i, :], Js[2][i, :],
                std, face.l[1], face.l[2], ω[i], equation, op, 2,
            )
        end

    else # ND == 3
        # X direction
        W = MMatrix{size(std, 1),nvariables(equation),rt}(undef)
        Fli = MMatrix{size(std, 1),nvariables(equation),rt}(undef)
        Fri = MMatrix{size(std, 1),nvariables(equation),rt}(undef)
        li = LinearIndices(std.faces[1])
        edge = std.edges[1]
        ω = std.faces[1].ω
        @inbounds for k in eachindex(std, 3), j in eachindex(std, 2)
            ind = li[j, k]
            @views _split_gauss_deriv_1d!(
                dQr[:, j, k, :],
                Fn[iface[1]][facepos[1]][ind, :], Fn[iface[2]][facepos[2]][ind, :],
                Qr[:, j, k, :], W, Fli, Fri, Ja[:, j, k], frames[1][:, j, k],
                Js[1][:, j, k], std, edge.l[1], edge.l[2], ω[ind], equation, op, 1,
            )
        end

        # Y direction
        W = MMatrix{size(std, 2),nvariables(equation),rt}(undef)
        Fli = MMatrix{size(std, 2),nvariables(equation),rt}(undef)
        Fri = MMatrix{size(std, 2),nvariables(equation),rt}(undef)
        li = LinearIndices(std.faces[3])
        edge = std.edges[2]
        ω = std.faces[3].ω
        @inbounds for k in eachindex(std, 3), i in eachindex(std, 1)
            ind = li[i, k]
            @views _split_gauss_deriv_1d!(
                dQr[i, :, k, :],
                Fn[iface[3]][facepos[3]][ind, :], Fn[iface[4]][facepos[4]][ind, :],
                Qr[i, :, k, :], W, Fli, Fri, Ja[i, :, k], frames[2][i, :, k],
                Js[2][i, :, k], std, edge.l[1], edge.l[2], ω[ind], equation, op, 2,
            )
        end

        # Z direction
        W = MMatrix{size(std, 3),nvariables(equation),rt}(undef)
        Fli = MMatrix{size(std, 3),nvariables(equation),rt}(undef)
        Fri = MMatrix{size(std, 3),nvariables(equation),rt}(undef)
        li = LinearIndices(std.faces[5])
        edge = std.edges[3]
        ω = std.faces[5].ω
        @inbounds for j in eachindex(std, 2), i in eachindex(std, 1)
            ind = li[i, j]
            @views _split_gauss_deriv_1d!(
                dQr[i, j, :, :],
                Fn[iface[5]][facepos[5]][ind, :], Fn[iface[6]][facepos[6]][ind, :],
                Qr[i, j, :, :], W, Fli, Fri, Ja[i, j, :], frames[3][i, j, :],
                Js[3][i, j, :], std, edge.l[1], edge.l[2], ω[ind], equation, op, 3,
            )
        end
    end
    return nothing
end

function _split_flux_1d!(F♯, Q, Ja, std, equation, op, idir)
    @inbounds for i in eachindex(std, idir), l in (i + 1):size(std, idir)
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
    return nothing
end

function volume_contribution!(
    dQ,
    Q,
    ielem,
    std::AbstractStdRegion{ND},
    dg::DiscontinuousGalerkin,
    op::SplitDivOperator,
) where {ND}
    # Unpack
    (; geometry, equation) = dg

    is_tensor_product(std) || throw(ArgumentError(
        "All the standard regions must be tensor-products."
    ))

    # Buffers
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
    Ja = geometry.elements[ielem].Ja
    @inbounds for i in eachindex(std)
        F = volumeflux(view(Q, i, :), equation)
        for ivar in eachvariable(equation)
            F̃ = contravariant(view(F, :, ivar), Ja[i])
            for idir in eachdirection(std)
                F♯[idir][ci[i][idir], i, ivar] = F̃[idir]
            end
        end
    end

    # Two-point fluxes
    Qr = reshape(Q, (size(std)..., nvariables(equation)))
    Jar = reshape(Ja, size(std))

    if ND == 1
        _split_flux_1d!(F♯[1], Q, Jar, std, equation, op, 1)

    elseif ND == 2
        F♯r = (
            reshape(F♯[1], (size(std, 1), size(std)..., nvariables(equation))),
            reshape(F♯[2], (size(std, 2), size(std)..., nvariables(equation))),
        )
        @inbounds for j in eachindex(std, 2)
            @views _split_flux_1d!(
                F♯r[1][:, :, j, :], Qr[:, j, :], Jar[:, j], std, equation, op, 1,
            )
        end
        @inbounds for i in eachindex(std, 1)
            @views _split_flux_1d!(
                F♯r[2][:, i, :, :], Qr[i, :, :], Jar[i, :], std, equation, op, 2,
            )
        end

    else # ND == 3
        F♯r = (
            reshape(F♯[1], (size(std, 1), size(std)..., nvariables(equation))),
            reshape(F♯[2], (size(std, 2), size(std)..., nvariables(equation))),
            reshape(F♯[3], (size(std, 3), size(std)..., nvariables(equation))),
        )
        @inbounds for k in eachindex(std, 3), j in eachindex(std, 2)
            @views _split_flux_1d!(
                F♯r[1][:, :, j, k, :], Qr[:, j, k, :], Jar[:, j, k], std, equation, op, 1,
            )
        end
        @inbounds for k in eachindex(std, 3), i in eachindex(std, 1)
            @views _split_flux_1d!(
                F♯r[2][:, i, :, k, :], Qr[i, :, k, :], Jar[i, :, k], std, equation, op, 2,
            )
        end
        @inbounds for j in eachindex(std, 2), i in eachindex(std, 1)
            @views _split_flux_1d!(
                F♯r[3][:, i, j, :, :], Qr[i, j, :, :], Jar[i, j, :], std, equation, op, 3,
            )
        end
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

function _ssfv_gauss_flux_1d!(
    F̄c, F♯, Fnl, Fnr, Q, W, Fli, Fri, K♯mat, Ja, frames, Js,
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

    nl = frames[1].n .* Js[1]
    nr = frames[end].n .* Js[end]
    Fl = nl' * volumeflux(Ql, equation)

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
    F̄c[1, :] .= -view(Fl, 1, :)
    @inbounds for v in eachvariable(equation)
        l_Fli = l * @view(Fli[:, v])
        r_Fri = r * @view(Fri[:, v])
        @views for i in 1:(size(std, idir)-1)
            F̄c[i + 1, v] = F̄c[i, v] - dot(K♯mat[i, :], F♯[:, i, v]) -
                           l[i] * (Fli[i, v] - l_Fli - Fnl[v]) +
                           r[i] * (Fri[i, v] - r_Fri + Fnr[v])
            if i == 1
                F̄c[2, v] -= Fnl[v] - Fl[v]
            end
        end
    end

    # Boundaries
    F̄c[1, :] .= -Fnl
    F̄c[end, :] .= Fnr

    # Interior points
    @inbounds for i in 2:size(std, idir)
        # FV fluxes
        Qln = rotate2face(view(Q, i - 1, :), frames[i], equation)
        Qrn = rotate2face(view(Q, i, :), frames[i], equation)
        Fn = numericalflux(Qln, Qrn, frames[i].n, equation, op.fvflux)
        F̄v = rotate2phys(Fn, frames[i], equation) |> MVector
        F̄v .*= Js[i]

        # Blending
        Wl = view(W, i - 1, :)
        Wr = view(W, i, :)
        b = dot(Wr - Wl, view(F̄c, i, :) - F̄v)
        δ = _ssfv_compute_delta(b, op.blend)
        F̄c[i, :] .= F̄v .+ δ .* (view(F̄c, i, :) .- F̄v)
    end
    return nothing
end

function surface_contribution!(
    dQ,
    Q,
    Fn,
    ielem,
    std::AbstractStdRegion{ND,<:GaussQuadrature},
    dg::DiscontinuousGalerkin,
    op::SSFVDivOperator,
) where {ND}
    # Unpack
    (; geometry, equation) = dg

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
    Ja = geometry.elements[ielem].Ja
    @inbounds for i in eachindex(std)
        F = volumeflux(view(Q, i, :), equation)
        for ivar in eachvariable(equation)
            F̃ = contravariant(view(F, :, ivar), Ja[i])
            for idir in eachdirection(std)
                F♯[idir][ci[i][idir], i, ivar] = F̃[idir]
            end
        end
    end

    iface = dg.mesh.elements[ielem].faceinds
    facepos = dg.mesh.elements[ielem].facepos

    Qr = reshape(Q, (size(std)..., nvariables(equation)))
    dQr = reshape(dQ, (size(std)..., nvariables(equation)))
    Jar = reshape(Ja, size(std))
    frames = geometry.subgrids[ielem].frames[]
    Js = geometry.subgrids[ielem].J[]

    if ND == 1
        F̄c = MMatrix{ndofs(std) + 1,nvariables(equation),eltype(Q)}(undef)
        W = MMatrix{ndofs(std),nvariables(equation),eltype(Q)}(undef)
        Fli = MMatrix{ndofs(std),nvariables(equation),eltype(Q)}(undef)
        Fri = MMatrix{ndofs(std),nvariables(equation),eltype(Q)}(undef)
        _ssfv_gauss_flux_1d!(
            F̄c, F♯[1],
            view(Fn[iface[1]][facepos[1]], 1, :), view(Fn[iface[2]][facepos[2]], 1, :),
            Q, W, Fli, Fri, std.K♯[1], Jar, frames[1], Js[1],
            std, std.l[1], std.l[2], equation, op, 1,
        )

        # Strong derivative
        @inbounds for v in eachvariable(equation), i in eachindex(std)
            @views dQ[i, v] += F̄c[i, v] - F̄c[i + 1, v]
        end

    elseif ND == 2
        # X direction
        F̄c = MMatrix{size(std, 1) + 1,nvariables(equation),eltype(Q)}(undef)
        F♯r = reshape(F♯[1], (size(std, 1), size(std)..., nvariables(equation)))
        W = MMatrix{size(std, 1),nvariables(equation),eltype(Q)}(undef)
        Fli = MMatrix{size(std, 1),nvariables(equation),eltype(Q)}(undef)
        Fri = MMatrix{size(std, 1),nvariables(equation),eltype(Q)}(undef)
        face = std.faces[3]
        ω = std.faces[1].ω
        @inbounds for j in eachindex(std, 2)
            @views _ssfv_gauss_flux_1d!(
                F̄c, F♯r[:, :, j, :],
                Fn[iface[1]][facepos[1]][j, :], Fn[iface[2]][facepos[2]][j, :],
                Qr[:, j, :], W, Fli, Fri, face.K♯[1],
                Jar[:, j], frames[1][:, j], Js[1][:, j],
                std, face.l[1], face.l[2], equation, op, 1,
            )

            # Strong derivative
            for v in eachvariable(equation), i in eachindex(std, 1)
                @views dQr[i, j, v] += (F̄c[i, v] - F̄c[i + 1, v]) * ω[j]
            end
        end

        # Y direction
        F̄c = MMatrix{size(std, 2) + 1,nvariables(equation),eltype(Q)}(undef)
        F♯r = reshape(F♯[2], (size(std, 2), size(std)..., nvariables(equation)))
        W = MMatrix{size(std, 2),nvariables(equation),eltype(Q)}(undef)
        Fli = MMatrix{size(std, 2),nvariables(equation),eltype(Q)}(undef)
        Fri = MMatrix{size(std, 2),nvariables(equation),eltype(Q)}(undef)
        face = std.faces[1]
        ω = std.faces[3].ω
        @inbounds for i in eachindex(std, 1)
            @views _ssfv_gauss_flux_1d!(
                F̄c, F♯r[:, i, :, :],
                Fn[iface[3]][facepos[3]][i, :], Fn[iface[4]][facepos[4]][i, :],
                Qr[i, :, :], W, Fli, Fri, face.K♯[1],
                Jar[i, :], frames[2][i, :], Js[2][i, :],
                std, face.l[1], face.l[2], equation, op, 2,
            )

            # Strong derivative
            for v in eachvariable(equation), j in eachindex(std, 2)
                @views dQr[i, j, v] += (F̄c[j, v] - F̄c[j + 1, v]) * ω[i]
            end
        end

    else # ND == 3
        # X direction
        F̄c = MMatrix{size(std, 1) + 1,nvariables(equation),eltype(Q)}(undef)
        F♯r = reshape(F♯[1], (size(std, 1), size(std)..., nvariables(equation)))
        W = MMatrix{size(std, 1),nvariables(equation),eltype(Q)}(undef)
        Fli = MMatrix{size(std, 1),nvariables(equation),eltype(Q)}(undef)
        Fri = MMatrix{size(std, 1),nvariables(equation),eltype(Q)}(undef)
        edge = std.edges[1]
        face = std.faces[1]
        li = LinearIndices(face)
        ω = face.ω
        @inbounds for k in eachindex(std, 3), j in eachindex(std, 2)
            ind = li[j, k]
            @views _ssfv_gauss_flux_1d!(
                F̄c, F♯r[:, :, j, k, :],
                Fn[iface[1]][facepos[1]][ind, :], Fn[iface[2]][facepos[2]][ind, :],
                Qr[:, j, k, :], W, Fli, Fri, edge.K♯[1], Jar[:, j, k],
                frames[1][:, j, k], Js[1][:, j, k],
                std, edge.l[1], edge.l[2], equation, op, 1,
            )

            # Strong derivative
            for v in eachvariable(equation), i in eachindex(std, 1)
                @views dQr[i, j, k, v] += (F̄c[i, v] - F̄c[i + 1, v]) * ω[ind]
            end
        end

        # Y direction
        F̄c = MMatrix{size(std, 2) + 1,nvariables(equation),eltype(Q)}(undef)
        F♯r = reshape(F♯[2], (size(std, 2), size(std)..., nvariables(equation)))
        W = MMatrix{size(std, 2),nvariables(equation),eltype(Q)}(undef)
        Fli = MMatrix{size(std, 2),nvariables(equation),eltype(Q)}(undef)
        Fri = MMatrix{size(std, 2),nvariables(equation),eltype(Q)}(undef)
        edge = std.edges[2]
        face = std.faces[3]
        li = LinearIndices(face)
        ω = face.ω
        @inbounds for k in eachindex(std, 3), i in eachindex(std, 1)
            ind = li[i, k]
            @views _ssfv_gauss_flux_1d!(
                F̄c, F♯r[:, i, :, k, :],
                Fn[iface[3]][facepos[3]][ind, :], Fn[iface[4]][facepos[4]][ind, :],
                Qr[i, :, k, :], W, Fli, Fri, edge.K♯[1], Jar[i, :, k],
                frames[2][i, :, k], Js[2][i, :, k],
                std, edge.l[1], edge.l[2], equation, op, 2,
            )

            # Strong derivative
            for v in eachvariable(equation), j in eachindex(std, 2)
                @views dQr[i, j, k, v] += (F̄c[j, v] - F̄c[j + 1, v]) * ω[ind]
            end
        end

        # Z direction
        F̄c = MMatrix{size(std, 3) + 1,nvariables(equation),eltype(Q)}(undef)
        F♯r = reshape(F♯[3], (size(std, 3), size(std)..., nvariables(equation)))
        W = MMatrix{size(std, 3),nvariables(equation),eltype(Q)}(undef)
        Fli = MMatrix{size(std, 3),nvariables(equation),eltype(Q)}(undef)
        Fri = MMatrix{size(std, 3),nvariables(equation),eltype(Q)}(undef)
        edge = std.edges[3]
        face = std.faces[5]
        li = LinearIndices(face)
        ω = face.ω
        @inbounds for j in eachindex(std, 2), i in eachindex(std, 1)
            ind = li[i, j]
            @views _ssfv_gauss_flux_1d!(
                F̄c, F♯r[:, i, j, :, :],
                Fn[iface[5]][facepos[5]][ind, :], Fn[iface[6]][facepos[6]][ind, :],
                Qr[i, j, :, :], W, Fli, Fri, edge.K♯[1], Jar[i, j, :],
                frames[i, j, :], Js[3][i, j, :],
                std, edge.l[1], edge.l[2], equation, op, 3,
            )

            # Strong derivative
            for v in eachvariable(equation), k in eachindex(std, 3)
                @views dQr[i, j, k, v] += (F̄c[k, v] - F̄c[k + 1, v]) * ω[ind]
            end
        end
    end
    return nothing
end

function _ssfv_flux_1d!(F̄c, Q, Qmat, Ja, frames, Js, std, equation, op, idir)
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
        Qln = rotate2face(view(Q, i - 1, :), frames[i], equation)
        Qrn = rotate2face(view(Q, i, :), frames[i], equation)
        Fn = numericalflux(Qln, Qrn, frames[i].n, equation, op.fvflux)
        F̄v = rotate2phys(Fn, frames[i], equation) |> MVector
        F̄v .*= Js[i]

        # Blending
        Wl = vars_cons2entropy(view(Q, i - 1, :), equation)
        Wr = vars_cons2entropy(view(Q, i, :), equation)
        b = dot(Wr - Wl, view(F̄c, i, :) - F̄v)
        δ = _ssfv_compute_delta(b, op.blend)
        F̄c[i, :] = F̄v + δ .* (view(F̄c, i, :) - F̄v)
    end
    return nothing
end

function volume_contribution!(
    dQ,
    Q,
    ielem,
    std::AbstractStdRegion{ND},
    dg::DiscontinuousGalerkin,
    op::SSFVDivOperator,
) where {ND}
    # Unpack
    (; geometry, equation) = dg

    is_tensor_product(std) || throw(ArgumentError(
        "All the standard regions must be tensor-products."
    ))

    Qr = reshape(Q, (size(std)..., nvariables(equation)))
    dQr = reshape(dQ, (size(std)..., nvariables(equation)))
    Ja = reshape(geometry.elements[ielem].Ja, size(std))
    frames = geometry.subgrids[ielem].frames[]
    Js = geometry.subgrids[ielem].J[]

    if ND == 1
        F̄ = MMatrix{(ndofs(std) + 1),nvariables(equation),eltype(Q)}(undef)
        _ssfv_flux_1d!(F̄, Q, std.Q[1], Ja, frames[1], Js[1], std, equation, op, 1)

        # Strong derivative
        @inbounds for v in eachvariable(equation), i in eachindex(std)
            dQ[i, v] += F̄[i, v] - F̄[i + 1, v]
        end

    elseif ND == 2
        # X direction
        Qmat = std.faces[3].Q[1]
        F̄ = MMatrix{size(std, 1) + 1,nvariables(equation),eltype(Q)}(undef)
        ω = std.faces[1].ω
        @inbounds for j in eachindex(std, 2)
            @views _ssfv_flux_1d!(
                F̄, Qr[:, j, :], Qmat, Ja[:, j],
                frames[1][:, j], Js[1][:, j],
                std, equation, op, 1,
            )

            # Strong derivative
            for v in eachvariable(equation), i in eachindex(std, 1)
                @views dQr[i, j, v] += (F̄[i, v] - F̄[i + 1, v]) * ω[j]
            end
        end

        # Y derivative
        Qmat = std.faces[1].Q[1]
        F̄ = MMatrix{size(std, 2) + 1,nvariables(equation),eltype(Q)}(undef)
        ω = std.faces[3].ω
        @inbounds for i in eachindex(std, 1)
            @views _ssfv_flux_1d!(
                F̄, Qr[i, :, :], Qmat, Ja[i, :],
                frames[2][i, :], Js[2][i, :],
                std, equation, op, 2,
            )

            # Strong derivative
            for v in eachvariable(equation), j in eachindex(std, 2)
                @views dQr[i, j, v] += (F̄[j, v] - F̄[j + 1, v]) * ω[i]
            end
        end

    else # ND == 3
        # X direction
        Qmat = std.edges[1].Q[1]
        F̄ = MMatrix{size(std, 1) + 1,nvariables(equation),eltype(Q)}(undef)
        face = std.faces[1]
        li = LinearIndices(face)
        @inbounds for k in eachindex(std, 3), j in eachindex(std, 2)
            @views _ssfv_flux_1d!(
                F̄, Qr[:, j, k, :], Qmat, Ja[:, j, k],
                frames[1][:, j, k], Js[1][:, j, k],
                std, equation, op, 1,
            )

            # Strong derivative
            ind = li[j, k]
            for v in eachvariable(equation), i in eachindex(std, 1)
                @views dQr[i, j, k, v] += (F̄[i, v] - F̄[i + 1, v]) * face.ω[ind]
            end
        end

        # Y direction
        Qmat = std.edges[2].Q[1]
        F̄ = MMatrix{size(std, 2) + 1,nvariables(equation),eltype(Q)}(undef)
        face = std.faces[3]
        li = LinearIndices(face)
        @inbounds for k in eachindex(std, 3), i in eachindex(std, 1)
            @views _ssfv_flux_1d!(
                F̄, Qr[i, :, k, :], Qmat, Ja[i, :, k],
                frames[2][i, :, k], Js[2][i, :, k],
                std, equation, op, 2,
            )

            # Strong derivative
            ind = li[i, k]
            for v in eachvariable(equation), j in eachindex(std, 2)
                @views dQr[i, j, k, v] += (F̄[j, v] - F̄[j + 1, v]) * face.ω[ind]
            end
        end

        # Z direction
        Qmat = std.edges[3].Q[1]
        F̄ = MMatrix{size(std, 3) + 1,nvariables(equation),eltype(Q)}(undef)
        face = std.faces[5]
        li = LinearIndices(face)
        @inbounds for j in eachindex(std, 2), i in eachindex(std, 1)
            @views _ssfv_flux_1d!(
                F̄, Qr[i, j, :, :], Qmat, Ja[i, j, :],
                frames[3][i, j, :], Js[3][i, j, :],
                std, equation, op, 3,
            )

            # Strong derivative
            ind = li[i, j]
            for v in eachvariable(equation), k in eachindex(std, 3)
                @views dQr[i, j, k, v] += (F̄[k, v] - F̄[k + 1, v]) * face.ω[ind]
            end
        end
    end
    return nothing
end

function volume_contribution!(
    _,
    _,
    _,
    _::AbstractStdRegion{ND,<:GaussQuadrature},
    _::DiscontinuousGalerkin,
    _::SSFVDivOperator,
) where {ND}
    # Do everything in the surface operator since we need the values of the Riemann problem
    return nothing
end
