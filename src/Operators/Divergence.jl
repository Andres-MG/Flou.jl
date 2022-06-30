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
            dQ, std.lω[s], Fn[face][pos],
            -one(eltype(dQ)), one(eltype(dQ)),
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

    # Buffers
    F = MArray{Tuple{ndim,nvariables(equation),ndofs(std)},eltype(Q)}(undef)
    F̃ = MArray{Tuple{ndofs(std),nvariables(equation),ndim},eltype(Q)}(undef)

    # Volume fluxes
    @inbounds for i in eachindex(std)
        @views volumeflux!(F[:, :, i], Q[ireg][i, :, ieloc], equation)
    end

    # Contravariant fluxes
    ielem = reg2loc(dofhandler, ireg, ieloc)
    for ivar in eachvariable(equation)
        for i in eachindex(std)
            Ja = element(physelem, ielem).Ja[i]
            @views contravariant!(F̃[i, ivar, :], F[:, ivar, i], Ja)
        end
    end

    # Weak derivative
    @inbounds for s in eachindex(std.K)
        mul!(
            dQ, std.K[s], view(F̃, :, :, s),
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

    # Buffers
    F = MArray{Tuple{ndim,nvariables(equation),ndofs(std)},eltype(Q)}(undef)
    F̃ = MArray{Tuple{ndofs(std),nvariables(equation),ndim},eltype(Q)}(undef)

    # Volume fluxes
    @inbounds for i in eachindex(std)
        @views volumeflux!(F[:, :, i], Q[ireg][i, :, ieloc], equation)
    end

    # Contravariant fluxes
    for ivar in eachvariable(equation)
        for i in eachindex(std)
            Ja = element(physelem, ielem).Ja[i]
            @views contravariant!(F̃[i, ivar, :], F[:, ivar, i], Ja)
        end
    end

    # Weak derivative
    @inbounds for s in eachindex(std.K)
        mul!(
            dQ, std.K[s], view(F̃, :, :, s),
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

function twopointflux! end

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
    F = MArray{Tuple{ND,nvariables(equation),ndofs(std)},eltype(Q)}(undef)
    F̃ = MVector{ND,eltype(Q)}(undef)
    F♯ = [
        Array{eltype(Q),3}(undef, size(std, idir), ndofs(std), nvariables(equation))
        for idir in eachdirection(std)
    ]

    # Volume fluxes
    @inbounds for i in eachindex(std)
        @views volumeflux!(F[:, :, i], Q[ireg][i, :, ieloc], equation)
    end

    # Indexing
    ci = CartesianIndices(std)

    # Contravariant fluxes
    Ja = element(physelem, ielem).Ja
    for ivar in eachvariable(equation)
        for i in eachindex(std)
            contravariant!(F̃, view(F, :, ivar, i), Ja[i])
            for idir in eachdirection(std)
                F♯[idir][ci[i][idir], i, ivar] = F̃[idir]
            end
        end
    end

    # Two-point fluxes
    Qr = reshape(view(Q[ireg], :, :, ieloc), (size(std)..., size(Q[ireg], 2)))
    dQr = reshape(dQ, (size(std)..., size(dQ, 2)))
    F♯r = [
        reshape(F♯[idir], (size(std, idir), size(std)..., size(F♯[idir]) |> last))
        for idir in eachdirection(std)
    ]
    Jar = reshape(Ja, size(std))

    # 1D
    if ND == 1
        @inbounds for i in eachindex(std), l in (i + 1):ndofs(std)
            twopointflux!(
                view(F♯r[1], l, i, :),
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
                twopointflux!(
                    view(F♯r[1], l, i, j, :),
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
                twopointflux!(
                    view(F♯r[2], l, i, j, :),
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
    # 1D
    if ND == 1
        @inbounds for v in eachvariable(equation), i in eachindex(std)
            for k in eachindex(std)
                dQr[i, v] += std.K♯[1][i, k] * F♯r[1][k, i, v]
            end
        end

    # 2D
    elseif ND == 2
        @inbounds for v in eachvariable(equation)
            for j in eachindex(std, 2), i in eachindex(std, 1)
                for k in eachindex(std, 1)
                    dQr[i, j, v] += std.K♯[1][i, j, k] * F♯r[1][k, i, j, v]
                end
                for k in eachindex(std, 2)
                    dQr[i, j, v] += std.K♯[2][i, j, k] * F♯r[2][k, i, j, v]
                end
            end
        end

    # 3D
    else # ND == 3
        error("Not implemented yet!")
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

    # Buffers
    F̄ = [
        Matrix{eltype(Q)}(
            undef,
            ndofs(std) + ndofs(std) ÷ size(std, idir),
            nvariables(equation),
        )
        for idir in eachdirection(std)
    ]
    F̄t = MVector{nvariables(equation),eltype(Q)}(undef)
    F̄v = MVector{nvariables(equation),eltype(Q)}(undef)
    Qln = MVector{nvariables(equation),eltype(Q)}(undef)
    Qrn = MVector{nvariables(equation),eltype(Q)}(undef)
    Fn = MVector{nvariables(equation),eltype(Q)}(undef)

    # Fluxes at the subcell interfaces
    Qr = reshape(view(Q[ireg], :, :, ieloc), (size(std)..., size(Q[ireg], 2)))
    dQr = reshape(dQ, (size(std)..., size(dQ, 2)))
    Jar = reshape(element(physelem, ielem).Ja, size(std))
    ns = elementgrid(physelem, ielem).n
    ts = elementgrid(physelem, ielem).t
    bs = elementgrid(physelem, ielem).b
    Js = elementgrid(physelem, ielem).Jf

    # 1D
    if ND == 1
        F̄r = F̄[1]

        # Boundaries
        fill!(@view(F̄r[1, :]), zero(eltype(Q)))
        fill!(@view(F̄r[end, :]), zero(eltype(Q)))

        # Interior points
        @inbounds for i in 2:ndofs(std)

            # Two-point fluxes
            fill!(view(F̄r, i, :), zero(eltype(Q)))
            for k in i:ndofs(std), l in 1:(i - 1)
                twopointflux!(
                    F̄t,
                    view(Qr, l, :),
                    view(Qr, k, :),
                    Ja[l],
                    Ja[k],
                    equation,
                    op.tpflux,
                )
                for v in eachvariable(equation)
                    F̄r[i, v] += 2 * std.Q[1][l, k] * F̄t[v]
                end
            end

            # FV fluxes
            numericalflux!(
                F̄v,
                view(Qr, (i - 1), :),
                view(Qr, i, :),
                ns[1][i],
                equation,
                op.fvflux,
            )

            # Blending
            Wl = vars_cons2entropy(view(Qr, (i - 1), :), equation)
            Wr = vars_cons2entropy(view(Qr, i, :), equation)
            b = dot(Wr - Wl, view(F̄r, i, :) - F̄v)
            δ = _ssfv_compute_delta(b, op.blend)
            F̄r[i, :] .= F̄v .+ δ .* (view(F̄r, i, :) .- F̄v)
        end

    # 2D
    elseif ND == 2
        F̄r = [
            reshape(F̄[1], (size(std, 1) + 1, size(std, 2), size(F̄[1]) |> last)),
            reshape(F̄[2], (size(std, 1), size(std, 2) + 1, size(F̄[2]) |> last)),
        ]

        @inbounds for j in eachindex(std, 2)
            # Boundaries
            fill!(@view(F̄r[1][1, j, :]), zero(eltype(Q)))
            fill!(@view(F̄r[1][end, j, :]), zero(eltype(Q)))

            # Interior points
            for i in 2:size(std, 1)

                # Two-point fluxes
                fill!(view(F̄r[1], i, j, :), zero(eltype(Q)))
                for k in i:size(std, 1), l in 1:(i - 1)
                    twopointflux!(
                        F̄t,
                        view(Qr, l, j, :),
                        view(Qr, k, j, :),
                        view(Jar[l, j], :, 1),
                        view(Jar[k, j], :, 1),
                        equation,
                        op.tpflux,
                    )
                    for v in eachvariable(equation)
                        F̄r[1][i, j, v] += 2 * std.Q[1][l, k] * F̄t[v]
                    end
                end

                # FV fluxes
                rotate2face!(
                    Qln,
                    view(Qr, (i - 1), j, :),
                    ns[1][i, j],
                    ts[1][i, j],
                    bs[1][i, j],
                    equation,
                )
                rotate2face!(
                    Qrn,
                    view(Qr, i, j, :),
                    ns[1][i, j],
                    ts[1][i, j],
                    bs[1][i, j],
                    equation,
                )
                numericalflux!(
                    Fn,
                    Qln,
                    Qrn,
                    ns[1][i, j],
                    equation,
                    op.fvflux,
                )
                rotate2phys!(F̄v, Fn, ns[1][i, j], ts[1][i, j], bs[1][i, j], equation)
                F̄v .*= Js[1][i, j]

                # Blending
                Wl = vars_cons2entropy(view(Qr, (i - 1), j, :), equation)
                Wr = vars_cons2entropy(view(Qr, i, j, :), equation)
                b = dot(Wr .- Wl, view(F̄r[1], i, j, :) .- F̄v)
                δ = _ssfv_compute_delta(b, op.blend)
                F̄r[1][i, j, :] .= F̄v .+ δ .* (view(F̄r[1], i, j, :) .- F̄v)
            end
        end
        @inbounds for i in eachindex(std, 1)
            # Boundaries
            fill!(@view(F̄r[2][i, 1, :]), zero(eltype(Q)))
            fill!(@view(F̄r[2][i, end, :]), zero(eltype(Q)))

            # Interior points
            for j in 2:size(std, 2)

                # Two-point fluxes
                fill!(view(F̄r[2], i, j, :), zero(eltype(Q)))
                for k in j:size(std, 2), l in 1:(j - 1)
                    twopointflux!(
                        F̄t,
                        view(Qr, i, l, :),
                        view(Qr, i, k, :),
                        view(Jar[i, l], :, 2),
                        view(Jar[i, k], :, 2),
                        equation,
                        op.tpflux,
                    )
                    for v in eachvariable(equation)
                        F̄r[2][i, j, v] += 2 * std.Q[2][l, k] * F̄t[v]
                    end
                end

                # FV fluxes
                rotate2face!(
                    Qln,
                    view(Qr, i, (j - 1), :),
                    ns[2][j],
                    ts[2][j],
                    bs[2][j],
                    equation,
                )
                rotate2face!(
                    Qrn,
                    view(Qr, i, j, :),
                    ns[2][j],
                    ts[2][j],
                    bs[2][j],
                    equation,
                )
                numericalflux!(
                    Fn,
                    Qln,
                    Qrn,
                    ns[2][j],
                    equation,
                    op.fvflux,
                )
                rotate2phys!(F̄v, Fn, ns[2][j], ts[2][j], bs[2][j], equation)
                F̄v .*= Js[2][j]

                # Blending
                Wl = vars_cons2entropy(view(Qr, i, (j - 1), :), equation)
                Wr = vars_cons2entropy(view(Qr, i, j, :), equation)
                b = dot(Wr .- Wl, view(F̄r[2], i, j, :) .- F̄v)
                δ = _ssfv_compute_delta(b, op.blend)
                F̄r[2][i, j, :] .= F̄v .+ δ .* (view(F̄r[2], i, j, :) .- F̄v)
            end
        end

    # 3D
    else # ND == 3
        error("Not implemented yet!")
    end

    # Strong derivative
    # 1D
    if ND == 1
        @inbounds for v in eachvariable(equation)
            for i in eachindex(std)
                dQr[i, v] += F̄r[1][i, v] - F̄r[1][(i + 1), v]
            end
        end

    # 2D
    elseif ND == 2
        @inbounds for v in eachvariable(equation)
            for j in eachindex(std, 2), i in eachindex(std, 1)
                dQr[i, j, v] += (F̄r[1][i, j, v] - F̄r[1][(i + 1), j, v]) *
                                face(std, 2).ω[j]
                dQr[i, j, v] += (F̄r[2][i, j, v] - F̄r[2][i, (j + 1), v]) *
                                face(std, 1).ω[i]
            end
        end

    # 3D
    else # ND == 3
        error("Not implemented yet!")
    end
    return nothing
end

function _ssfv_compute_delta(b, c)
    δ = sqrt(b^2 + c)
    δ = (δ - b) / δ
    # δ = if b <= 0
    #     one(b)
    # elseif b >= c
    #     zero(b)
    # else
    #     (1 + cospi(b / c)) / 2
    # end
    return max(δ, 0.4)
end

# function volume_contribution!(
#     dQ,
#     Q,
#     ielem,
#     std::AbstractStdRegion{ND,<:GaussQuadrature},
#     dg::DiscontinuousGalerkin,
#     op::SplitDivOperator,
# ) where {ND}
#     # Unpack
#     (; dofhandler, physelem, equation) = dg

#     ireg, ieloc = loc2reg(dofhandler, ielem)
#     is_tensor_product(std) || throw(ArgumentError(
#         "All the standard regions must be tensor-products."
#     ))

#     # Buffers
#     F = MArray{Tuple{ND,nvariables(equation),ndofs(std)},eltype(Q)}(undef)
#     F̃ = MVector{ND,eltype(Q)}(undef)
#     F♯ = [
#         Array{eltype(Q),3}(undef, size(std, idir), ndofs(std), nvariables(equation))
#         for idir in eachdirection(std)
#     ]

#     # Volume fluxes
#     @inbounds for i in eachindex(std)
#         @views volumeflux!(F[:, :, i], Q[ireg][i, :, ieloc], equation)
#     end

#     # Indexing
#     ci = CartesianIndices(std)

#     # Contravariant fluxes
#     Ja = element(physelem, ielem).Ja
#     for ivar in eachvariable(equation)
#         for i in eachindex(std)
#             contravariant!(F̃, view(F, :, ivar, i), Ja[i])
#             for idir in eachdirection(std)
#                 F♯[idir][ci[i][idir], i, ivar] = F̃[idir]
#             end
#         end
#     end

#     # Two-point fluxes
#     Qr = reshape(view(Q[ireg], :, :, ieloc), (size(std)..., size(Q[ireg], 2)))
#     dQr = reshape(dQ, (size(std)..., size(dQ, 2)))
#     F♯r = [
#         reshape(F♯[idir], (size(std, idir), size(std)..., size(F♯[idir]) |> last))
#         for idir in eachdirection(std)
#     ]
#     Jar = reshape(Ja, size(std))

#     # 1D
#     if ND == 1
#         # Entropy-projected variables and fluxes
#         W = MMatrix{ndofs(std),nvariables(equation),eltype(Q)}(undef)
#         @inbounds for i in eachindex(std)
#             W[i, :] .= vars_cons2entropy(view(Q[ireg], i, :, ieloc), equation)
#         end
#         Ŵl = std.l[1]' * W
#         Ŵr = std.l[2]' * W
#         Q̂l = vars_entropy2cons(Ŵl, equation)
#         Q̂r = vars_entropy2cons(Ŵr, equation)

#         Fli = MMatrix{ndofs(std),nvariables(equation),eltype(Q)}(undef)
#         Fri = MMatrix{ndofs(std),nvariables(equation),eltype(Q)}(undef)
#         @inbounds for i in eachindex(std)
#             twopointflux!(
#                 view(Fli, i, :),
#                 view(Qr, i, :),
#                 Q̂l,
#                 Jar[i],
#                 Jar[i],
#                 equation,
#                 op.tpflux,
#             )
#             twopointflux!(
#                 view(Fri, i, :),
#                 view(Qr, i, :),
#                 Q̂r,
#                 Jar[i],
#                 Jar[i],
#                 equation,
#                 op.tpflux,
#             )
#             for l in (i + 1):ndofs(std)
#                 twopointflux!(
#                     view(F♯r[1], l, i, :),
#                     view(Qr, i, :),
#                     view(Qr, l, :),
#                     Jar[i],
#                     Jar[l],
#                     equation,
#                     op.tpflux,
#                 )
#                 @views copy!(F♯r[1][i, l, :], F♯r[1][l, i, :])
#             end
#         end

#     # 2D
#     elseif ND == 2
#         @inbounds for j in eachindex(std, 2), i in eachindex(std, 1)
#             for l in (i + 1):size(std, 1)
#                 twopointflux!(
#                     view(F♯r[1], l, i, j, :),
#                     view(Qr, i, j, :),
#                     view(Qr, l, j, :),
#                     view(Jar[i, j], :, 1),
#                     view(Jar[l, j], :, 1),
#                     equation,
#                     op.tpflux,
#                 )
#                 @views copy!(F♯r[1][i, l, j, :], F♯r[1][l, i, j, :])
#             end
#             for l in (j + 1):size(std, 2)
#                 twopointflux!(
#                     view(F♯r[2], l, i, j, :),
#                     view(Qr, i, j, :),
#                     view(Qr, i, l, :),
#                     view(Jar[i, j], :, 2),
#                     view(Jar[i, l], :, 2),
#                     equation,
#                     op.tpflux,
#                 )
#                 @views copy!(F♯r[2][j, i, l, :], F♯r[2][l, i, j, :])
#             end
#         end

#     # 3D
#     else # ND == 3
#         error("Not implemented yet!")
#     end

#     # Strong derivative
#     # 1D
#     if ND == 1
#         @inbounds for v in eachvariable(equation), i in eachindex(std)
#             for k in eachindex(std)
#                 @views dQr[i, v] += std.K♯[1][i, k] * F♯r[1][k, i, v] +
#                     std.l[1][i] * (Fli[i, v] - std.l[1]' * Fli[:, v]) -
#                     std.l[2][i] * (Fri[i, v] - std.l[2]' * Fri[:, v])
#             end
#         end

#     # 2D
#     elseif ND == 2
#         @inbounds for v in eachvariable(equation)
#             for j in eachindex(std, 2), i in eachindex(std, 1)
#                 for k in eachindex(std, 1)
#                     dQr[i, j, v] += std.K♯[1][i, j, k] * F♯r[1][k, i, j, v]
#                 end
#                 for k in eachindex(std, 2)
#                     dQr[i, j, v] += std.K♯[2][i, j, k] * F♯r[2][k, i, j, v]
#                 end
#             end
#         end

#     # 3D
#     else # ND == 3
#         error("Not implemented yet!")
#     end
#     return nothing
# end
