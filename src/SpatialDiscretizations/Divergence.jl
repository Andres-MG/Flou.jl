abstract type AbstractDivOperator <: AbstractOperator end

function volume_div_operator!  end
function surface_div_operator!  end

function surface_div_operator!(dQ, Fn, mesh, dh, stdvec, eq, ::AbstractDivOperator)
    ndim = spatialdim(mesh)
    for ireg in eachregion(dh)
        std = stdvec[ireg]

        @flouthreads for ieloc in eachelement(dh, ireg)
            ie = reg2loc(dh, ireg, ieloc)
            iface = element(mesh, ie).faceinds
            facepos = element(mesh, ie).facepos

            # Tensor-product elements
            if is_tensor_product(std)
                dQr = reshape(
                    view(dQ[ireg], :, :, ieloc),
                    (size(std)..., size(dQ[ireg], 2)),
                )

                # 1D
                if ndim == 1
                    @inbounds for v in eachvariable(eq)
                        for i in eachindex(std)
                            dQr[i, v] -= std.lω[1][i] * Fn[iface[1]][facepos[1]][1, v]
                            dQr[i, v] -= std.lω[2][i] * Fn[iface[2]][facepos[2]][1, v]
                        end
                    end

                # 2D
                elseif ndim == 2
                    @inbounds for v in eachvariable(eq), j in eachindex(std, 2)
                        for i in eachindex(std, 1)
                            dQr[i, j, v] -= std.lω[1][i, j] * Fn[iface[1]][facepos[1]][j, v]
                            dQr[i, j, v] -= std.lω[2][i, j] * Fn[iface[2]][facepos[2]][j, v]
                            dQr[i, j, v] -= std.lω[3][j, i] * Fn[iface[3]][facepos[3]][i, v]
                            dQr[i, j, v] -= std.lω[4][j, i] * Fn[iface[4]][facepos[4]][i, v]
                        end
                    end

                # 3D
                else # ndim == 3
                    error("Not implemented yet!")
                end

            # Non-tensor-product elements
            else
                @inbounds for v in eachvariable
                    for s in eachindex(iface, facepos), i in eachindex(std)
                        for k in eachindex(std)
                            dQ[ireg][i, v, ieloc] -= std.lω[s][i, k] *
                                                     Fn[iface[s]][facepos[s]][k, v]
                        end
                    end
                end
            end
        end
    end
end

#==========================================================================================#
#                                 Weak divergence operator                                 #

struct WeakDivOperator <: AbstractDivOperator end

function volume_div_operator!(dQ, Q, dh, stdvec, physelem, eq, ::WeakDivOperator)
    for ireg in eachregion(dh)
        std = stdvec[ireg]
        ndim = spatialdim(std)

        # Buffers
        Fb = @threadbuff Array{eltype(Q),3}(undef, ndim, nvariables(eq), ndofs(std))
        F̃b = @threadbuff Array{eltype(Q),3}(undef, ndofs(std), nvariables(eq), ndim)
        @flouthreads for ieloc in eachelement(dh, ireg)
            F = Fb[Threads.threadid()]
            F̃ = F̃b[Threads.threadid()]

            # Volume fluxes
            @inbounds for i in eachindex(std)
                @views volumeflux!(F[:, :, i], Q[ireg][i, :, ieloc], eq)
            end

            # Contravariant fluxes
            ie = reg2loc(dh, ireg, ieloc)
            for ivar in eachvariable(eq)
                for i in eachindex(std)
                    Ja = element(physelem, ie).Ja[i]
                    @views contravariant!(F̃[i, ivar, :], F[:, ivar, i], Ja)
                end
            end

            # Weak derivative
            # Tensor-product element
            if is_tensor_product(std)
                dQr = reshape(
                    view(dQ[ireg], :, :, ieloc),
                    (size(std)..., size(dQ[ireg], 2)),
                )
                F̃r = reshape(F̃, (size(std)..., size(F̃)[2:end]...))

                # 1D
                if ndim == 1
                    @inbounds for v in eachvariable(eq)
                        for k in eachindex(std), i in eachindex(std)
                            dQr[i, v] += std.K[1][i, k] * F̃r[k, v, 1]
                        end
                    end

                # 2D
                elseif ndim == 2
                    @inbounds for v in eachvariable(eq)
                        for j in eachindex(std, 2), i in eachindex(std, 1)
                            for k in eachindex(std, 1)
                                dQr[i, j, v] += std.K[1][i, j, k] * F̃r[k, j, v, 1]
                            end
                            for k in eachindex(std, 2)
                                dQr[i, j, v] += std.K[2][i, j, k] * F̃r[i, k, v, 2]
                            end
                        end
                    end

                # 3D
                else # ndim == 3
                    error("Not implemented yet!")
                end

            # Non-tensor-product element
            else
                @inbounds for s in eachindex(std.K)
                    for v in eachvariable(eq), i in eachindex(std)
                        for k in eachindex(std)
                            dQ[ireg][i, v, ieloc] += std.K[s][i, k] * F̃[k, v, s]
                        end
                    end
                end
            end
        end
    end
    return nothing
end

#==========================================================================================#
#                                Strong divergence operator                                #

struct StrongDivOperator <: AbstractDivOperator end

function volume_div_operator!(dQ, Q, dh, stdvec, physelem, eq, ::StrongDivOperator)
    for ireg in eachregion(dh)
        std = stdvec[ireg]
        ndim = spatialdim(std)

        # Buffers
        Fb = @threadbuff Array{eltype(Q),3}(undef, ndim, nvariables(eq), ndofs(std))
        F̃b = @threadbuff Array{eltype(Q),3}(undef, ndofs(std), nvariables(eq), ndim)
        @flouthreads for ieloc in eachelement(dh, ireg)
            F = Fb[Threads.threadid()]
            F̃ = F̃b[Threads.threadid()]

            # Volume fluxes
            @inbounds for i in eachindex(std)
                @views volumeflux!(F[:, :, i], Q[ireg][i, :, ieloc], eq)
            end

            # Contravariant fluxes
            ie = reg2loc(dh, ireg, ieloc)
            for ivar in eachvariable(eq)
                for i in eachindex(std)
                    Ja = element(physelem, ie).Ja[i]
                    @views contravariant!(F̃[i, ivar, :], F[:, ivar, i], Ja)
                end
            end

            # Weak derivative
            # Tensor-product element
            if is_tensor_product(std)
                dQr = reshape(
                    view(dQ[ireg], :, :, ieloc),
                    (size(std)..., size(dQ[ireg], 2)),
                )
                F̃r = reshape(F̃, (size(std)..., size(F̃)[2:end]...))

                # 1D
                if ndim == 1
                    @inbounds for v in eachvariable(eq)
                        for k in eachindex(std), i in eachindex(std)
                            dQr[i, v] += std.Ks[1][i, k] * F̃r[k, v, 1]
                        end
                    end

                # 2D
                elseif ndim == 2
                    @inbounds for v in eachvariable(eq)
                        for j in eachindex(std, 2), i in eachindex(std, 1)
                            for k in eachindex(std, 1)
                                dQr[i, j, v] += std.Ks[1][i, j, k] * F̃r[k, j, v, 1]
                            end
                            for k in eachindex(std, 2)
                                dQr[i, j, v] += std.Ks[2][i, j, k] * F̃r[i, k, v, 2]
                            end
                        end
                    end

                # 3D
                else # ndim == 3
                    error("Not implemented yet!")
                end

            # Non-tensor-product element
            else
                @inbounds for s in eachindex(std.K)
                    for v in eachvariable(eq), i in eachindex(std)
                        for k in eachindex(std)
                            dQ[ireg][i, v, ieloc] += std.Ks[s][i, k] * F̃[k, v, s]
                        end
                    end
                end
            end
        end
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

function volume_div_operator!(dQ, Q, dh, stdvec, physelem, eq, op::SplitDivOperator)
    for ireg in eachregion(dh)
        std = stdvec[ireg]
        ndim = spatialdim(std)
        is_tensor_product(std) ||
            throw(ArgumentError("All the standard regions must be tensor-products."))
        all(isa.(quadratures(std), GaussLobattoQuadrature)) ||
            throw(ArgumentError("Only GLL nodes admit a split-form formulation."))

        # Buffers
        Fb = @threadbuff Array{eltype(Q),3}(undef, ndim, nvariables(eq), ndofs(std))
        F♯b = @threadbuff [
            Array{eltype(Q),3}(
                undef, size(std, idir), ndofs(std), nvariables(eq)
            )
            for idir in eachdirection(std)
        ]
        @flouthreads for ieloc in eachelement(dh, ireg)
            F = Fb[Threads.threadid()]
            F̃ = MVector{ndim,eltype(Q)}(undef)
            F♯ = F♯b[Threads.threadid()]

            # Volume fluxes
            @inbounds for i in eachindex(std)
                @views volumeflux!(F[:, :, i], Q[ireg][i, :, ieloc], eq)
            end

            # Indexing
            ci = CartesianIndices(std)
            ie = reg2loc(dh, ireg, ieloc)

            # Contravariant fluxes
            Ja = element(physelem, ie).Ja
            for ivar in eachvariable(eq)
                for i in eachindex(std)
                    contravariant!(F̃, view(F, :, ivar, i), Ja[i])
                    for idir in eachdirection(std)
                        F♯[idir][ci[i][idir], i, ivar] = F̃[idir]
                    end
                end
            end

            # Two-point fluxes
            Qr = reshape(
                view(Q[ireg], :, :, ieloc),
                (size(std)..., size(Q[ireg], 2)),
            )
            dQr = reshape(
                view(dQ[ireg], :, :, ieloc),
                (size(std)..., size(dQ[ireg], 2)),
            )
            F♯r = [
                reshape(F♯[idir], (size(std, idir), size(std)..., size(F♯[idir]) |> last))
                for idir in eachdirection(std)
            ]
            Jar = reshape(Ja, size(std))

            # 1D
            if ndim == 1
                @inbounds for i in eachindex(std), l in (i + 1):ndofs(std)
                    twopointflux!(
                        view(F♯r[1], l, i, :),
                        view(Qr, i, :),
                        view(Qr, l, :),
                        Jar[i],
                        Jar[l],
                        eq,
                        eq.tpflux,
                    )
                    @views copy!(F♯r[1][i, l, :], F♯r[1][l, i, :])
                end

            # 2D
            elseif ndim == 2
                @inbounds for j in eachindex(std, 2), i in eachindex(std, 1)
                    for l in (i + 1):size(std, 1)
                        twopointflux!(
                            view(F♯r[1], l, i, j, :),
                            view(Qr, i, j, :),
                            view(Qr, l, j, :),
                            view(Jar[i, j], :, 1),
                            view(Jar[l, j], :, 1),
                            eq,
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
                            eq,
                            op.tpflux,
                        )
                        @views copy!(F♯r[2][j, i, l, :], F♯r[2][l, i, j, :])
                    end
                end

            # 3D
            else # ndim == 3
                error("Not implemented yet!")
            end

            # Strong derivative
            # 1D
            if ndim == 1
                @inbounds for v in eachvariable(eq), i in eachindex(std)
                    for k in eachindex(std)
                        dQr[i, v] += std.K♯[1][i, k] * F♯r[1][k, i, v]
                    end
                end

            # 2D
            elseif ndim == 2
                @inbounds for v in eachvariable(eq)
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
            else # ndim == 3
                error("Not implemented yet!")
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

function volume_div_operator!(dQ, Q, dh, stdvec, physelem, eq, op::SSFVDivOperator)
    for ireg in eachregion(dh)
        std = stdvec[ireg]
        ndim = spatialdim(std)
        is_tensor_product(std) ||
            throw(ArgumentError("All the standard regions must be tensor-products."))
        all(isa.(quadratures(std), GaussLobattoQuadrature)) ||
            throw(ArgumentError("Only GLL nodes admit a split-form formulation."))

        # Buffers
        F̄b = @threadbuff [
            Matrix{eltype(Q)}(
                undef, ndofs(std) + ndofs(std) ÷ size(std, idir), nvariables(eq)
            )
            for idir in eachdirection(std)
        ]

        @flouthreads for ieloc in eachelement(dh, ireg)
            F̄ = F̄b[Threads.threadid()]
            F̄t = MVector{nvariables(eq),eltype(Q)}(undef)
            F̄v = MVector{nvariables(eq),eltype(Q)}(undef)
            Qln = MVector{nvariables(eq),eltype(Q)}(undef)
            Qrn = MVector{nvariables(eq),eltype(Q)}(undef)
            Fn = MVector{nvariables(eq),eltype(Q)}(undef)

            # Fluxes at the subcell interfaces
            Qr = reshape(
                view(Q[ireg], :, :, ieloc),
                (size(std)..., size(Q[ireg], 2)),
            )
            dQr = reshape(
                view(dQ[ireg], :, :, ieloc),
                (size(std)..., size(dQ[ireg], 2)),
            )
            ie = reg2loc(dh, ireg, ieloc)
            Jar = reshape(element(physelem, ie).Ja, size(std))
            ns = elementgrid(physelem, ie).n
            ts = elementgrid(physelem, ie).t
            bs = elementgrid(physelem, ie).b
            Js = elementgrid(physelem, ie).Jf

            # 1D
            if ndim == 1
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
                            eq,
                            op.tpflux,
                        )
                        for v in eachvariable(eq)
                            F̄r[i, v] += 2 * std.Q[1][l, k] * F̄t[v]
                        end
                    end

                    # FV fluxes
                    numericalflux!(
                        F̄v,
                        view(Qr, (i - 1), :),
                        view(Qr, i, :),
                        ns[1][i],
                        eq,
                        op.fvflux,
                    )

                    # Blending
                    Wl = entropyvariables(view(Qr, (i - 1), :), eq)
                    Wr = entropyvariables(view(Qr, i, :), eq)
                    b = dot(Wr - Wl, view(F̄r, i, :) - F̄v)
                    δ = sqrt(b^2 + op.blend)
                    δ = (δ - b) / δ
                    F̄r[i, :] .= F̄v .+ δ .* (view(F̄r, i, :) .- F̄v)
                end

            # 2D
            elseif ndim == 2
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
                                eq,
                                op.tpflux,
                            )
                            for v in eachvariable(eq)
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
                            eq,
                        )
                        rotate2face!(
                            Qrn,
                            view(Qr, i, j, :),
                            ns[1][i, j],
                            ts[1][i, j],
                            bs[1][i, j],
                            eq,
                        )
                        numericalflux!(
                            Fn,
                            Qln,
                            Qrn,
                            ns[1][i, j],
                            eq,
                            op.fvflux,
                        )
                        rotate2phys!(F̄v, Fn, ns[1][i, j], ts[1][i, j], bs[1][i, j], eq)
                        F̄v .*= Js[1][i, j]

                        # Blending
                        Wl = entropyvariables(view(Qr, (i - 1), j, :), eq)
                        Wr = entropyvariables(view(Qr, i, j, :), eq)
                        b = dot(Wr .- Wl, view(F̄r[1], i, j, :) .- F̄v)
                        δ = sqrt(b^2 + op.blend)
                        δ = (δ - b) / δ
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
                                eq,
                                op.tpflux,
                            )
                            for v in eachvariable(eq)
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
                            eq,
                        )
                        rotate2face!(
                            Qrn,
                            view(Qr, i, j, :),
                            ns[2][j],
                            ts[2][j],
                            bs[2][j],
                            eq,
                        )
                        numericalflux!(
                            Fn,
                            Qln,
                            Qrn,
                            ns[2][j],
                            eq,
                            op.fvflux,
                        )
                        rotate2phys!(F̄v, Fn, ns[2][j], ts[2][j], bs[2][j], eq)
                        F̄v .*= Js[2][j]

                        # Blending
                        Wl = entropyvariables(view(Qr, i, (j - 1), :), eq)
                        Wr = entropyvariables(view(Qr, i, j, :), eq)
                        b = dot(Wr .- Wl, view(F̄r[2], i, j, :) .- F̄v)
                        δ = sqrt(b^2 + op.blend)
                        δ = (δ - b) / δ
                        F̄r[2][i, j, :] .= F̄v .+ δ .* (view(F̄r[2], i, j, :) .- F̄v)
                    end
                end

            # 3D
            else # ndim == 3
                error("Not implemented yet!")
            end

            # Strong derivative
            # 1D
            if ndim == 1
                @inbounds for v in eachvariable(eq)
                    for i in eachindex(std)
                        dQr[i, v] += F̄r[1][i, v] - F̄r[1][(i + 1), v]
                    end
                end

            # 2D
            elseif ndim == 2
                @inbounds for v in eachvariable(eq)
                    for j in eachindex(std, 2), i in eachindex(std, 1)
                        dQr[i, j, v] += (F̄r[1][i, j, v] - F̄r[1][(i + 1), j, v]) *
                                        face(std, 2).M[j, j]
                        dQr[i, j, v] += (F̄r[2][i, j, v] - F̄r[2][i, (j + 1), v]) *
                                        face(std, 1).M[i, i]
                    end
                end

            # 3D
            else # ndim == 3
                error("Not implemented yet!")
            end
        end
    end
    return nothing
end
