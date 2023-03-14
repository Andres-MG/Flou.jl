# Copyright (C) 2023 Andrés Mateo Gabín
#
# This file is part of Flou.jl.
#
# Flou.jl is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# Flou.jl is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Flou.jl. If
# not, see <https://www.gnu.org/licenses/>.

abstract type AbstractDivOperator <: AbstractOperator end

"""
    requires_subgrid(divergence, std)

Return `true` if the specified `divergence` operator needs the allocation of the
subcell grid.
"""
function requires_subgrid(::AbstractDivOperator, ::AbstractStdRegion)
    return false
end

function _volumeflux!(F̃, Q, Ja, equation::HyperbolicEquation)
    @inbounds for i in eachindex(Q.dofs)
        Mᵀ = Ja[i]
        F = volumeflux(Q.dofs[i], equation)
        for id in eachdim(F̃)
            F̃.dofs[i, id] = contravariant(F, Mᵀ, id)
        end
    end
    return nothing
end

#==========================================================================================#
#                                FR reconstruction operator                                #

function _surface_contribution!(
    dQ,
    Fn,
    ielem,
    std::AbstractStdRegion,
    disc::MultielementDisc,
    ::AbstractDivOperator,
)
    # Unpack
    (; mesh) = disc

    # Element-local data
    faces = mesh.elements[ielem].faceinds
    sides = mesh.elements[ielem].facepos
    dQe = dQ.elements[ielem]

    # Use a tensor-product reconstruction when possible
    if is_tensor_product(std)
        nd = spatialdim(std)
        if nd >= 1
            inds = tpdofs(std, Val(1))
            fl = Fn.faces[faces[1]].sides[sides[1]]
            fr = Fn.faces[faces[2]].sides[sides[2]]
            _reconstruction_tensorproduct!(dQe, fl, fr, inds, std.∂g)
        end
        if nd >= 2
            inds = tpdofs(std, Val(2))
            fl = Fn.faces[faces[3]].sides[sides[3]]
            fr = Fn.faces[faces[4]].sides[sides[4]]
            _reconstruction_tensorproduct!(dQe, fl, fr, inds, std.∂g)
        end
        if nd >= 3
            inds = tpdofs(std, Val(3))
            fl = Fn.faces[faces[5]].sides[sides[5]]
            fr = Fn.faces[faces[6]].sides[sides[6]]
            _reconstruction_tensorproduct!(dQe, fl, fr, inds, std.∂g)
        end

    # Otherwise, use the general reconstruction
    else
        rt = datatype(dQ)
        @inbounds for (∂g, face, side) in zip(std.∂g, faces, sides)
            Fid = Fn.faces[face].sides[side]
            mul!(dQe.dofs, ∂g, Fid.dofs, -one(rt), one(rt))
        end
    end

    return nothing
end

function _reconstruction_tensorproduct!(dQ, fl, fr, allinds, ∂g)
    ∂gl, ∂gr = ∂g
    @inbounds for (inds, fli, fri) in zip(allinds, fl.dofs, fr.dofs)
        for (ii, i) in enumerate(inds)
            dQ.dofs[i] -= ∂gl[ii] * fli + ∂gr[ii] * fri
        end
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
    disc::MultielementDisc,
    equation::AbstractEquation,
    ::StrongDivOperator,
)
    # Element-local data
    dQe = dQ.elements[ielem]
    Qe = Q.elements[ielem]
    F̃e = std.cache.block[Threads.threadid()][1]

    # Volume fluxes
    Ja = disc.geometry.elements[ielem].metric
    _volumeflux!(F̃e, Qe, Ja, equation)

    # Strong divergence for tensor-product elements
    if is_tensor_product(std)
        nd = spatialdim(std)
        if nd >= 1
            inds = tpdofs(std, Val(1))
            _vol_strongdiv_tensorproduct!(dQe, F̃e, inds, std.Ds, 1)
        end
        if nd >= 2
            inds = tpdofs(std, Val(2))
            _vol_strongdiv_tensorproduct!(dQe, F̃e, inds, std.Ds, 2)
        end
        if nd >= 3
            inds = tpdofs(std, Val(3))
            _vol_strongdiv_tensorproduct!(dQe, F̃e, inds, std.Ds, 3)
        end

    # Strong divergence for non-tensor-product elements
    else
        rt = datatype(dQe)
        @inbounds for id in eachdim(std)
            mul!(dQe.dofs, std.Ds[id], view(F̃e.dofs, :, id), -one(rt), one(rt))
        end
    end

    return nothing
end

function _vol_strongdiv_tensorproduct!(dQ, F̃, allinds, Ds, dir)
    rt = datatype(dQ)
    @inbounds for i in allinds
        @views mul!(dQ.dofs[i], Ds, F̃.dofs[i, dir], -one(rt), one(rt))
    end
    return nothing
end


function surface_contribution!(
    dQ,
    _,
    ielem,
    std::AbstractStdRegion,
    disc::MultielementDisc,
    ::AbstractEquation,
    op::StrongDivOperator,
)
    return _surface_contribution!(dQ, disc.cache.Fn, ielem, std, disc, op)
end

#==========================================================================================#
#                                 Split divergence operator                                #

"""
    SplitDivOperator([tpflux=numflux.avg], numflux)

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

function SplitDivOperator(numflux)
    return SplitDivOperator(numflux.avg, numflux)
end

function requires_subgrid(::SplitDivOperator, std::AbstractStdRegion)
    return !(std |> basis |> hasboundaries)
end

function volume_contribution!(
    dQ,
    Q,
    ielem,
    std::AbstractStdRegion,
    disc::MultielementDisc,
    equation::AbstractEquation,
    op::SplitDivOperator,
)
    # Non-tensor-product elements are not implemented
    if !is_tensor_product(std)
        throw(ArgumentError(
            "Split-divergence operator not implemented for non-tensor-product elements."
        ))
    end

    # Element-local data
    tid = Threads.threadid()
    dQe = dQ.elements[ielem]
    Qe = Q.elements[ielem]
    F̃e = std.cache.block[tid][1]
    F♯e = std.cache.sharp[tid][1]

    # Flux (diagonal terms)
    Ja = disc.geometry.elements[ielem].metric
    _volumeflux!(F̃e, Qe, Ja, equation)

    # Two-point fluxes and split divergence
    nd = spatialdim(std)
    if nd >= 1
        inds = tpdofs(std, Val(1))
        _flux_splitdiv_tensorproduct!(F♯e, F̃e, Qe, Ja, inds, equation, op, 1)
        _vol_splitdiv_tensorproduct!(dQe, F♯e, inds, std.D♯)
    end
    if nd >= 2
        inds = tpdofs(std, Val(2))
        _flux_splitdiv_tensorproduct!(F♯e, F̃e, Qe, Ja, inds, equation, op, 2)
        _vol_splitdiv_tensorproduct!(dQe, F♯e, inds, std.D♯)
    end
    if nd >= 3
        inds = tpdofs(std, Val(3))
        _flux_splitdiv_tensorproduct!(F♯e, F̃e, Qe, Ja, inds, equation, op, 3)
        _vol_splitdiv_tensorproduct!(dQe, F♯e, inds, std.D♯)
    end

    return nothing
end

function _flux_splitdiv_tensorproduct!(F♯, F̃, Q, Ja, allinds, equation, op, dir)
    @inbounds for inds in allinds
        for (ii, i) in enumerate(inds)
            # Copy diagonal entries
            F♯.dofs[ii, i] = F̃.dofs[i, dir]

            # Compute off-diagonal entries
            npts = length(inds)
            for il in (ii + 1):npts
                l = inds[il]
                F♯.dofs[il, i] = twopointflux(
                    Q.dofs[i],
                    Q.dofs[l],
                    view(Ja[i], :, dir),
                    view(Ja[l], :, dir),
                    equation,
                    op.tpflux,
                )
                F♯.dofs[ii, l] = F♯.dofs[il, i]
            end
        end
    end
    return nothing
end

function _vol_splitdiv_tensorproduct!(dQ, F♯, allinds, D♯)
    @inbounds for inds in allinds
        for (ij, j) in enumerate(inds)
            for (ii, i) in enumerate(inds)
                dQ.dofs[i] -= D♯[ii, ij] * F♯.dofs[ii, j]
            end
        end
    end
    return nothing
end

function surface_contribution!(
    dQ,
    Q,
    ielem,
    std::AbstractStdRegion,
    disc::MultielementDisc,
    equation::AbstractEquation,
    op::SplitDivOperator,
)
    Fn = disc.cache.Fn
    return if requires_subgrid(op, std)
        _splitdiv_nb_surface_contribution!(dQ, Q, Fn, ielem, std, disc, equation, op)
    else
        _surface_contribution!(dQ, Fn, ielem, std, disc, op)
    end
end

function _splitdiv_nb_surface_contribution!(
    dQ,
    Q,
    Fn,
    ielem,
    std::AbstractStdRegion,
    disc::MultielementDisc,
    equation::AbstractEquation,
    op::SplitDivOperator,
)
    # Non-tensor-product elements are not implemented
    if !is_tensor_product(std)
        throw(ArgumentError(
            "Split-divergence operator not implemented for non-tensor-product elements."
        ))
    end

    # Unpack
    (; mesh, geometry) = disc

    # Indices
    tid = Threads.threadid()
    faces = mesh.elements[ielem].faceinds
    sides = mesh.elements[ielem].facepos

    # Element-local data
    dQe = dQ.elements[ielem]
    Qe = Q.elements[ielem]
    We = std.cache.state[tid][1]
    Fl = std.cache.state[tid][2]
    Fr = std.cache.state[tid][3]

    Ja = geometry.elements[ielem].metric
    frames = geometry.subgrids[ielem].frames
    Js = geometry.subgrids[ielem].jac

    # 1D data
    l, r = std.l
    ∂g = std.∂g

    # Precompute entropy variables
    @inbounds for i in eachindex(We.dofs, Qe.dofs)
        We.dofs[i] = vars_cons2entropy(Qe.dofs[i], equation)
    end

    # Flux differencing
    nd = spatialdim(std)
    if nd >= 1
        inds = tpdofs(std, Val(1))
        sinds = tpdofs_subgrid(std, Val(1))
        Fnl = Fn.faces[faces[1]].sides[sides[1]]
        Fnr = Fn.faces[faces[2]].sides[sides[2]]
        _flux_splitdiv_nb_tensorproduct!(
            Fl, Fr, Fnl, Fnr, Qe, We, Ja, frames, Js, l, r, inds, sinds, equation, op, 1,
        )
        _surf_splitdiv_nb_tensorproduct!(dQe, Fl, Fr, inds, ∂g)
    end
    if nd >= 2
        inds = tpdofs(std, Val(2))
        sinds = tpdofs_subgrid(std, Val(2))
        Fnl = Fn.faces[faces[3]].sides[sides[3]]
        Fnr = Fn.faces[faces[4]].sides[sides[4]]
        _flux_splitdiv_nb_tensorproduct!(
            Fl, Fr, Fnl, Fnr, Qe, We, Ja, frames, Js, l, r, inds, sinds, equation, op, 2,
        )
        _surf_splitdiv_nb_tensorproduct!(dQe, Fl, Fr, inds, ∂g)
    end
    if nd >= 3
        inds = tpdofs(std, Val(3))
        sinds = tpdofs_subgrid(std, Val(3))
        Fnl = Fn.faces[faces[5]].sides[sides[5]]
        Fnr = Fn.faces[faces[6]].sides[sides[6]]
        _flux_splitdiv_nb_tensorproduct!(
            Fl, Fr, Fnl, Fnr, Qe, We, Ja, frames, Js, l, r, inds, sinds, equation, op, 3,
        )
        _surf_splitdiv_nb_tensorproduct!(dQe, Fl, Fr, inds, ∂g)
    end

    return nothing
end

function _flux_splitdiv_nb_tensorproduct!(
    Fl, Fr, Fnl, Fnr, Q, W, Ja, frames, Js, l, r, allinds, subinds, equation, op, dir,
)
    # Compute fluxes by 1D "rows"
    @inbounds for (inds, sinds, Fnli, Fnri) in zip(allinds, subinds, Fnl.dofs, Fnr.dofs)
        # Entropy-projected variables
        i1 = sinds |> first
        i2 = sinds |> last
        nl = frames[dir][i1].n * Js[dir][i1]
        nr = frames[dir][i2].n * Js[dir][i2]
        Wl = l' * view(W.dofs, inds)
        Wr = r' * view(W.dofs, inds)
        Ql = vars_entropy2cons(Wl, equation)
        Qr = vars_entropy2cons(Wr, equation)

        # Contribution from entropy-projected fluxes
        for i in inds
            Fl.dofs[i] = twopointflux(
                Q.dofs[i],
                Ql,
                view(Ja[i], :, dir),
                nl,
                equation,
                op.tpflux,
            )
            Fr.dofs[i] = twopointflux(
                Q.dofs[i],
                Qr,
                view(Ja[i], :, dir),
                nr,
                equation,
                op.tpflux,
            )
        end

        # Subgrid fluxes
        l_Fl = l' * view(Fl.dofs, inds)
        r_Fr = r' * view(Fr.dofs, inds)
        for i in inds
            Fl.dofs[i] -= l_Fl + Fnli
            Fr.dofs[i] -= r_Fr - Fnri
        end
    end

    return nothing
end

function _surf_splitdiv_nb_tensorproduct!(dQ, Fl, Fr, allinds, ∂g)
    ∂gl, ∂gr = ∂g
    @inbounds for inds in allinds
        for (ii, i) in enumerate(inds)
            dQ.dofs[i] += ∂gl[ii] * Fl.dofs[i] - ∂gr[ii] * Fr.dofs[i]
        end
    end
    return nothing
end

#==========================================================================================#
#                               Telescopic divergence operator                             #

"""
    HybridDivOperator([tpflux=numflux.avg, [fvflux=numflux]], numflux, blend)

Split form operator in telescopic form. Only implemented for tensor-product elements.

The telescopic operator approach effectively turns the initial split formulation into a
subelement finite volume scheme. The `twopointflux` and `fvflux` are both combined into
an entropy-stable interface flux that controls the volume dissipation through the
subcell interface Riemann solver.
"""
struct HybridDivOperator{
    T<:AbstractNumericalFlux,
    F<:AbstractNumericalFlux,
    N<:AbstractNumericalFlux,
    RT,
} <: AbstractDivOperator
    tpflux::T
    fvflux::F
    numflux::N
    blend::RT
end

function HybridDivOperator(tpflux, numflux, blend)
    return HybridDivOperator(tpflux, numflux, numflux, blend)
end

function HybridDivOperator(numflux, blend)
    return HybridDivOperator(numflux.avg, numflux, numflux, blend)
end

requires_subgrid(::HybridDivOperator, ::AbstractStdRegion) = true

function volume_contribution!(
    dQ,
    Q,
    ielem,
    std::AbstractStdRegion{ND},
    disc::MultielementDisc,
    equation::AbstractEquation,
    op::HybridDivOperator,
) where {
    ND
}
    return if std |> basis |> hasboundaries
        _hybrid_volume_contribution!(dQ, Q, ielem, std, disc, equation, op)
    else
        # Everything as a surface contribution
        nothing
    end
end

function _hybrid_volume_contribution!(
    dQ,
    Q,
    ielem,
    std::AbstractStdRegion,
    disc::MultielementDisc,
    equation::AbstractEquation,
    op::HybridDivOperator,
)
    # Non-tensor-product elements are not implemented
    if !is_tensor_product(std)
        throw(ArgumentError(
            "Hybrid HO-FV is only implemented for tensor-product elements."
        ))
    end

    # Unpack
    (; geometry) = disc

    # Element-local data
    dQe = dQ.elements[ielem]
    Qe = Q.elements[ielem]
    We = std.cache.state[Threads.threadid()][1]

    Ja = geometry.elements[ielem].metric
    frames = geometry.subgrids[ielem].frames
    Js = geometry.subgrids[ielem].jac

    # 1D data
    nd = spatialdim(std)
    std1d = nd == 1 ? std : (nd == 2 ? std.face : std.edge)
    ω = std1d.ω
    F̄e = std1d.cache.subcell[Threads.threadid()][1]

    # Precompute entropy variables
    @inbounds for i in eachindex(We.dofs, Qe.dofs)
        We.dofs[i] = vars_cons2entropy(Qe.dofs[i], equation)
    end

    if nd >= 1
        inds = tpdofs(std, Val(1))
        sinds = tpdofs_subgrid(std, Val(1))
        _vol_hybrid_tensorproduct!(
            dQe, F̄e, Qe, We, std.D, ω, Ja, frames, Js, inds, sinds, equation, op, 1,
        )
    end
    if nd >= 2
        inds = tpdofs(std, Val(2))
        sinds = tpdofs_subgrid(std, Val(2))
        _vol_hybrid_tensorproduct!(
            dQe, F̄e, Qe, We, std.D, ω, Ja, frames, Js, inds, sinds, equation, op, 2,
        )
    end
    if nd >= 3
        inds = tpdofs(std, Val(3))
        sinds = tpdofs_subgrid(std, Val(3))
        _vol_hybrid_tensorproduct!(
            dQe, F̄e, Qe, We, std.D, ω, Ja, frames, Js, inds, sinds, equation, op, 3,
        )
    end

    return nothing
end

function _vol_hybrid_tensorproduct!(
    dQ, F̄, Q, W, D, ω, Ja, frames, Js, allinds, subinds, equation, op, dir,
)
    @inbounds for (inds, sinds) in zip(allinds, subinds)
        # Zero everything, boundaries will remain zero
        fill!(F̄, zero(datatype(Q)))

        # Fluxes in the subgrid
        npts = length(inds)
        for ii in 2:npts
            # Two-point fluxes
            for ik in ii:npts
                k = inds[ik]
                for il in 1:(ii - 1)
                    l = inds[il]
                    F̄t = twopointflux(
                        Q.dofs[l],
                        Q.dofs[k],
                        view(Ja[l], :, dir),
                        view(Ja[k], :, dir),
                        equation,
                        op.tpflux,
                    )
                    F̄.dofs[ii] += 2ω[il] * D[il, ik] * F̄t
                end
            end

            # Indices
            i = inds[ii]       # Global index
            il = inds[ii - 1]  # Global left node index
            is = sinds[ii]     # Global subgrid index

            # FV fluxes
            frame = frames[dir][is]
            Qln = rotate2face(Q.dofs[il], frame, equation)
            Qrn = rotate2face(Q.dofs[i], frame, equation)
            Fn = numericalflux(Qln, Qrn, frame.n, equation, op.fvflux)
            F̄v = rotate2phys(Fn, frame, equation)
            F̄v *= Js[dir][is]

            # Blending
            Wl = W.dofs[il]
            Wr = W.dofs[i]
            b = dot(Wr - Wl, F̄.dofs[ii] - F̄v)
            δ = _hybrid_compute_delta(b, op.blend)
            F̄.dofs[ii] = (1 - δ) * F̄v + δ * F̄.dofs[ii]
        end

        # Flux differencing
        for (ii, i) in enumerate(inds)
            dQ.dofs[i] += (F̄.dofs[ii] - F̄.dofs[ii + 1]) / ω[ii]
        end
    end

    return nothing
end

function surface_contribution!(
    dQ,
    Q,
    ielem,
    std::AbstractStdRegion{ND},
    disc::MultielementDisc,
    equation::AbstractEquation,
    op::HybridDivOperator,
) where {
    ND
}
    Fn = disc.cache.Fn
    return if std |> basis |> hasboundaries
        _surface_contribution!(dQ, Fn, ielem, std, disc, op)
    else
        _hybrid_nb_surface_contribution!(dQ, Q, Fn, ielem, std, disc, equation, op)
    end
end

function _hybrid_nb_surface_contribution!(
    dQ,
    Q,
    Fn,
    ielem,
    std::AbstractStdRegion,
    disc::MultielementDisc,
    equation::AbstractEquation,
    op::HybridDivOperator,
)
    # Non-tensor-product elements are not implemented
    if !is_tensor_product(std)
        throw(ArgumentError(
            "Hybrid HO-FV is only implemented for tensor-product elements."
        ))
    end

    # Unpack
    (; geometry) = disc

    # Indices
    tid = Threads.threadid()
    faces = disc.mesh.elements[ielem].faceinds
    sides = disc.mesh.elements[ielem].facepos

    # Element-local data
    dQe = dQ.elements[ielem]
    Qe = Q.elements[ielem]
    We = std.cache.state[tid][1]
    Fl = std.cache.state[tid][2]
    Fr = std.cache.state[tid][3]
    F̃e = std.cache.block[tid][1]
    F♯e = std.cache.sharp[tid][1]

    Ja = geometry.elements[ielem].metric
    frames = geometry.subgrids[ielem].frames
    Js = geometry.subgrids[ielem].jac

    # 1D data
    nd = spatialdim(std)
    std1d = nd == 1 ? std : (nd == 2 ? std.face : std.edge)
    ω = std1d.ω
    l, r = std1d.l
    F̄e = std1d.cache.subcell[tid][1]

    # Volume fluxes (diagonal of F♯ matrices)
    _volumeflux!(F̃e, Qe, Ja, equation)

    # Precompute two-point and entropy-projected fluxes
    for i in eachindex(We.dofs, Qe.dofs)
        We.dofs[i] = vars_cons2entropy(Qe.dofs[i], equation)
    end

    # Flux differencing
    if nd >= 1
        inds = tpdofs(std, Val(1))
        sinds = tpdofs_subgrid(std, Val(1))
        Fnl = Fn.faces[faces[1]].sides[sides[1]]
        Fnr = Fn.faces[faces[2]].sides[sides[2]]
        _flux_splitdiv_tensorproduct!(F♯e, F̃e, Qe, Ja, inds, equation, op, 1)
        _flux_splitdiv_nb_tensorproduct!(
            Fl, Fr, Fnl, Fnr, Qe, We, Ja, frames, Js, l, r, inds, sinds, equation, op, 1,
        )
        _surf_hybrid_nb_tensorproduct!(
            dQe, Fl, Fr, Fnl, Fnr, F̄e, F♯e, Qe, We, std.D♯, frames, Js, l, r, ω,
            inds, sinds, equation, op, 1,
        )
    end
    if nd >= 2
        inds = tpdofs(std, Val(2))
        sinds = tpdofs_subgrid(std, Val(2))
        Fnl = Fn.faces[faces[3]].sides[sides[3]]
        Fnr = Fn.faces[faces[4]].sides[sides[4]]
        _flux_splitdiv_tensorproduct!(F♯e, F̃e, Qe, Ja, inds, equation, op, 2)
        _flux_splitdiv_nb_tensorproduct!(
            Fl, Fr, Fnl, Fnr, Qe, We, Ja, frames, Js, l, r, inds, sinds, equation, op, 2,
        )
        _surf_hybrid_nb_tensorproduct!(
            dQe, Fl, Fr, Fnl, Fnr, F̄e, F♯e, Qe, We, std.D♯, frames, Js, l, r, ω,
            inds, sinds, equation, op, 2,
        )
    end
    if nd >= 3
        inds = tpdofs(std, Val(3))
        sinds = tpdofs_subgrid(std, Val(3))
        Fnl = Fn.faces[faces[5]].sides[sides[5]]
        Fnr = Fn.faces[faces[6]].sides[sides[6]]
        _flux_splitdiv_tensorproduct!(F♯e, F̃e, Qe, Ja, inds, equation, op, 3)
        _flux_splitdiv_nb_tensorproduct!(
            Fl, Fr, Fnl, Fnr, Qe, We, Ja, frames, Js, l, r, inds, sinds, equation, op, 3,
        )
        _surf_hybrid_nb_tensorproduct!(
            dQe, Fl, Fr, Fnl, Fnr, F̄e, F♯e, Qe, We, std.D♯, frames, Js, l, r, ω,
            inds, sinds, equation, op, 3,
        )
    end
    return nothing
end

function _surf_hybrid_nb_tensorproduct!(
    dQ, Fl, Fr, Fnl, Fnr, F̄, F♯, Q, W, D♯, frames, Js, l, r, ω,
    allinds, subinds, equation, op, dir,
)
    # Compute fluxes by 1D "rows"
    @inbounds for (inds, sinds, Fnli, Fnri) in zip(allinds, subinds, Fnl.dofs, Fnr.dofs)
        # Subgrid fluxes
        # The "mathematically correct" way to do this is to compute `F̄[end]` also in the
        # loop, but this is a bit more accurate and efficient.
        npts = length(inds)
        F̄.dofs[1] = -Fnli
        F̄.dofs[end] = Fnri
        for ii in 1:(npts - 1)
            # Indices
            i = inds[ii]        # Global index
            ir = inds[ii + 1]   # Global right node index
            is = sinds[ii + 1]  # Global subgrid index

            # Subgrid fluxes
            F̄.dofs[ii + 1] = F̄.dofs[ii] +
                             view(D♯, ii, :)' * view(F♯.dofs, :, i) * ω[ii] -
                             l[ii] * Fl.dofs[i] +
                             r[ii] * Fr.dofs[i]

            # FV fluxes
            frame = frames[dir][is]
            Qln = rotate2face(Q.dofs[i], frame, equation)
            Qrn = rotate2face(Q.dofs[ir], frame, equation)
            Fn = numericalflux(Qln, Qrn, frame.n, equation, op.fvflux)
            F̄v = rotate2phys(Fn, frame, equation)
            F̄v *= Js[dir][is]

            # Blending
            Wl = W.dofs[i]
            Wr = W.dofs[ir]
            b = dot(Wr - Wl, F̄.dofs[ii + 1] - F̄v)
            δ = _hybrid_compute_delta(b, op.blend)
            F̄.dofs[ii + 1] = (1 - δ) * F̄v + δ * F̄.dofs[ii + 1]
        end

        # Flux differencing
        for (ii, i) in enumerate(inds)
            dQ.dofs[i] += (F̄.dofs[ii] - F̄.dofs[ii + 1]) / ω[ii]
        end
    end

    return nothing
end

function _hybrid_compute_delta(b, c)
    # Fisher
    δ = sqrt(b^2 + c)
    δ = (δ - b) / δ

    # Ours
    # δ = if b <= 0
    #     one(b)
    # elseif b >= c
    #     zero(b)
    # else
    #     (1 + cospi(b / c)) / 2
    # end

    # Constant
    # return c

    # Limiting
    return max(δ, 0.5)
    # return δ
end
