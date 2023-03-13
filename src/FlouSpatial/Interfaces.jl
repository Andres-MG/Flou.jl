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

abstract type AbstractNumericalFlux end

struct StdAverage <: AbstractNumericalFlux end

struct LxF{T,RT} <: AbstractNumericalFlux
    avg::T
    intensity::RT
end

function applyBCs!(Qf, disc::MultielementDisc, equation::AbstractEquation, time)
    (; mesh, geometry, bcs) = disc
    @flouthreads for ibc in eachboundary(mesh)  # TODO: @batch does not work w/ enumerate
        bc = bcs[ibc]
        for iface in eachbdface(mesh, ibc)
            applyBC!(
                Qf.faces[iface].sides[2].dofs,
                Qf.faces[iface].sides[1].dofs,
                geometry.faces[iface].coords,
                geometry.faces[iface].frames,
                time,
                equation,
                bc,
            )
        end
    end
    return nothing
end

function applyBC!(Qext, Qint, coords, frame, time, eq, bc)
    @inbounds for (i, Qi) in enumerate(Qint)
        Qext[i] = bc(Qi, coords[i], frame[i], time, eq)
    end
    return nothing
end

function project2faces!(Qf, Q, disc::MultielementDisc)
    # Unpack
    (; mesh, std) = disc

    @flouthreads for ie in eachelement(disc)
        # Indices
        faces = mesh.elements[ie].faceinds
        sides = mesh.elements[ie].facepos

        # Tensor-product elements
        if is_tensor_product(std)
            nd = spatialdim(std)
            if nd >= 1
                inds = tpdofs(std, Val(1))
                Ql = Qf.faces[faces[1]].sides[sides[1]]
                Qr = Qf.faces[faces[2]].sides[sides[2]]
                _project2faces_tensorproduct!(Ql, Qr, inds, std.l, Q.elements[ie])
            end
            if nd >= 2
                inds = tpdofs(std, Val(2))
                Ql = Qf.faces[faces[3]].sides[sides[3]]
                Qr = Qf.faces[faces[4]].sides[sides[4]]
                _project2faces_tensorproduct!(Ql, Qr, inds, std.l, Q.elements[ie])
            end
            if nd >= 3
                inds = tpdofs(std, Val(3))
                Ql = Qf.faces[faces[5]].sides[sides[5]]
                Qr = Qf.faces[faces[6]].sides[sides[6]]
                _project2faces_tensorproduct!(Ql, Qr, inds, std.l, Q.elements[ie])
            end

        # Generic elements
        else
            for (face, side, lf) in zip(faces, sides, std.l)
                mul!(Qf.faces[face].sides[side].dofs, lf, Q.elements[ie].dofs)
            end
        end
    end

    return nothing
end

function _project2faces_tensorproduct!(Ql, Qr, allinds, lv, Q)
    rt = datatype(Q)
    l, r = lv
    @inbounds for (iface, inds) in enumerate(allinds)
        # Left face
        fill!(Ql.dofsmut[iface], zero(rt))
        for (ii, i) in enumerate(inds)
            Ql.dofs[iface] += l[ii] * Q.dofs[i]
        end
        # Right face
        fill!(Qr.dofsmut[iface], zero(rt))
        for (ii, i) in enumerate(inds)
            Qr.dofs[iface] += r[ii] * Q.dofs[i]
        end
    end
    return nothing
end

function interface_fluxes!(
    Fn::FaceStateVector,
    Qf::FaceStateVector,
    disc::MultielementDisc,
    equation::AbstractEquation,
    riemannsolver::AbstractNumericalFlux,
)
    (; mesh, geometry, std) = disc
    fstd = std.face
    @flouthreads for iface in eachface(disc)
        orientation = mesh.faces[iface].orientation[]
        frames = geometry.faces[iface].frames
        Ql = Qf.faces[iface].sides[1]
        Qr = Qf.faces[iface].sides[2]
        @inbounds for i in eachdof(fstd)
            j = master2slave(i, orientation, fstd)
            Qln = rotate2face(Ql.dofs[i], frames[i], equation)
            Qrn = rotate2face(Qr.dofs[j], frames[i], equation)
            Fni = numericalflux(Qln, Qrn, frames[i].n, equation, riemannsolver)
            Fn.faces[iface].sides[1].dofs[i] = rotate2phys(Fni, frames[i], equation)
            Fn.faces[iface].sides[1].dofs[i] *= geometry.faces[iface].jac[i]
            Fn.faces[iface].sides[2].dofs[j] = -Fn.faces[iface].sides[1].dofs[i]
        end
    end
    return nothing
end

function interface_fluxes!(
    Fn::FaceBlockVector,
    Qf::FaceStateVector,
    disc::MultielementDisc,
    equation::AbstractEquation,
    riemannsolver::AbstractNumericalFlux,
)
    (; mesh, geometry, std) = disc
    fstd = std.face
    @flouthreads for iface in eachface(disc)
        orientation = mesh.faces[iface].orientation[]
        frame = geometry.faces[iface].frames
        Ql = Qf.faces[iface].sides[1]
        Qr = Qf.faces[iface].sides[2]
        @inbounds for i in eachdof(fstd)
            j = master2slave(i, orientation, fstd)
            Qln = rotate2face(Ql.dofs[i], frame[i], equation)
            Qrn = rotate2face(Qr.dofs[j], frame[i], equation)
            Fni = numericalflux(Qln, Qrn, frame[i].n, equation, riemannsolver)
            Fn.faces[iface].sides[1].dofs[i, :] .= rotate2phys(Fni, frame[i], equation)
            for d in eachdim(equation)
                Fn.faces[iface].sides[1].dofs[i, d] *= geometry.faces[iface].jac[i]
                Fn.faces[iface].sides[2].dofs[j, d] = -Fn.face[iface][1][i, d]
            end
        end
    end
    return nothing
end
