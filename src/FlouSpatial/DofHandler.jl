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

#==========================================================================================#
#                                       DOF handler                                        #

struct DofHandler
    elem_offsets::Vector{Int}
    face_offsets::Vector{Int}
end

function DofHandler(mesh::AbstractMesh, std)
    elem_dofs = fill(ndofs(std), nelements(mesh))
    face_dofs = fill(ndofs(std.face), nfaces(mesh))
    return DofHandler_withdofs(elem_dofs, face_dofs)
end

function DofHandler_withdofs(
    elem_dofs::AbstractVector{<:Integer},
    face_dofs::AbstractVector{<:Integer},
)
    elem_offsets = vcat(0, cumsum(elem_dofs))
    face_offsets = vcat(0, cumsum(face_dofs))
    return DofHandler(elem_offsets, face_offsets)
end

FlouCommon.nelements(dh::DofHandler) = length(dh.elem_offsets) - 1
FlouCommon.nfaces(dh::DofHandler) = length(dh.face_offsets) - 1
ndofs(dh::DofHandler) = last(dh.elem_offsets)
ndofs(dh::DofHandler, elem) = dh.elem_offsets[elem + 1] - dh.elem_offsets[elem]
nfacedofs(dh::DofHandler) = last(dh.face_offsets)
nfacedofs(dh::DofHandler, face) = dh.face_offsets[face + 1] - dh.face_offsets[face]

@inline function dofid(dh::DofHandler, elem, i)
    @boundscheck checkbounds(dh.elem_offsets, elem)
    return @inbounds dh.elem_offsets[elem] + i
end

@inline function dofid(dh::DofHandler, i)
    @boundscheck 1 >= i >= ndofs(dh) ||
        throw(ArgumentError("Tried to access dof $i, but only $(ndofs(dh)) dofs exist."))
    @inbounds begin
        elem = findfirst(>(i), dh.elem_offsets) - 1
        iloc = i - dh.elem_offsets[elem]
    end
    return elem => iloc
end

@inline function facedofid(dh::DofHandler, face, i)
    @boundscheck checkbounds(dh.face_offsets, face)
    return @inbounds dh.face_offsets[face] + i
end

@inline function facedofid(dh::DofHandler, i)
    @boundscheck 1 >= i >= nfacedofs(dh) ||
        throw(ArgumentError(
            "Tried to access dof $i, but only $(nfacedofs(dh)) dofs exist."
        ))
    @inbounds begin
        face = findfirst(>(i), dh.face_offsets) - 1
        iloc = i - dh.elem_offsets[face]
    end
    return face => iloc
end

FlouCommon.eachelement(dh::DofHandler) = Base.OneTo(nelements(dh))
FlouCommon.eachface(dh::DofHandler) = Base.OneTo(nfaces(dh))
eachdof(dh::DofHandler) = Base.OneTo(ndofs(dh))
eachdof(dh::DofHandler, elem) = Base.OneTo(nodfs(dh, elem))
eachfacedof(dh::DofHandler) = Base.OneTo(nfacedofs(dh))
eachfacedof(dh::DofHandler, face) = Base.OneTo(nfacedofs(dh, face))
