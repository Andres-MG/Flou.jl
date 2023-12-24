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

struct DofHandler{E,O}
    elem_offsets::O
    face_offsets::O
    nelements::Int
    nfaces::Int
end

function DofHandler(mesh::AbstractMesh, std)
    elem_dofs = ndofs(std)
    face_dofs = ndofs(std.face)
    return DofHandler{true,typeof(elem_dofs)}(
        elem_dofs,
        face_dofs,
        nelements(mesh),
        nfaces(mesh),
    )
end

function DofHandler_withdofs(
    elem_dofs::AbstractVector{<:Integer},
    face_dofs::AbstractVector{<:Integer},
)
    elem_offsets = vcat(0, cumsum(elem_dofs))
    face_offsets = vcat(0, cumsum(face_dofs))
    return DofHandler{false,typeof(elem_offsets)}(
        elem_offsets,
        face_offsets,
        length(elem_dofs),
        length(face_dofs),
    )
end

FlouCommon.nelements(dh::DofHandler) = dh.nelements
FlouCommon.nfaces(dh::DofHandler) = dh.nfaces
ndofs(dh::DofHandler{true}) = dh.nelements * dh.elem_offsets
ndofs(dh::DofHandler{false}) = last(dh.elem_offsets)
ndofs(dh::DofHandler{true}, _) = dh.elem_offsets
ndofs(dh::DofHandler{false}, elem) = dh.elem_offsets[elem + 1] - dh.elem_offsets[elem]
nfacedofs(dh::DofHandler{true}) = dh.nfaces * dh.face_offsets
nfacedofs(dh::DofHandler{false}) = last(dh.face_offsets)
nfacedofs(dh::DofHandler{true}, _) = dh.face_offsets
nfacedofs(dh::DofHandler{false}, face) = dh.face_offsets[face + 1] - dh.face_offsets[face]

@inline function elementoffset(dh::DofHandler{true}, elem)
    @boundscheck 1 <= elem <= nelements(dh) + 1 ||
        throw(ArgumentError(
            "Tried to access element $elem, but only $(nelements(dh) + 1) have offsets."
        ))
    return dh.elem_offsets * (elem - 1)
end
@inline function elementoffset(dh::DofHandler{false}, elem)
    @boundscheck checkbounds(dh.elem_offsets, elem)
    return @inbounds dh.elem_offsets[elem]
end

@inline function dofloc(dh::DofHandler{true}, i)
    @boundscheck 1 <= i <= ndofs(dh) ||
        throw(ArgumentError("Tried to access dof $i, but only $(ndofs(dh)) dofs exist."))
    elem = i ÷ dh.elem_offsets
    iloc = i - dh.elem_offsets * elem
    if iloc == 0
        return elem => dh.elem_offsets
    else
        return elem + 1 => iloc
    end
end
@inline function dofloc(dh::DofHandler{false}, i)
    @boundscheck 1 <= i <= ndofs(dh) ||
        throw(ArgumentError("Tried to access dof $i, but only $(ndofs(dh)) dofs exist."))
    @inbounds begin
        elem = findfirst(>(i), dh.elem_offsets) - 1
        iloc = i - dh.elem_offsets[elem]
    end
    return elem => iloc
end

@inline function faceoffset(dh::DofHandler{true}, face)
    @boundscheck 1 <= face <= nfaces(dh) + 1 ||
        throw(ArgumentError(
            "Tried to access face $face, but only $(nfaces(dh) + 1) have offsets."
        ))
    return dh.face_offsets * (face - 1)
end
@inline function faceoffset(dh::DofHandler{false}, face)
    @boundscheck checkbounds(dh.face_offsets, face)
    return @inbounds dh.face_offsets[face]
end

@inline function facedofloc(dh::DofHandler{true}, i)
    @boundscheck 1 <= i <= nfacedofs(dh) ||
        throw(ArgumentError(
            "Tried to access dof $i, but only $(nfacedofs(dh)) dofs exist."
        ))
    face = i ÷ dh.face_offsets
    iloc = i - dh.face_offsets * face
    if iloc == 0
        return face => dh.face_offsets
    else
        return face + 1 => iloc
    end
end
@inline function facedofloc(dh::DofHandler{false}, i)
    @boundscheck 1 <= i <= nfacedofs(dh) ||
        throw(ArgumentError(
            "Tried to access dof $i, but only $(nfacedofs(dh)) dofs exist."
        ))
    @inbounds begin
        face = findfirst(>(i), dh.face_offsets) - 1
        iloc = i - dh.face_offsets[face]
    end
    return face => iloc
end

FlouCommon.eachelement(dh::DofHandler) = Base.OneTo(nelements(dh))
FlouCommon.eachface(dh::DofHandler) = Base.OneTo(nfaces(dh))
eachdof(dh::DofHandler) = Base.OneTo(ndofs(dh))
eachdof(dh::DofHandler, elem) = Base.OneTo(ndofs(dh, elem))
eachfacedof(dh::DofHandler) = Base.OneTo(nfacedofs(dh))
eachfacedof(dh::DofHandler, face) = Base.OneTo(nfacedofs(dh, face))
