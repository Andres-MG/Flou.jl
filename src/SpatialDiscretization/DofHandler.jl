#==========================================================================================#
#                                       DOF handler                                        #

struct DofHandler
    element2std::Vector{Int}
    elem_offsets::Vector{Int}
    face_offsets::Vector{Int}
end

function DofHandler(mesh::AbstractMesh, stdvec, element2std)
    # Element offsets
    elem_dofs = [ndofs(stdvec[i]) for i in element2std]

    # Face offsets
    face_dofs = Vector{Int}(undef, nfaces(mesh))
    for (iface, face) in enumerate(mesh.faces)
        elem1, elem2 = face.eleminds
        if elem2 == 0
            pos = face.elempos[1]
            std = stdvec[element2std[elem1]]
            fstd = std.faces[pos]
            n1 = n2 = ndofs(fstd)
        else
            pos1, pos2 = face.elempos
            std1, std2 = stdvec[element2std[elem1]], stdvec[element2std[elem2]]
            fstd1, fstd2 = std1.faces[pos1], std2.faces[pos2]
            n1, n2 = ndofs(fstd1), ndofs(fstd2)
        end
        if n1 == n2
            face_dofs[iface] = n1
        else
            error("Face $(iface) has a different number of dofs on each side.")
        end
    end
    return DofHandler_withdofs(element2std, elem_dofs, face_dofs)
end

function DofHandler_withdofs(
    element2std::AbstractVector{<:Integer},
    elem_dofs::AbstractVector{<:Integer},
    face_dofs::AbstractVector{<:Integer},
)
    elem_offsets = vcat(0, cumsum(elem_dofs))
    face_offsets = vcat(0, cumsum(face_dofs))
    return DofHandler(element2std, elem_offsets, face_offsets)
end

nelements(dh::DofHandler) = length(dh.element2std)
nfaces(dh::DofHandler) = length(dh.face_offsets) - 1
ndofs(dh::DofHandler) = last(dh.elem_offsets)
nfacedofs(dh::DofHandler) = 2 * last(dh.face_offsets)

@inline function get_stdid(dh::DofHandler, elem)
    @boundscheck checkbounds(dh.element2std, elem)
    return @inbounds dh.element2std[elem]
end

@inline function get_dofid(dh::DofHandler, elem, i)
    @boundscheck checkbounds(dh.elem_offsets, elem)
    return @inbounds dh.elem_offsets[elem] + i
end

function get_dofid(dh::DofHandler, i)
    @boundscheck 1 >= i >= ndofs(dh) ||
        throw(ArgumentError("Tried to access dof $i, but only $(ndofs(dh)) dofs exist."))
    @inbounds begin
        elem = findfirst(>(i), dh.elem_offsets) - 1
        iloc = i - dh.elem_offsets[elem]
    end
    return elem => iloc
end

eachelement(dh::DofHandler) = Base.OneTo(nelements(dh))
eachface(dh::DofHandler) = Base.OneTo(nfaces(dh))
eachdof(dh::DofHandler) = Base.OneTo(ndofs(dh))

