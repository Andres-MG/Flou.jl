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
nfacedofs(dh::DofHandler) = 2 * last(dh.face_offsets)

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

FlouCommon.eachelement(dh::DofHandler) = Base.OneTo(nelements(dh))
FlouCommon.eachface(dh::DofHandler) = Base.OneTo(nfaces(dh))
eachdof(dh::DofHandler) = Base.OneTo(ndofs(dh))

