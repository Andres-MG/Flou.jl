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

abstract type AbstractStdRegion{ND,NT} end

"""
    nodetype(std)

Return the node distribution type of `std`.
"""
function nodetype(s::AbstractStdRegion)
    return s.nodetype
end

"""
    dofsize(std; equispaced=Val(false))

Return the number of solution nodes in each spatial direction (for non-tensor-product
regions simply return the total number of nodes).

See also [`ndofs`](@ref), [`eachdof`](@ref).
"""
function dofsize(
    s::AbstractStdRegion{ND};
    equispaced::Val{E}=Val(false),
) where {
    ND,
    E,
}
    return if E
        ntuple(i -> equisize(s, i), ND)
    else
        ntuple(_ -> nnodes(s |> nodetype), ND)
    end
end

"""
    dofsize(std, i; equispaced=Val(false))

Return the number of solution nodes in the `i` direction (equivalent to `dofsize(std)` with
non-tensor-product regions).
"""
function dofsize(s::AbstractStdRegion, i::Integer; equispaced=Val(false))
    return dofsize(s; equispaced)[i]
end

"""
    lineardofs(std; equispaced=Val(false))

Return a `LinearIndices` object useful to convert indices from tensor to linear form.

See also [`cartesiandofs`](@ref).
"""
function lineardofs(s::AbstractStdRegion; equispaced=Val(false))
    return LinearIndices(dofsize(s; equispaced))
end

"""
    cartesiandofs(std; equispaced=Val(false))

Return a `CartesianIndices` object useful to convert indices from linear to tensor form.

See also [`lineardofs`](@ref).
"""
function cartesiandofs(s::AbstractStdRegion; equispaced=Val(false))
    return CartesianIndices(dofsize(s; equispaced))
end

"""
    ndofs(std; equispaced=Val(false))

Return the number of solution nodes.

See also [`eachdof`](@ref), [`dofsize`](@ref).
"""
function ndofs(s::AbstractStdRegion; equispaced=Val(false))
    return dofsize(s; equispaced) |> prod
end

"""
    ndofs(std, i; equispaced=Val(false))

Return the number of solution nodes in the `i` direction (equivalent to `ndofs(std)` in
non-tensor-product elements).
"""
function ndofs(s::AbstractStdRegion, i::Integer; equispaced=Val(false))
    return dofsize(s, i; equispaced)
end

"""
    eachdof(std; equispaced=Val(false))

Return a range that covers all the indices of the solution nodes.

See also [`ndofs`](@ref), [`dofsize`](@ref).
"""
function eachdof(s::AbstractStdRegion; equispaced=Val(false))
    return Base.OneTo(ndofs(s; equispaced))
end

"""
    eachdof(std, i; equispaced=Val(false))

Return a range that covers all the indices of the solution nodes in the `i` direction
(equivalent to `eachdof(std)` in non-tensor-product regions).
"""
function eachdof(s::AbstractStdRegion, i::Integer; equispaced=Val(false))
    return Base.OneTo(ndofs(s, i; equispaced))
end

"""
    spatialdim(std)

Return the spatial dimension of `std`.
"""
function FlouCommon.spatialdim(::AbstractStdRegion{ND}) where {ND}
    return ND
end

"""
    ndirections(std)

Return the number of directions of `std`. In tensor-product regions it is equivalent to
`spatialdim(std)`.

See also [`eachdirection`](@ref).
"""
function ndirections(s::AbstractStdRegion)
    return spatialdim(s)
end

"""
    eachdirection(std)

Return a range that covers all the directions of `std`.

See also [`ndirections`](@ref).
"""
function eachdirection(s::AbstractStdRegion)
    return Base.OneTo(ndirections(s))
end

"""
    is_tensor_product(std)

Return `true` if `std` is a tensor-product region.
"""
function is_tensor_product(::AbstractStdRegion)
    return true
end

"""
    project2equispaced!(Qe, Q, std)

Project the nodal solution values `Q` in region `std` to equispaced nodes, `Qe`.
"""
function project2equispaced!(
    Qe::AbstractVecOrMat,
    Q::AbstractVecOrMat,
    s::AbstractStdRegion,
)
    return mul!(Qe, s.node2eq, Q)
end

"""
    nequispaced(std)

Return the number of equispaced nodes, useful for plotting and/or saving to files.
"""
function nequispaced(s::AbstractStdRegion)
    return length(s.ξe)
end

"""
    equisize(std, i)

Return the number of equispaced nodes in direction `i`, useful for plotting and/or saving
to files.
"""
function equisize(s::AbstractStdRegion{ND}, _) where {ND}
    n = nequispaced(s)
    return ND == 1 ? n : (ND == 2 ? Int(sqrt(n)) : Int(cbrt(n)))
end

"""
    equisize(std)

Return the number of equispaced nodes in each spatial direction, useful for plotting
and/or saving to files (for non-tensor-product regions simply return the total number of
nodes).
"""
function equisize(s::AbstractStdRegion{ND}) where {ND}
    return ntuple(i -> equisize(s, i), ND)
end

#==========================================================================================#
#                                      Standard point                                      #

abstract type AbstractStdPoint <: AbstractStdRegion{0,GaussNodes} end

ndirections(::AbstractStdPoint) = 1
nvertices(::AbstractStdPoint) = 1

function slave2master(i::Integer, _, ::AbstractStdPoint)
    return i
end

function master2slave(i::Integer, _, ::AbstractStdPoint)
    return i
end

#==========================================================================================#
#                                     Standard segment                                     #

abstract type AbstractStdSegment{NT} <: AbstractStdRegion{1,NT} end

ndirections(::AbstractStdSegment) = 1
nvertices(::AbstractStdSegment) = 2

function slave2master(i::Integer, orientation, std::AbstractStdSegment)
    return if orientation == 0
        i
    else # orientation == 1
        ndofs(std) - (i - 1)
    end
end

function master2slave(i::Integer, orientation, std::AbstractStdSegment)
    return if orientation == 0
        i
    else # orientation == 1
        ndofs(std) - (i - 1)
    end
end

vtk_type(::AbstractStdSegment) = UInt8(68)

function vtk_connectivities(s::AbstractStdSegment)
    conns = [1, ndofs(s)]
    append!(conns, 2:(ndofs(s) - 1))
    return conns .- 1
end

#==========================================================================================#
#                                  Standard quadrilateral                                  #

abstract type AbstractStdQuad{NT} <: AbstractStdRegion{2,NT} end

ndirections(::AbstractStdQuad) = 2
nvertices(::AbstractStdQuad) = 4

function slave2master(i::Integer, orientation, std::AbstractStdQuad)
    s = cartesiandofs(std)[i]
    li = lineardofs(std)
    return if orientation == 0
        i
    elseif orientation == 1
        li[ndofs(std, 1) - s[2] + 1, s[1]]
    elseif orientation == 2
        li[ndofs(std, 1) - s[1] + 1, ndofs(std, 2) - s[2] + 1]
    elseif orientation == 3
        li[s[2], ndofs(std, 2) - s[1] + 1]
    elseif orientation == 4
        li[s[2], s[1]]
    elseif orientation == 5
        li[ndofs(std, 1) - s[1] + 1, s[2]]
    elseif orientation == 6
        li[ndofs(std, 1) - s[2] + 1, ndofs(std, 2) - s[1] + 1]
    else # orientation == 7
        li[s[1], ndofs(std, 2) - s[2] + 1]
    end
end

function master2slave(i::Integer, orientation, std::AbstractStdQuad)
    m = cartesiandofs(std)[i]
    li = lineardofs(std)
    return if orientation == 0
        i
    elseif orientation == 1
        li[m[2], ndofs(std, 1) - m[1] + 1]
    elseif orientation == 2
        li[ndofs(std, 1) - m[1] + 1, ndofs(std, 2) - m[2] + 1]
    elseif orientation == 3
        li[ndofs(std, 2) - m[2] + 1, m[1]]
    elseif orientation == 4
        li[m[2], m[1]]
    elseif orientation == 5
        li[ndofs(std, 1) - m[1] + 1, m[2]]
    elseif orientation == 6
        li[ndofs(std, 2) - m[2] + 1, ndofs(std, 1) - m[1] + 1]
    else # orientation == 7
        li[m[1], ndofs(std, 2) - m[2] + 1]
    end
end

vtk_type(::AbstractStdQuad) = UInt8(70)

function vtk_connectivities(s::AbstractStdQuad)
    n = ndofs(s, 1)
    li = lineardofs(s)
    corners = [li[1, 1], li[n, 1], li[n, n], li[1, n]]
    edges = reduce(vcat, [
        li[2:(n - 1), 1], li[n, 2:(n - 1)],
        li[2:(n - 1), n], li[1, 2:(n - 1)],
    ])
    interior = vec(li[2:(n - 1), 2:(n - 1)])
    return mapreduce(x -> x .- 1, vcat, (corners, edges, interior))
end

#==========================================================================================#
#                                    Standard hexahedron                                   #

abstract type AbstractStdHex{NT} <: AbstractStdRegion{3,NT} end

ndirections(::AbstractStdHex) = 3
nvertices(::AbstractStdHex) = 8

function slave2master(_, _, _::AbstractStdHex)
    error("Not implemented yet!")
end

function master2slave(_, _, _::AbstractStdHex)
    error("Not implemented yet!")
end

vtk_type(::AbstractStdHex) = UInt8(72)

function vtk_connectivities(s::AbstractStdHex)
    n = ndofs(s, 1)
    li = lineardofs(s)
    corners = [
        li[1, 1, 1], li[n, 1, 1], li[n, n, 1], li[1, n, 1],
        li[1, 1, n], li[n, 1, n], li[n, n, n], li[1, n, n],
    ]
    edges = reduce(vcat, [
        li[2:(n - 1), 1, 1], li[n, 2:(n - 1), 1],
        li[2:(n - 1), n, 1], li[1, 2:(n - 1), 1],
        li[2:(n - 1), 1, n], li[n, 2:(n - 1), n],
        li[2:(n - 1), n, n], li[1, 2:(n - 1), n],
        li[1, 1, 2:(n - 1)], li[n, 1, 2:(n - 1)],
        li[n, n, 2:(n - 1)], li[1, n, 2:(n - 1)],
    ])
    faces = reduce(vcat, [
        li[1, 2:(n - 1), 2:(n - 1)], li[n, 2:(n - 1), 2:(n - 1)],
        li[2:(n - 1), 1, 2:(n - 1)], li[2:(n - 1), n, 2:(n - 1)],
        li[2:(n - 1), 2:(n - 1), 1], li[2:(n - 1), 2:(n - 1), n],
    ] .|> vec)
    interior = vec(li[2:(end - 1), 2:(end - 1), 2:(end - 1)])
    return mapreduce(x -> x .- 1, vcat, (corners, edges, faces, interior))
end
