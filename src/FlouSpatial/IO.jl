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

function FlouBiz.open_for_write!(file::HDF5.File, disc::MultielementDisc)
    # Unpack
    (; mesh) = disc

    #VTKHDF group
    root = HDF5.create_group(file, "/VTKHDF")

    # Version
    HDF5.write_attribute(root, "Version", [1, 0])

    # File type (TODO: using ASCII strings must be easier...)
    filetype = "UnstructuredGrid"
    space = HDF5.dataspace(filetype)
    type = HDF5.datatype(filetype)
    HDF5.h5t_set_cset(type.id, HDF5.H5T_CSET_ASCII)
    att = HDF5.create_attribute(root, "Type", type, space)
    HDF5.h5a_write(att.id, type.id, pointer(filetype))

    # Mesh data
    std = disc.std
    points = datatype(disc)[]
    connectivities = Int[]
    offsets = Int[0]
    types = UInt8[]
    regions = Int[]
    for ie in eachelement(disc)
        # Point coordinates
        padding = zeros(SVector{3 - spatialdim(mesh),eltype(points)})
        for ξ in std.ξe
            append!(points, phys_coords(ξ, mesh, ie))
            append!(points, padding)
        end

        # Connectivities
        conns = vtk_connectivities(std) .+ last(offsets)
        append!(connectivities, conns)

        # Offsets
        push!(offsets, nequispaced(std) + last(offsets))

        # Types
        push!(types, vtk_type(std))

        # Regions
        push!(regions, region(mesh, ie))
    end

    points = reshape(points, (3, :))
    root["NumberOfPoints"] = [size(points, 2)]
    root["Points"] = points

    root["NumberOfConnectivityIds"] = [length(connectivities)]
    root["Connectivity"] = connectivities

    root["NumberOfCells"] = [length(types)]
    root["Types"] = types
    root["Offsets"] = offsets

    root["CellData/Region"] = regions
    return nothing
end

function FlouBiz.pointdata2VTKHDF(Q::GlobalStateVector, disc::MultielementDisc)
    (; std) = disc
    rt = datatype(Q)
    npoints = ndofs(disc)
    nvars = nvariables(Q)

    Qe = [rt[] for _ in 1:nvars]
    sizehint!.(Qe, npoints)

    tmp = std.cache.state[1][1]
    for ie in eachelement(disc)
        for iv in 1:nvars
            project2equispaced!(tmp.vars[iv], Q.elements[ie].vars[iv], std)
            append!(Qe[iv], tmp.vars[iv])
        end
    end
    return Qe
end

function FlouBiz.pointdata2VTKHDF(_Q::AbstractVecOrMat, disc::MultielementDisc)
    Q = GlobalStateVector(_Q, disc.dofhandler)
    return FlouBiz.pointdata2VTKHDF(Q, disc)
end

function FlouBiz.pointdata2VTKHDF(_Q::GlobalBlockVector, disc::MultielementDisc)
    (; dofhandler, std) = disc
    Q = GlobalBlockVector(_Q, dofhandler)
    rt = datatype(Q)
    npoints = ndofs(disc)
    dims = spatialdim(Q)
    nvars = nvariables(Q)

    Qe = [rt[] for _ in 1:nvars, _ in 1:dims]
    sizehint!.(Qe, npoints)

    tmp = std.cache.block[1][1]
    for ie in eachelement(disc)
        for id in 1:dims, iv in 1:nvars
            project2equispaced!(tmp.vars[iv, id], Q.elements[ie].vars[iv, id], std)
            append!(Qe[iv, id], tmp.vars[iv, id])
        end
    end
    return Qe |> vec
end

function FlouBiz.pointdata2VTKHDF(_Q::AbstractArray{<:Any,3}, disc::MultielementDisc)
    Q = GlobalBlockVector(_Q, disc.dofhandler)
    return FlouBiz.pointdata2VTKHDF(Q, disc)
end
