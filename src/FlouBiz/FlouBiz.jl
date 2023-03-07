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

module FlouBiz

using ..FlouCommon
using HDF5: HDF5, File, h5open, write

export FlouFile
export open_for_write, add_fielddata!, add_celldata!, add_pointdata!, add_solution!
export close_file!

struct FlouFile
    name::String
    handler::File
end

function open_for_write!(::File, AbstractSpatialDiscretization) end

"""
    open_for_write(filename, disc)

Create (or rewrite) a VTKHDF file with name `filename`, returning a handler that can be
used to add data.
"""
function open_for_write(filename::String, disc::AbstractSpatialDiscretization)
    fh = h5open(filename, "w")
    open_for_write!(fh, disc)
    return FlouFile(filename, fh)
end

"""
    add_fielddata!(file, data, name)

Add field information (metadata) with `name` to `file`.
"""
function add_fielddata!(file::FlouFile, data, name)
    write(file.handler, "VTKHDF/FieldData/" * name, data)
    return nothing
end

"""
    add_celldata!(file, data, name)

Add cell values to `file`. `length(data)` must equal the number of elements in the
discretization since cell values are constant across the nodes of each element.
"""
function add_celldata!(file::FlouFile, data, name)
    write(file.handler, "VTKHDF/CellData/" * name, data)
    return nothing
end

function pointdata2VTKHDF(::AbstractArray, ::AbstractSpatialDiscretization) end

"""
    add_pointdata!(file, data, disc, name)

Add point values to `file`. There must be a value per solution node in the discretization.
"""
function add_pointdata!(
    file::FlouFile,
    data::AbstractArray,
    disc::AbstractSpatialDiscretization,
    name::String,
)
    datavec = pointdata2VTKHDF(data, disc)
    write(file.handler, "VTKHDF/PointData/" * name, datavec)
    return nothing
end

function solution2VTKHDF end

"""
    add_solution!(file, sol, dis, equation)

Add the point values of the different variables of `equation` to `file`. It is simply a
wrapper around [`add_pointdata!`](@ref).
"""
function add_solution!(
    file::FlouFile,
    sol::AbstractArray,
    disc::AbstractSpatialDiscretization,
    equation::AbstractEquation,
)
    datavec = pointdata2VTKHDF(sol, disc)
    vnames = variablenames(equation)
    for (iv, data) in enumerate(datavec)
        write(file.handler, "VTKHDF/PointData/" * vnames[iv], data)
    end
    return nothing
end

"""
    close_file!(file)

Release the resources used by HDF5 to handle the VTKHDF `file`.
"""
function close_file!(file::FlouFile)
    close(file.handler)
    return nothing
end

end # FlouBiz
