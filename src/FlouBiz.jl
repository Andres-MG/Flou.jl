struct SolutionFile
    name::String
    handler::HDF5.File
end

function open_for_write! end

function open_for_write(filename::String, disc::AbstractSpatialDiscretization)
    fh = HDF5.h5open(filename, "w")
    open_for_write!(fh, disc)
    return SolutionFile(filename, fh)
end

function add_fielddata!(file::SolutionFile, data, name)
    HDF5.write(file.handler, "VTKHDF/FieldData/" * name, data)
    return nothing
end

function add_celldata!(file::SolutionFile, data, name)
    HDF5.write(file.handler, "VTKHDF/CellData/" * name, data)
    return nothing
end

function pointdata2VTKHDF end

function add_pointdata!(
    file::SolutionFile,
    data::AbstractArray,
    disc::AbstractSpatialDiscretization,
    name::String,
)
    datavec = pointdata2VTKHDF(data, disc)
    HDF5.write(file.handler, "VTKHDF/PointData/" * name, datavec)
    return nothing
end

function solution2VTKHDF end

function add_solution!(
    file::SolutionFile,
    sol::AbstractArray,
    disc::AbstractSpatialDiscretization,
    equation::AbstractEquation,
)
    datavec = pointdata2VTKHDF(sol, disc)
    vnames = variablenames(equation)
    for (iv, data) in enumerate(datavec)
        HDF5.write(file.handler, "VTKHDF/PointData/" * vnames[iv], data)
    end
    return nothing
end

function close_file!(file::SolutionFile)
    close(file.handler)
    return nothing
end
