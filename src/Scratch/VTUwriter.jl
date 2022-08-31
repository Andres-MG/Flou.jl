function _save2vtu(filename, Q, disc)
    # Unpack
    (; mesh, stdvec, dofhandler, equation) = disc
    endianness = Base.ENDIAN_BOM == 0x04030201 ? "LittleEndian" : "BigEndian"

    # Root node
    headertype = UInt
    root = ElementNode("VTKFile")
    root["type"] = "UnstructuredGrid"
    root["version"] = 2.2
    root["byte_order"] = endianness
    root["header_type"] = headertype
    grid = addelement!(root, "UnstructuredGrid")

    points = eltype(Q)[]
    solution = [eltype(Q)[] for _ in eachvariable(equation)]
    connectivities = UInt[]
    offsets = [UInt(0)]
    types = UInt8[]
    regions = UInt8[]
    for ir in eachregion(dofhandler)
        std = stdvec[ir]
        Qt = similar(Q, ndofs(std))
        for ieloc in eachelement(dofhandler, ir)
            # Point coordinates
            ie = reg2loc(dofhandler, ir, ieloc)
            padding = zeros(SVector{3 - spatialdim(mesh),eltype(points)})
            for ξ in std.ξe
                append!(points, phys_coords(ξ, mesh, ie))
                append!(points, padding)
            end

            # Point values
            for iv in eachvariable(equation)
                project2equispaced!(Qt, view(Q[ir], :, iv, ieloc), std)
                append!(solution[iv], Qt)
            end

            # Connectivities
            conns = _VTK_connectivities(std) .+ last(offsets)
            append!(connectivities, conns)

            # Offsets
            push!(offsets, UInt(ndofs(std) + last(offsets)))

            # Types
            push!(types, _VTK_type(std))

            # Regions
            push!(regions, UInt8(ir))
        end
    end
    popfirst!(offsets)

    # Piece node (only one partition)
    buffer = IOBuffer()
    piece = addelement!(grid, "Piece")

    # Points section
    piece["NumberOfPoints"] = length(points) ÷ 3
    pointnode = addelement!(piece, "Points")

    # Points
    data = addelement!(pointnode, "DataArray")
    data["type"] = eltype(points)
    data["Name"] = "Points"
    data["NumberOfComponents"] = 3
    data["format"] = "appended"
    data["offset"] = position(buffer)

    write(buffer, headertype(sizeof(points)))
    write(buffer, points)

    # Cells section
    piece["NumberOfCells"] = length(offsets)
    cellnode = addelement!(piece, "Cells")

    # Connectivities
    data = addelement!(cellnode, "DataArray")
    data["type"] = eltype(connectivities)
    data["Name"] = "connectivity"
    data["format"] = "appended"
    data["offset"] = position(buffer)

    write(buffer, headertype(sizeof(connectivities)))
    write(buffer, connectivities)

    # Offsets
    data = addelement!(cellnode, "DataArray")
    data["type"] = eltype(offsets)
    data["Name"] = "offsets"
    data["format"] = "appended"
    data["offset"] = position(buffer)

    write(buffer, headertype(sizeof(offsets)))
    write(buffer, offsets)

    # Types
    data = addelement!(cellnode, "DataArray")
    data["type"] = eltype(types)
    data["Name"] = "types"
    data["format"] = "appended"
    data["offset"] = position(buffer)

    write(buffer, headertype(sizeof(types)))
    write(buffer, types)

    # Point data
    pointdata = addelement!(piece, "PointData")
    for iv in eachvariable(equation)
        vname = variablenames(equation)[iv]
        data = addelement!(pointdata, "DataArray")
        data["type"] = eltype(solution[iv])
        data["Name"] = vname
        data["format"] = "appended"
        data["offset"] = position(buffer)

        write(buffer, headertype(sizeof(solution[iv])))
        write(buffer, solution[iv])
    end

    # Cell data
    celldata = addelement!(piece, "CellData")
    data = addelement!(celldata, "DataArray")
    data["type"] = eltype(regions)
    data["Name"] = "region"
    data["format"] = "appended"
    data["offset"] = position(buffer)

    write(buffer, headertype(sizeof(regions)))
    write(buffer, regions)

    # Write data to file
    filebuff = IOBuffer()
    prettyprint(filebuff, root)
    filestr = String(take!(filebuff))
    parts = rsplit(filestr, "\n", limit=2, keepempty=true)
    open(filename, "w") do fh
        println(fh, parts[1])
        println(fh, "  <AppendedData encoding=\"raw\">")
        write(fh, "    _", take!(buffer), "\n")
        println(fh, "  </AppendedData>")
        println(fh, parts[2])
    end
    return nothing
end
