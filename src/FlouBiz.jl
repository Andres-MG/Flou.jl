function save(filename, Q::StateVector, disc)
    ext = split(filename, ".")[end]
    if ext == "csv"
        _save2csv(filename, Q, disc)
    elseif ext == "hdf"
        _save2vtkhdf(filename, Q, disc)
    end
    return nothing
end

function _save2csv(filename, Q, disc)
    open(filename, "w") do fh
        # Header
        dim = spatialdim(disc.mesh)
        if dim == 1
            print(fh, "x,")
        elseif dim == 2
            print(fh, "x,y,")
        else # dim == 3
            print(fh, "x,y,z,")
        end
        join(fh, variablenames(disc.equation), ",")
        println(fh)

        # Print format
        f = "%.7e" * ",%.7e" ^ (dim - 1 + nvariables(disc.equation)) |> Printf.Format

        # Data
        dh = disc.dofhandler
        for ireg in eachregion(dh), ieloc in eachelement(dh, ireg)
            ie = reg2loc(dh, ireg, ieloc)
            X = coords(disc.physelem, ie)
            for i in eachindex(X)
                Printf.format(fh, f, X[i]..., Q[ireg][i, :, ie]...)
                println(fh)
            end
        end
    end
    return nothing
end

function _save2vtkhdf(filename, Q, disc)
    # Write to VTK HDF file (only one partition)
    HDF5.h5open(filename, "w") do fh
        # Unpack
        (; mesh, stdvec, dofhandler, equation) = disc

        #VTKHDF group
        root = HDF5.create_group(fh, "/VTKHDF")
        HDF5.attributes(root)["Version"] = [1, 0]

        points = eltype(Q)[]
        solution = [eltype(Q)[] for _ in eachvariable(equation)]
        connectivities = Int[]
        offsets = Int[0]
        types = UInt8[]
        regions = Int[]
        for ir in eachregion(dofhandler)
            std = stdvec[ir]
            Qt = similar(Q, ndofs(std))
            for ieloc in eachelement(dofhandler, ir)
                # Point coordinates
                ie = reg2loc(dofhandler, ir, ieloc)
                padding = zeros(SVector{3 - spatialdim(mesh),eltype(points)})
                for ξ in std.ξe
                    append!(points, coords(ξ, mesh, ie))
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
                push!(offsets, ndofs(std) + last(offsets))

                # Types
                push!(types, _VTK_type(std))

                # Regions
                push!(regions, ir)
            end
        end

        points = reshape(points, (3, :))
        HDF5.write(root, "NumberOfPoints", [size(points, 2)])
        HDF5.write(root, "Points", points)

        HDF5.write(root, "NumberOfConnectivityIds", [length(connectivities)])
        HDF5.write(root, "Connectivity", connectivities)

        HDF5.write(root, "NumberOfCells", [length(types)])
        HDF5.write(root, "Types", types)
        HDF5.write(root, "Offsets", offsets)

        for iv in eachvariable(equation)
            vname = variablenames(equation)[iv]
            HDF5.write(root, "PointData/$(vname)", solution[iv])
        end

        HDF5.write(root, "CellData/region", regions)
    end
end

_VTK_type(::StdSegment) = UInt8(68)
_VTK_type(::StdQuad) = UInt8(70)
_VTK_type(::StdTri) = UInt8(69)

function _VTK_connectivities(s::StdSegment)
    conns = [1, length(s)]
    append!(conns, 2:(length(s) - 1))
    return conns .- 1
end

function _VTK_connectivities(s::StdQuad)
    nx, ny = size(s)
    li = LinearIndices(s)
    conns = [
        li[1, 1], li[nx, 1], li[nx, ny], li[1, ny],
        li[2:(end - 1), 1]..., li[nx, 2:(end - 1)]...,
        li[2:(end - 1), ny]..., li[1, 2:(end - 1)]...,
        li[2:(end - 1), 2:(end - 1)]...,
    ]
    return conns .- 1
end

function _VTK_connectivities(::StdTri)
    error("Not implemented yet!")
end
