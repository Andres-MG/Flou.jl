function FlouBiz.open_for_write!(file::File, disc::AbstractFluxReconstruction)
    # Unpack
    (; mesh) = disc

    #VTKHDF group
    root = create_group(file, "/VTKHDF")
    attributes(root)["Version"] = [1, 0]

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
        push!(offsets, ndofs(std, equispaced=Val(true)) + last(offsets))

        # Types
        push!(types, vtk_type(std))

        # Regions
        push!(regions, region(mesh, ie))
    end

    points = reshape(points, (3, :))
    write(root, "NumberOfPoints", [size(points, 2)])
    write(root, "Points", points)

    write(root, "NumberOfConnectivityIds", [length(connectivities)])
    write(root, "Connectivity", connectivities)

    write(root, "NumberOfCells", [length(types)])
    write(root, "Types", types)
    write(root, "Offsets", offsets)

    write(root, "CellData/Region", regions)
    return nothing
end

function FlouBiz.pointdata2VTKHDF(Q::StateVector, disc::AbstractFluxReconstruction)
    rt = eltype(Q)
    npoints = ndofs(disc)
    nvars = nvariables(Q)

    Qe = [rt[] for _ in 1:nvars]
    for i in 1:nvars
        sizehint!(Qe[i], npoints)
    end

    std = disc.std
    tmp = HybridVector{nvars,rt}(undef, ndofs(std, equispaced=Val(true)))
    for ie in eachelement(disc)
        project2equispaced!(tmp, Q.element[ie], std)
        for iv in 1:nvars
            append!(Qe[iv], view(tmp.flat, iv, :))
        end
    end
    return Qe
end

function FlouBiz.pointdata2VTKHDF(Q::BlockVector, disc::AbstractFluxReconstruction)
    rt = eltype(Q)
    npoints = ndofs(disc)
    dims = spatialdim(Q)
    nvars = nvariables(Q)

    Qe = [rt[] for _ in 1:nvars, _ in 1:dims]
    for i in eachindex(Qe)
        sizehint!(Qe[i], npoints)
    end

    std = disc.std
    tmp = HybridMatrix{nvars,rt}(undef, ndofs(std, equispaced=Val(true)), dims)
    for ie in eachelement(disc)
        project2equispaced!(tmp, Q.element[ie], std)
        for id in 1:dims, iv in 1:nvars
            append!(Qe[iv, id], view(tmp.flat, iv, :, id))
        end
    end
    return Qe |> vec
end
