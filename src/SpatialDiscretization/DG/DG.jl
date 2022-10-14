abstract type DiscontinuousGalerkin{EQ,RT} <: AbstractSpatialDiscretization{EQ,RT} end

Base.eltype(::DiscontinuousGalerkin{EQ,RT}) where {EQ,RT} = RT

function get_std end

function _VTK_type end

function _VTK_connectivities end

function open_for_write!(file::HDF5.File, dg::DiscontinuousGalerkin{EQ,RT}) where {EQ,RT}
    # Unpack
    (; mesh) = dg

    #VTKHDF group
    root = HDF5.create_group(file, "/VTKHDF")
    HDF5.attributes(root)["Version"] = [1, 0]

    points = RT[]
    connectivities = Int[]
    offsets = Int[0]
    types = UInt8[]
    regions = Int[]
    for ie in eachelement(dg)
        std = get_std(dg, ie)
        # Point coordinates
        padding = zeros(SVector{3 - get_spatialdim(mesh),eltype(points)})
        for ξ in std.ξe
            append!(points, phys_coords(ξ, mesh, ie))
            append!(points, padding)
        end

        # Connectivities
        conns = _VTK_connectivities(std) .+ last(offsets)
        append!(connectivities, conns)

        # Offsets
        push!(offsets, ndofs(std, true) + last(offsets))

        # Types
        push!(types, _VTK_type(std))

        # Regions
        push!(regions, get_region(mesh, ie))
    end

    points = reshape(points, (3, :))
    HDF5.write(root, "NumberOfPoints", [size(points, 2)])
    HDF5.write(root, "Points", points)

    HDF5.write(root, "NumberOfConnectivityIds", [length(connectivities)])
    HDF5.write(root, "Connectivity", connectivities)

    HDF5.write(root, "NumberOfCells", [length(types)])
    HDF5.write(root, "Types", types)
    HDF5.write(root, "Offsets", offsets)

    HDF5.write(root, "CellData/Region", regions)
    return nothing
end

function pointdata2VTKHDF(data, dg::DiscontinuousGalerkin)
    npoints = ndofs(dg)
    rt = eltype(data)
    data_ = StateVector(data, dg.dofhandler)

    datavec = rt[]
    sizehint!(datavec, npoints)
    tmp = Vector{rt}(undef, ndofs(get_std(dg, 1), true))
    for ie in eachelement(dg)
        std = get_std(dg, ie)
        resize!(tmp, ndofs(std, true))
        project2equispaced!(tmp, data_[ie], std)
        append!(datavec, tmp)
    end
    return datavec
end

function solution2VTKHDF(Q, dg::DiscontinuousGalerkin)
    npoints = ndofs(dg)
    rt = eltype(Q)
    Q_ = StateVector(Q, dg.dofhandler)

    Qe = [rt[] for _ in eachvariable(dg)]
    for i in eachvariable(dg)
        sizehint!(Qe[i], npoints)
    end
    tmp = Vector{rt}(undef, ndofs(get_std(dg, 1), true))
    for iv in eachvariable(dg), ie in eachelement(dg)
        std = get_std(dg, ie)
        resize!(tmp, ndofs(std, true))
        project2equispaced!(tmp, view(Q_[ie], :, iv), std)
        append!(Qe[iv], tmp)
    end
    return Qe
end

abstract type AbstractNumericalFlux end

struct StdAverageNumericalFlux <: AbstractNumericalFlux end

struct LxFNumericalFlux{T,RT} <: AbstractNumericalFlux
    avg::T
    intensity::RT
end

function numericalflux end

function rotate2face end

function rotate2phys end

function project2faces!(Qf, Q, dg::DiscontinuousGalerkin)
    # Unpack
    (; mesh) = dg

    @flouthreads for ie in eachelement(dg)
        std = get_std(dg, ie)
        iface = mesh.elements[ie].faceinds
        facepos = mesh.elements[ie].facepos
        @inbounds for (s, (face, pos)) in enumerate(zip(iface, facepos))
            mul!(Qf[face][pos], std.l[s], Q[ie])
        end
    end
    return nothing
end

function applyBCs!(Qf, dg::DiscontinuousGalerkin, time)
    (; mesh, geometry, equation, bcs) = dg
    @flouthreads for ibc in eachboundary(mesh)
        for iface in eachbdface(mesh, ibc)
            _dg_applyBC!(
                Qf[iface][2],
                Qf[iface][1],
                geometry.faces[iface].coords,
                geometry.faces[iface].frames,
                time,
                equation,
                bcs[ibc],
            )
        end
    end
    return nothing
end

function interface_fluxes!(Fn, Qf, dg::DiscontinuousGalerkin, riemannsolver)
    (; mesh, geometry, equation) = dg
    @flouthreads for iface in eachface(dg)
        (; eleminds, elempos, orientation) = mesh.faces[iface]
        frame = geometry.faces[iface].frames

        Ql = Qf[iface][1]
        Qr = Qf[iface][2]
        std = get_std(dg, eleminds[1]).faces[elempos[1]]
        @inbounds for i in eachindex(std)
            j = master2slave(i, orientation[], std)
            Qln = rotate2face(view(Ql, i, :), frame[i], equation)
            Qrn = rotate2face(view(Qr, j, :), frame[i], equation)
            Fni = numericalflux(Qln, Qrn, frame[i].n, equation, riemannsolver)
            Fn[iface][1][i, :] = rotate2phys(Fni, frame[i], equation)
            for ivar in eachvariable(equation)
                Fn[iface][1][i, ivar] *= geometry.faces[iface].J[i]
                Fn[iface][2][j, ivar] = -Fn[iface][1][i, ivar]
            end
        end
    end
    return nothing
end

function apply_sourceterm!(dQ, Q, dg::DiscontinuousGalerkin, time)
    (; geometry, source!) = dg
    @flouthreads for i in eachdof(dg)
        x = geometry.elements.coords[i]
        source!(dQ.data[i], Q.data[i], x, time)
    end
    return nothing
end

function apply_massmatrix!(dQ, dg::DiscontinuousGalerkin)
    (; geometry) = dg
    @flouthreads for ie in eachelement(dg)
        ldiv!(geometry.elements[ie].M[], dQ[ie])
    end
    return nothing
end

@inline function _dg_applyBC!(Qext, Qint, coords, frame, time, eq, bc)
    @boundscheck begin
        n = size(Qext, 1)
        size(Qext, 1) == size(Qint, 1) && size(Qext, 2) == size(Qint, 2) ||
            throw(ArgumentError("Qext and Qint must have the same dimensions."))
        length(coords) == n || throw(ArgumentError("coords must have length $(n)."))
        length(frame) == n || throw(ArgumentError("frame must have length $(n)."))
    end
    for (i, Qi) in enumerate(eachrow(Qint))
        Qext[i, :] = stateBC(Qi, coords[i], frame[i], time, eq, bc)
    end
    return nothing
end

function integrate(f::AbstractVector, dg::DiscontinuousGalerkin)
    f_ = StateVector(f, dg.dofhandler)
    return integrate(f_, dg, 1)
end

function integrate(f::StateVector, dg::DiscontinuousGalerkin, ivar::Integer=1)
    (; geometry) = dg
    integral = zero(eltype(f))
    for ie in eachelement(dg)
        integral += integrate(view(f[ie], :, ivar), geometry.elements[ie])
    end
    return integral
end

include("DGSEM.jl")

include("Operators/Operators.jl")
