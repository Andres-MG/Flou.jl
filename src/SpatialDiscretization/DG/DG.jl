abstract type DiscontinuousGalerkin{RT} <: AbstractSpatialDiscretization{RT} end

datatype(::DiscontinuousGalerkin{RT}) where {RT} = RT

abstract type DGcache{RT} <: AbstractSpatialDiscretizationCache{RT} end

function _VTK_type end

function _VTK_connectivities end

function open_for_write!(file::HDF5.File, dg::DiscontinuousGalerkin)
    # Unpack
    (; mesh) = dg

    #VTKHDF group
    root = HDF5.create_group(file, "/VTKHDF")
    HDF5.attributes(root)["Version"] = [1, 0]

    std = dg.std
    points = datatype(dg)[]
    connectivities = Int[]
    offsets = Int[0]
    types = UInt8[]
    regions = Int[]
    for ie in eachelement(dg)
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
        push!(offsets, ndofs(std, equispaced=true) + last(offsets))

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

function pointdata2VTKHDF(Q::StateVector{nvars}, dg::DiscontinuousGalerkin) where {nvars}
    rt = eltype(Q)
    npoints = ndofs(dg)

    Qe = [rt[] for _ in 1:nvars]
    for i in 1:nvars
        sizehint!(Qe[i], npoints)
    end

    std = dg.std
    tmp = HybridVector{nvars,rt}(undef, ndofs(std, equispaced=true))
    for ie in eachelement(dg)
        project2equispaced!(tmp, Q.element[ie], std)
        for iv in 1:nvars
            append!(Qe[iv], view(tmp.flat, iv, :))
        end
    end
    return Qe
end

function pointdata2VTKHDF(Q::BlockVector{nvars}, dg::DiscontinuousGalerkin) where {nvars}
    rt = eltype(Q)
    npoints = ndofs(dg)
    dims = get_spatialdim(Q)

    Qe = [rt[] for _ in 1:nvars, _ in 1:dims]
    for i in eachindex(Qe)
        sizehint!(Qe[i], npoints)
    end

    std = dg.std
    tmp = HybridMatrix{nvars,rt}(undef, ndofs(std, equispaced=true), dims)
    for ie in eachelement(dg)
        project2equispaced!(tmp, Q.element[ie], std)
        for id in 1:dims, iv in 1:nvars
            append!(Qe[iv, id], view(tmp.flat, iv, :, id))
        end
    end
    return Qe |> vec
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
    (; mesh, std) = dg

    @flouthreads for ie in eachelement(dg)
        iface = mesh.elements[ie].faceinds
        facepos = mesh.elements[ie].facepos
        @inbounds for (s, (face, pos)) in enumerate(zip(iface, facepos))
            mul!(Qf.face[face][pos], std.l[s], Q.element[ie])
        end
    end
    return nothing
end

function applyBCs!(Qf, dg::DiscontinuousGalerkin, equation::AbstractEquation, time)
    (; mesh, geometry, bcs) = dg
    @flouthreads for ibc in eachboundary(mesh)  # TODO: @batch does not work w/ enumerate
        bc = bcs[ibc]
        for iface in eachbdface(mesh, ibc)
            _dg_applyBC!(
                Qf.face[iface][2],
                Qf.face[iface][1],
                geometry.faces[iface].coords,
                geometry.faces[iface].frames,
                time,
                equation,
                bc,
            )
        end
    end
    return nothing
end

function interface_fluxes!(
    Fn::FaceStateVector,
    Qf::FaceStateVector,
    dg::DiscontinuousGalerkin,
    equation::AbstractEquation,
    riemannsolver,
)
    (; mesh, geometry, std) = dg
    fstd = std.face
    @flouthreads for iface in eachface(dg)
        orientation = mesh.faces[iface].orientation[]
        frame = geometry.faces[iface].frames
        Ql = Qf.face[iface][1]
        Qr = Qf.face[iface][2]
        @inbounds for i in eachindex(fstd)
            j = master2slave(i, orientation, fstd)
            Qln = rotate2face(Ql[i], frame[i], equation)
            Qrn = rotate2face(Qr[j], frame[i], equation)
            Fni = numericalflux(Qln, Qrn, frame[i].n, equation, riemannsolver)
            Fn.face[iface][1][i] = rotate2phys(Fni, frame[i], equation)
            Fn.face[iface][1][i] *= geometry.faces[iface].J[i]
            Fn.face[iface][2][j] = -Fn.face[iface][1][i]
        end
    end
    return nothing
end

function interface_fluxes!(
    Fn::FaceBlockVector,
    Qf::FaceStateVector,
    dg::DiscontinuousGalerkin,
    equation::AbstractEquation,
    riemannsolver,
)
    (; mesh, geometry, std) = dg
    fstd = std.face
    @flouthreads for iface in eachface(dg)
        orientation = mesh.faces[iface].orientation[]
        frame = geometry.faces[iface].frames
        Ql = Qf.face[iface][1]
        Qr = Qf.face[iface][2]
        @inbounds for i in eachindex(fstd)
            j = master2slave(i, orientation, fstd)
            Qln = rotate2face(Ql[i], frame[i], equation)
            Qrn = rotate2face(Qr[j], frame[i], equation)
            Fni = numericalflux(Qln, Qrn, frame[i].n, equation, riemannsolver)
            Fn.face[iface][1][i, :] .= rotate2phys(Fni, frame[i], equation)
            for d in eachdim(equation)
                Fn.face[iface][1][i, d] *= geometry.faces[iface].J[i]
                Fn.face[iface][2][j, d] = -Fn.face[iface][1][i, d]
            end
        end
    end
    return nothing
end

function apply_sourceterm!(dQ, Q, dg::DiscontinuousGalerkin, time)
    (; geometry, source!) = dg
    @flouthreads for i in eachdof(dg)
        x = geometry.elements.coords[i]
        source!(dQ.dof[i], Q.dof[i], x, time)
    end
    return nothing
end

function apply_massmatrix!(dQ, dg::DiscontinuousGalerkin)
    (; geometry) = dg
    @flouthreads for ie in eachelement(dg)
        ldiv!(geometry.elements.M[ie], dQ.element[ie])
    end
    return nothing
end

@inline function _dg_applyBC!(Qext, Qint, coords, frame, time, eq, bc)
    @boundscheck begin
        n = length(Qext)
        nv = length(first(Qext))
        length(Qint) == n && innerdim(Qint) == nv ||
            throw(ArgumentError("Qext and Qint must have the same dimensions."))
        length(coords) == n || throw(ArgumentError("coords must have length $(n)."))
        length(frame) == n || throw(ArgumentError("frame must have length $(n)."))
    end
    @inbounds for (i, Qi) in enumerate(Qint)
        Qext[i] = bc(Qi, coords[i], frame[i], time, eq)
    end
    return nothing
end

function integrate(f::AbstractVector, dg::DiscontinuousGalerkin)
    f_ = StateVector{1}(f, dg.dofhandler)
    return integrate(f_, dg, 1)
end

function integrate(f::StateVector, dg::DiscontinuousGalerkin, ivar::Integer)
    (; geometry) = dg
    integral = zero(eltype(f))
    @flouthreads for ie in eachelement(dg)
        integral += integrate(view(f.element[ie].flat, ivar, :), geometry.elements[ie])
    end
    return integral
end

function integrate(f::StateVector, dg::DiscontinuousGalerkin)
    (; geometry) = dg
    integral = zero(eltype(f))
    @flouthreads for ie in eachelement(dg)
        integral += integrate(f.element[ie], geometry.elements[ie])
    end
    return integral
end

include("DGSEM.jl")

include("Operators/Operators.jl")
