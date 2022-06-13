abstract type DiscontinuousGalerkin{EQ,RT} <: AbstractSpatialDiscretization{EQ,RT} end

include("DofHandler.jl")

function StateVector(raw, dh::DofHandlerDG, stdvec, nvars)
    nd = [ndofs.(stdvec)...]
    nelems = [nelements(dh, i) for i in eachregion(dh)]
    return StateVector(raw, nd, nelems, nvars)
end

function StateVector{RT}(
    value::Union{UndefInitializer,Missing,Nothing},
    dh::DofHandlerDG,
    stdvec,
    nvars,
) where {
    RT,
}
    nd = [ndofs.(stdvec)...]
    nelems = [nelements(dh, i) for i in eachregion(dh)]
    rawlen = sum(nd .* nvars .* nelems)
    raw = Vector{RT}(value, rawlen)
    return StateVector(raw, nd, nelems, nvars)
end

function MortarStateVector{RT}(value, mesh, stdvec, dh::DofHandlerDG, nvars) where {RT}
    dims = Vector{NTuple{2,NTuple{2,Int}}}(undef, nfaces(mesh))
    for i in eachface(mesh)
        elem = face(mesh, i).eleminds[1]
        reg, _ = loc2reg(dh, elem)
        std = stdvec[reg]
        pos = face(mesh, i).elempos[1]
        idir = (pos - 1) ÷ 2 + 1
        ndof = ndofs(std) ÷ size(std, idir)
        dims[i] = (
            (ndof, nvars),
            (ndof, nvars),
        )
    end
    MortarStateVector{RT}(value, dims)
end

function save2csv(filename, Q, dg)
    open(filename, "w") do fh
        # Header
        dim = spatialdim(dg.mesh)
        if dim == 1
            print(fh, "x,")
        elseif dim == 2
            print(fh, "x,y,")
        else # dim == 3
            print(fh, "x,y,z,")
        end
        join(fh, variablenames(dg.equation), ",")
        println(fh)

        # Print format
        f = "%.7e" * ",%.7e" ^ (dim - 1 + nvariables(dg.equation)) |> Printf.Format

        # Data
        dh = dg.dofhandler
        for ireg in eachregion(dh), ieloc in eachelement(dh, ireg)
            ie = reg2loc(dh, ireg, ieloc)
            X = coords(dg.physelem, ie)
            for i in eachindex(X)
                Printf.format(fh, f, X[i]..., Q[ireg][i, :, ie]...)
                println(fh)
            end
        end
    end
    return nothing
end

function _VTK_type end

function _VTK_connectivities end

function save2vtkhdf(filename, Q, dg)
    # Write to VTK HDF file (only one partition)
    HDF5.h5open(filename, "w") do fh
        # Unpack
        (; mesh, stdvec, dofhandler, equation) = dg

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

abstract type AbstractNumericalFlux end

struct StdAverageNumericalFlux <: AbstractNumericalFlux end

struct LxFNumericalFlux{T,RT} <: AbstractNumericalFlux
    avg::T
    intensity::RT
end

function numericalflux! end

function rotate2face! end

function rotate2phys! end

# TODO: maybe move to StdRegions.jl??
function project2faces!(Qf, Q, dg::DiscontinuousGalerkin)
    # Unpack
    (; mesh, dofhandler, stdvec, equation) = dg

    ndim = spatialdim(mesh)
    for ireg in eachregion(dofhandler)
        std = stdvec[ireg]
        @flouthreads for ieloc in eachelement(dofhandler, ireg)
            ie = reg2loc(dofhandler, ireg, ieloc)
            iface = element(mesh, ie).faceinds
            facepos = element(mesh, ie).facepos

            if is_tensor_product(std)
                Qr = reshape(
                    view(Q[ireg], :, :, ieloc),
                    (size(std)..., size(Q[ireg], 2)),
                )

                # 1D
                if ndim == 1
                    @inbounds for v in eachvariable(equation)
                        Qf[iface[1]][facepos[1]][1, v] = zero(eltype(Qf))
                        Qf[iface[2]][facepos[2]][1, v] = zero(eltype(Qf))
                        for k in eachindex(std)
                            Qf[iface[1]][facepos[1]][1, v] += std.l[1][k] * Qr[k, v]
                            Qf[iface[2]][facepos[2]][1, v] += std.l[2][k] * Qr[k, v]
                        end
                    end

                # 2D
                elseif ndim == 2
                    @inbounds for v in eachvariable(equation)
                        for j in eachindex(std, 2)
                            Qf[iface[1]][facepos[1]][j, v] = zero(eltype(Qf))
                            Qf[iface[2]][facepos[2]][j, v] = zero(eltype(Qf))
                            for k in eachindex(std, 1)
                                Qf[iface[1]][facepos[1]][j, v] += std.l[1][k] * Qr[k, j, v]
                                Qf[iface[2]][facepos[2]][j, v] += std.l[2][k] * Qr[k, j, v]
                            end
                        end
                        for i in eachindex(std, 1)
                            Qf[iface[3]][facepos[3]][i, v] = zero(eltype(Qf))
                            Qf[iface[4]][facepos[4]][i, v] = zero(eltype(Qf))
                            for k in eachindex(std, 2)
                                Qf[iface[3]][facepos[3]][i, v] += std.l[3][k] * Qr[i, k, v]
                                Qf[iface[4]][facepos[4]][i, v] += std.l[4][k] * Qr[i, k, v]
                            end
                        end
                    end

                # 3D
                else # ndim == 3
                    error("Not implemented yet!")
                end
            else
                @inbounds for v in eachvariable(equation)
                    for i in eachindex(std), s in eachindex(iface, facepos)
                        Qf[iface[s]][facepos[s]][i, v] = zero(eltype(Qf))
                        for k in eachindex(std)
                            Qf[iface[s]][facepos[s]][i, v] = std.l[2][k] *
                                                             Q[ireg][k, v, ieloc]
                        end
                    end
                end
            end
        end
    end
    return nothing
end

function applyBCs!(Qf, dg::DiscontinuousGalerkin, time)
    (; mesh, physface, equation, bcs) = dg
    @flouthreads for ibc in eachboundary(mesh)
        for iface in eachbdface(mesh, ibc)
            _dg_applyBC!(
                Qf[iface][2],
                Qf[iface][1],
                coords(physface, iface),
                face(physface, iface).n,
                face(physface, iface).t,
                face(physface, iface).b,
                time,
                equation,
                bcs[ibc],
            )
        end
    end
    return nothing
end

function interface_fluxes!(Fn, Qf, dg::DiscontinuousGalerkin, riemannsolver)
    (; mesh, dofhandler, stdvec, physface, equation) = dg
    @flouthreads for iface in eachface(mesh)
        (; eleminds, elempos, orientation) = face(mesh, iface)
        n = face(physface, iface).n
        t = face(physface, iface).t
        b = face(physface, iface).b

        Qln = MVector{nvariables(equation),eltype(Qf)}(undef)
        Qrn = MVector{nvariables(equation),eltype(Qf)}(undef)
        Fni = MVector{nvariables(equation),eltype(Fn)}(undef)
        Ql = Qf[iface][1]
        Qr = Qf[iface][2]
        ireg = loc2reg(dofhandler, eleminds[1]).first
        std = face(stdvec[ireg], elempos[1])
        @inbounds for i in eachindex(std)
            j = slave2master(i, orientation, std)
            rotate2face!(Qln, view(Ql, i, :), n[i], t[i], b[i], equation)
            rotate2face!(Qrn, view(Qr, j, :), n[i], t[i], b[i], equation)
            numericalflux!(Fni, Qln, Qrn, n[i], equation, riemannsolver)
            rotate2phys!(view(Fn[iface][1], i, :), Fni, n[i], t[i], b[i], equation)
            for ivar in eachvariable(equation)
                Fn[iface][1][i, ivar] *= face(physface, iface).J[i]
                Fn[iface][2][j, ivar] = -Fn[iface][1][i, ivar]
            end
        end
    end
    return nothing
end

function apply_massmatrix!(dQ, dg::DiscontinuousGalerkin)
    (; dofhandler, physelem) = dg
    @flouthreads for ie in eachelement(dofhandler)
        ireg, ieloc = loc2reg(dofhandler, ie)
        ldiv!(
            element(physelem, ie).M,
            view(dQ[ireg], :, :, ieloc),
        )
    end
    return nothing
end

function apply_sourceterm!(dQ, Q, dg::DiscontinuousGalerkin, time)
    (; dofhandler, physelem, source!) = dg
    for ie in eachelement(dofhandler)
        ireg, ieloc = loc2reg(dofhandler, ie)
        x = coords(physelem, ie)
        source!(view(dQ[ireg], :, :, ieloc), view(Q[ireg], :, :, ieloc), x, time, ireg)
    end
    return nothing
end

@inline function _dg_applyBC!(Qext, Qint, coords, n, t, b, time, eq, bc)
    @boundscheck begin
        size(Qext, 1) == size(Qint, 1) && size(Qext, 2) == size(Qint, 2) ||
            throw(ArgumentError("Qext and Qint must have the same dimensions."))
    end
    for (i, Qi) in enumerate(eachrow(Qint))
        copy!(view(Qext, i, :), Qi)
        stateBC!(view(Qext, i, :), coords[i], n[i], t[i], b[i], time, eq, bc)
    end
    return nothing
end


include("DGSEM.jl")
