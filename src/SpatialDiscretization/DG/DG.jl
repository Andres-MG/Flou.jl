abstract type DiscontinuousGalerkin{EQ,RT} <: AbstractSpatialDiscretization{EQ,RT} end

include("DofHandler.jl")

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
function project2faces!(Qf, Q, dg::DiscontinuousGalerkin, eq)
    # Unpack
    (; mesh, dofhandler, stdvec) = dg

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
                    @inbounds for v in eachvariable(eq)
                        Qf[iface[1]][facepos[1]][1, v] = zero(eltype(Qf))
                        Qf[iface[2]][facepos[2]][1, v] = zero(eltype(Qf))
                        for k in eachindex(std)
                            Qf[iface[1]][facepos[1]][1, v] += std.l[1][k] * Qr[k, v]
                            Qf[iface[2]][facepos[2]][1, v] += std.l[2][k] * Qr[k, v]
                        end
                    end

                # 2D
                elseif ndim == 2
                    @inbounds for v in eachvariable(eq)
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
                @inbounds for v in eachvariable(eq)
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

function applyBCs!(Qf, dg::DiscontinuousGalerkin, time, eq)
    (; mesh, physface, bcs) = dg
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
                eq,
                bcs[ibc],
            )
        end
    end
    return nothing
end

function interface_fluxes!(Fn, Qf, dg::DiscontinuousGalerkin, eq, riemannsolver)
    (; mesh, dofhandler, stdvec, physface) = dg
    @flouthreads for iface in eachface(mesh)
        (; eleminds, elempos, orientation) = face(mesh, iface)
        n = face(physface, iface).n
        t = face(physface, iface).t
        b = face(physface, iface).b

        Qln = MVector{nvariables(eq),eltype(Qf)}(undef)
        Qrn = MVector{nvariables(eq),eltype(Qf)}(undef)
        Fni = MVector{nvariables(eq),eltype(Fn)}(undef)
        Ql = Qf[iface][1]
        Qr = Qf[iface][2]
        ireg = loc2reg(dofhandler, eleminds[1]).first
        std = face(stdvec[ireg], elempos[1])
        @inbounds for i in eachindex(std)
            j = slave2master(i, orientation, std)
            rotate2face!(Qln, view(Ql, i, :), n[i], t[i], b[i], eq)
            rotate2face!(Qrn, view(Qr, j, :), n[i], t[i], b[i], eq)
            numericalflux!(Fni, Qln, Qrn, n[i], eq, riemannsolver)
            rotate2phys!(view(Fn[iface][1], i, :), Fni, n[i], t[i], b[i], eq)
            for ivar in eachvariable(eq)
                Fn[iface][1][i, ivar] *= face(physface, iface).J[i]
                Fn[iface][2][j, ivar] = -Fn[iface][1][i, ivar]
            end
        end
    end
    return nothing
end

function apply_massmatrix!(dQ, dg::DiscontinuousGalerkin)
    (; dofhahdler, physelem) = dg
    @flouthreads for ie in eachelement(dofhahdler)
        ireg, ieloc = loc2reg(dofhahdler, ie)
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

function _dg_applyBC!(Qext, Qint, coords, n, t, b, time, eq, bc)
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
