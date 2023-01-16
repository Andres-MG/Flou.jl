abstract type AbstractNumericalFlux end

struct StdAverageNumericalFlux <: AbstractNumericalFlux end

struct LxFNumericalFlux{T,RT} <: AbstractNumericalFlux
    avg::T
    intensity::RT
end

function applyBCs!(Qf, disc::HighOrderElements, equation::AbstractEquation, time)
    (; mesh, geometry, bcs) = disc
    @flouthreads for ibc in eachboundary(mesh)  # TODO: @batch does not work w/ enumerate
        bc = bcs[ibc]
        for iface in eachbdface(mesh, ibc)
            applyBC!(
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

function applyBC!(Qext, Qint, coords, frame, time, eq, bc)
    @inbounds for (i, Qi) in enumerate(Qint)
        Qext[i] = bc(Qi, coords[i], frame[i], time, eq)
    end
    return nothing
end

function interface_fluxes!(
    Fn::FaceStateVector,
    Qf::FaceStateVector,
    disc::HighOrderElements,
    equation::AbstractEquation,
    riemannsolver,
)
    (; mesh, geometry, std) = disc
    fstd = std.face
    @flouthreads for iface in eachface(disc)
        orientation = mesh.faces[iface].orientation[]
        frame = geometry.faces[iface].frames
        Ql = Qf.face[iface][1]
        Qr = Qf.face[iface][2]
        @inbounds for i in eachdof(fstd)
            j = master2slave(i, orientation, fstd)
            Qln = rotate2face(Ql[i], frame[i], equation)
            Qrn = rotate2face(Qr[j], frame[i], equation)
            Fni = numericalflux(Qln, Qrn, frame[i].n, equation, riemannsolver)
            Fn.face[iface][1][i] = rotate2phys(Fni, frame[i], equation)
            Fn.face[iface][1][i] *= geometry.faces[iface].jac[i]
            Fn.face[iface][2][j] = -Fn.face[iface][1][i]
        end
    end
    return nothing
end

function interface_fluxes!(
    Fn::FaceBlockVector,
    Qf::FaceStateVector,
    disc::HighOrderElements,
    equation::AbstractEquation,
    riemannsolver,
)
    (; mesh, geometry, std) = disc
    fstd = std.face
    @flouthreads for iface in eachface(disc)
        orientation = mesh.faces[iface].orientation[]
        frame = geometry.faces[iface].frames
        Ql = Qf.face[iface][1]
        Qr = Qf.face[iface][2]
        @inbounds for i in eachdof(fstd)
            j = master2slave(i, orientation, fstd)
            Qln = rotate2face(Ql[i], frame[i], equation)
            Qrn = rotate2face(Qr[j], frame[i], equation)
            Fni = numericalflux(Qln, Qrn, frame[i].n, equation, riemannsolver)
            Fn.face[iface][1][i, :] .= rotate2phys(Fni, frame[i], equation)
            for d in eachdim(equation)
                Fn.face[iface][1][i, d] *= geometry.faces[iface].jac[i]
                Fn.face[iface][2][j, d] = -Fn.face[iface][1][i, d]
            end
        end
    end
    return nothing
end

function apply_sourceterm!(dQ, Q, disc::HighOrderElements, time)
    (; geometry, source!) = disc
    @flouthreads for i in eachdof(disc)
        x = geometry.elements.coords[i]
        source!(dQ.dof[i], Q.dof[i], x, time)
    end
    return nothing
end

function integrate(f::AbstractVector, disc::HighOrderElements)
    f_ = StateVector{1}(f, disc.dofhandler)
    return integrate(f_, disc, 1)
end

function integrate(f::StateVector, disc::HighOrderElements, ivar::Integer)
    (; geometry) = disc
    integral = zero(eltype(f))
    @flouthreads for ie in eachelement(disc)
        integral += integrate(view(f.element[ie].flat, ivar, :), geometry.elements[ie])
    end
    return integral
end

function integrate(f::StateVector, disc::HighOrderElements)
    (; geometry) = disc
    integral = zero(eltype(f))
    @flouthreads for ie in eachelement(disc)
        integral += integrate(f.element[ie], geometry.elements[ie])
    end
    return integral
end
