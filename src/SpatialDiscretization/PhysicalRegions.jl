#==========================================================================================#
#                                     Reference frame                                      #

struct ReferenceFrame{ND,RT}
    n::SVector{ND,RT}
    t::SVector{ND,RT}
    b::SVector{ND,RT}
end

#==========================================================================================#
#                                 Physical element subgrid                                 #

struct PhysicalSubgrid{F,J}
    frames::F
    J::J
end

struct PhysicalSubgridVector{ND,RT} <: AbstractVector{PhysicalSubgrid}
    dh::DofHandler
    frames::Vector{NTuple{ND,Array{ReferenceFrame{ND,RT}}}}
    J::Vector{NTuple{ND,Array{RT,ND}}}
    function PhysicalSubgridVector(dh, frames, J)
        length(frames) == length(J) && length(frames) == nelements(dh) ||
            throw(DimensionMismatch(
                "There must be as many frames and Jacobians as elements in the DofHandler."
            ))
        nd = length(frames[1])
        rt = eltype(J[1][1])
        return new{nd,rt}(dh, frames, J)
    end
end

Base.length(sv::PhysicalSubgridVector) = length(sv.frames)
Base.size(sv::PhysicalSubgridVector) = (length(sv),)

@inline function Base.getindex(sv::PhysicalSubgridVector, i)
    @boundscheck 1 <= i <= nelements(sv.dh) || throw(BoundsError(sv, i))
    return @inbounds PhysicalSubgrid(
        view(sv.frames, i),
        view(sv.J, i),
    )
end

function PhysicalSubgridVector(stdvec, dh::DofHandler, mesh::AbstractMesh)
    subgridvec = []
    for ie in eachelement(mesh)
        std = stdvec[get_stdid(dh, ie)]
        push!(subgridvec, PhysicalSubgrid(std, mesh, ie))
    end
    return PhysicalSubgridVector(
        dh,
        [s.frames for s in subgridvec],
        [s.J for s in subgridvec],
    )
end

function PhysicalSubgrid(std, ::CartesianMesh{1,RT}, _) where {RT}
    nx = ndofs(std) + 1
    frames = ([
        ReferenceFrame(
            SVector(one(RT)),
            SVector(zero(RT)),
            SVector(zero(RT)),
        )
        for _ in 1:nx
    ],)
    J = (fill(one(RT), nx),)
    return PhysicalSubgrid(frames, J)
end

function PhysicalSubgrid(std, mesh::CartesianMesh{2,RT}, _) where {RT}
    (; Δx) = mesh
    nx, ny = size(std)
    frames = (
        [
            ReferenceFrame(
                SVector(one(RT), zero(RT)),
                SVector(zero(RT), one(RT)),
                SVector(zero(RT), zero(RT)),
            )
            for _ in 1:(nx+1), _ in 1:ny
        ],
        [
            ReferenceFrame(
                SVector(zero(RT), one(RT)),
                SVector(-one(RT), zero(RT)),
                SVector(zero(RT), zero(RT)),
            )
            for _ in 1:nx, _ in 1:(ny+1)
        ],
    )
    J = (
        fill(Δx[2] / 2, nx + 1, ny),
        fill(Δx[1] / 2, nx, ny + 1),
    )
    return PhysicalSubgrid(frames, J)
end

function PhysicalSubgrid(std, mesh::CartesianMesh{3,RT}, _) where {RT}
    (; Δx) = mesh
    nx, ny, nz = size(std)
    frames = (
        [
            ReferenceFrame(
                SVector(one(RT), zero(RT), zero(RT)),
                SVector(zero(RT), one(RT), zero(RT)),
                SVector(zero(RT), zero(RT), one(RT)),
            )
            for _ in 1:(nx+1), _ in 1:ny, _ in 1:nz
        ],
        [
            ReferenceFrame(
                SVector(zero(RT), one(RT), zero(RT)),
                SVector(zero(RT), zero(RT), one(RT)),
                SVector(one(RT), zero(RT), zero(RT)),
            )
            for _ in 1:nx, _ in 1:(ny+1), _ in 1:nz
        ],
        [
            ReferenceFrame(
                SVector(zero(RT), zero(RT), one(RT)),
                SVector(one(RT), zero(RT), zero(RT)),
                SVector(zero(RT), one(RT), zero(RT)),
            )
            for _ in 1:nx, _ in 1:ny, _ in 1:(nz+1)
        ],
    )
    J = (
        fill(Δx[2] * Δx[3] / 4, nx + 1, ny, nz),
        fill(Δx[1] * Δx[3] / 4, nx, ny + 1, nz),
        fill(Δx[1] * Δx[2] / 4, nx, ny, nz + 1),
    )
    return PhysicalSubgrid(frames, J)
end

function PhysicalSubgrid(std, mesh::StepMesh{RT}, ie) where {RT}
    (; Δx) = mesh
    ireg = get_region(mesh, ie)
    nx, ny = size(std)
    frames = (
        [
            ReferenceFrame(
                SVector(one(RT), zero(RT)),
                SVector(zero(RT), one(RT)),
                SVector(zero(RT), zero(RT)),
            )
            for _ in 1:(nx+1), _ in 1:ny
        ],
        [
            ReferenceFrame(
                SVector(zero(RT), one(RT)),
                SVector(-one(RT), zero(RT)),
                SVector(zero(RT), zero(RT)),
            )
            for _ in 1:nx, _ in 1:(ny+1)
        ],
    )
    J = (
        fill(Δx[ireg][2] / 2, nx + 1, ny),
        fill(Δx[ireg][1] / 2, nx, ny + 1),
    )
    return PhysicalSubgrid(frames, J)
end

function PhysicalSubgrid(std, mesh::UnstructuredMesh{1,RT}, ie) where {RT}
    nx = ndofs(std)
    frames = (Vector{ReferenceFrame{1,RT}}(undef, nx + 1),)
    J = (Vector{RT}(undef, nx + 1),)
    for i in 1:(nx+1)
        ξ = std.ξc[1][i]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        s = map_jacobian(main, mesh, ie) |> sign
        frames[1][i] = ReferenceFrame(
            SVector(s * dual[1]),
            SVector(zero(RT)),
            SVector(zero(RT)),
        )
        J[1][i] = one(RT)
    end
    return PhysicalSubgrid(frames, J)
end

function PhysicalSubgrid(std, mesh::UnstructuredMesh{2,RT}, ie) where {RT}
    nx, ny = size(std)
    frames = (
        Matrix{ReferenceFrame{2,RT}}(undef, nx + 1, ny),
        Matrix{ReferenceFrame{2,RT}}(undef, nx, ny + 1),
    )
    J = (Matrix{RT}(undef, nx + 1, ny), Matrix{RT}(undef, nx, ny + 1))
    for j in 1:ny, i in 1:(nx+1)  # Vertical faces
        ξ = std.ξc[1][i, j]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        s = map_jacobian(main, mesh, ie) |> sign
        n = s * dual[1]
        t = s * normalize(main[2])
        b = SVector(zero(RT), zero(RT))
        J[1][i, j] = norm(n)
        n /= J[1][i, j]
        frames[1][i, j] = ReferenceFrame(n, t, b)
    end
    for j in 1:(ny+1), i in 1:nx  # Horizontal faces
        ξ = std.ξc[2][i, j]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        s = map_jacobian(main, mesh, ie) |> sign
        n = s * dual[2]
        t = -s * normalize(main[1])
        b = SVector(zero(RT), zero(RT))
        J[2][i, j] = norm(n)
        n /= J[2][i, j]
        frames[2][i, j] = ReferenceFrame(n, t, b)
    end
    return (frames, J)
end

function PhysicalSubgrid(std, mesh::UnstructuredMesh{3,RT}, ie) where {RT}
    nx, ny, nz = size(std)
    frames = (
        Array{ReferenceFrame{3,RT},3}(undef, nx + 1, ny, nz),
        Array{ReferenceFrame{3,RT},3}(undef, nx, ny + 1, nz),
        Array{ReferenceFrame{3,RT},3}(undef, nx, ny, nz + 1),
    )
    J = (
        Array{RT,3}(undef, nx + 1, ny, nz),
        Array{RT,3}(undef, nx, ny + 1, nz),
        Array{RT,3}(undef, nx, ny, nz + 1),
    )
    for k in 1:nz, j in 1:ny, i in 1:(nx+1)  # X faces
        ξ = std.ξc[1][i, j, k]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        frames = ReferenceFrame(
            SVector(dual[1]),
            SVector(normalize(main[2])),
            SVector(normalize(cross)),
        )
        n = dual[1]
        t = normalize(main[2])
        b = normalize(cross(n, t))
        J[1][i, j, k] = norm(n)
        n /= J[1][i, j, k]
        frames[1][i, j, k] = ReferenceFrame(n, t, b)
    end
    for k in 1:nz, j in 1:(ny+1), i in 1:nx  # Y faces
        ξ = std.ξc[2][i, j, k]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        n = dual[2]
        t = normalize(main[3])
        b = normalize(cross(n, t))
        J[2][i, j, k] = norm(n)
        n /= J[2][i, j, k]
        frames[2][i, j, k] = ReferenceFrame(n, t, b)
    end
    for k in 1:(nz+1), j in 1:ny, i in 1:nx  # Z faces
        ξ = std.ξc[3][i, j, k]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        n = dual[3]
        t = normalize(main[1])
        b = normalize(cross(n, t))
        J[3][i, j, k] = norm(n)
        n /= J[3][i, j, k]
        frames[3][i, j, k] = ReferenceFrame(n, t, b)
    end
    return PhysicalSubgrid(frames, J)
end

#==========================================================================================#
#                                     Physical element                                     #

struct PhysicalElement{C,MM,MF,J,V}
    coords::C
    M::MM
    Mfac::MF
    Ja::J
    volume::V
end

struct PhysicalElementVector{ND,RT,MM,MF,ND2} <: AbstractVector{PhysicalElement}
    dh::DofHandler
    coords::Vector{SVector{ND,RT}}
    M::Vector{MM}
    Mfac::Vector{MF}
    Ja::Vector{SMatrix{ND,ND,RT,ND2}}
    volumes::Vector{RT}
end

Base.length(ev::PhysicalElementVector) = length(ev.volumes)
Base.size(ev::PhysicalElementVector) = (length(ev),)

@inline function Base.getindex(ev::PhysicalElementVector, i::Integer)
    @boundscheck 1 <= i <= nelements(ev.dh) || throw(BoundsError(ev, i))
    @inbounds begin
        i1 = ev.dh.elem_offsets[i] + 1
        i2 = ev.dh.elem_offsets[i+1]
        return PhysicalElement(
            view(ev.coords, i1:i2),
            view(ev.M, i),
            view(ev.Mfac, i),
            view(ev.Ja, i1:i2),
            view(ev.volumes, i),
        )
    end
end

function PhysicalElementVector(
    stdvec,
    dh::DofHandler,
    mesh::AbstractMesh{ND,RT},
) where {
    ND,
    RT,
}
    coords = SVector{ND,RT}[]
    sizehint!(coords, ndofs(dh))
    Mvec = []
    Mfacvec = []
    Javec = []
    volumevec = []

    for ie in eachelement(mesh)
        std = stdvec[get_stdid(dh, ie)]
        elem = PhysicalElement(std, mesh, ie)
        append!(coords, elem.coords)
        push!(Mvec, elem.M)
        push!(Mfacvec, elem.Mfac)
        append!(Javec, elem.Ja)
        push!(volumevec, elem.volume)
    end

    Mvec = [Mvec...]
    Mfacvec = [Mfacvec...]
    Javec = [Javec...]
    volumevec = [volumevec...]

    return PhysicalElementVector(dh, coords, Mvec, Mfacvec, Javec, volumevec)
end

@inline function integrate(f::AbstractVector, elem::PhysicalElement)
    return sum(elem.Mfac[] * f)
end

function PhysicalElement(std, mesh::CartesianMesh{ND,RT}, ie) where {ND,RT}
    (; Δx) = mesh

    # Coordinates
    xyz = [phys_coords(ξ, mesh, ie) for ξ in std.ξ]

    # Jacobian and dual basis
    J = fill(prod(Δx) / (2^ND), ndofs(std))
    Ja = if ND == 1
        fill(SMatrix{1,1}(one(RT)), ndofs(std))
    elseif ND == 2
        fill(
            SMatrix{2,2}(
                Δx[2] / 2, 0,
                0, Δx[1] / 2,
            ),
            ndofs(std),
        )
    else # ND == 3
        fill(
            SMatrix{3,3}(
                Δx[2] * Δx[3] / 4, 0, 0,
                0, Δx[1] * Δx[3] / 4, 0,
                0, 0, Δx[1] * Δx[2] / 4,
            ),
            ndofs(std),
        )
    end

    # Mass matrix
    M = massmatrix(std, J)
    Mfac = M |> factorize

    # Volume
    vol = sum(M)

    return PhysicalElement(xyz, M, Mfac, Ja, vol)
end

function PhysicalElement(std, mesh::StepMesh{RT}, ie) where {RT}
    (; Δx) = mesh
    ireg = get_region(mesh, ie)

    # Coordinates
    xyz = [phys_coords(ξ, mesh, ie) for ξ in std.ξ]

    # Jacobian and dual basis
    J = fill(prod(Δx[ireg]) / 4, ndofs(std))
    Ja = fill(
        SMatrix{2,2}(
            Δx[ireg][2] / 2, 0,
            0, Δx[ireg][1] / 2,
        ),
        ndofs(std),
    )

    # Mass matrix
    M = massmatrix(std, J)
    Mfac = M |> factorize

    # Volume
    vol = sum(M)

    return PhysicalElement(xyz, M, Mfac, Ja, vol)
end

function PhysicalElement(std, mesh::UnstructuredMesh{ND,RT}, ie) where {ND,RT}
    # Coordinates
    xyz = [phys_coords(ξ, mesh, ie) for ξ in std.ξ]

    # Jacobian and dual basis
    J = Vector{RT}(undef, ndofs(std))
    Ja = Vector{SMatrix{ND,ND,RT,ND * ND}}(undef, ndofs(std))
    for i in eachindex(std)
        main = map_basis(std.ξ[i], mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        J[i] = map_jacobian(main, mesh, ie)
        if ND == 1
            Ja[i] = SMatrix{1,1}(dual[1])
            J[i] = abs(J[i])
        elseif ND == 2
            Ja[i] = SMatrix{2,2}(vcat(dual[1], dual[2]))
            J[i] = abs(J[i])
        else # ND == 3
            Ja[i] = SMatrix{3,3}(vcat(dual[1], dual[2], dual[3]))
            J[i] > 0 || throw(DomainError(
                J[i], "Found a negative Jacobian in element $(ie)."
            ))
        end
    end

    # Mass matrix
    M = massmatrix(std, J)
    Mfac = M |> factorize

    # Volume
    vol = sum(M)

    return PhysicalElement(xyz, M, Mfac, Ja, vol)
end

#==========================================================================================#
#                                       Physical face                                      #

struct PhysicalFace{C,F,J,M,S}
    coords::C
    frames::F
    J::J
    M::M
    surface::S
end

struct PhysicalFaceVector{ND,RT,M} <: AbstractVector{PhysicalFace}
    dh::DofHandler
    coords::Vector{SVector{ND,RT}}
    frames::Vector{ReferenceFrame{ND,RT}}
    J::Vector{RT}
    M::Vector{M}
    surfaces::Vector{RT}
end

Base.length(fv::PhysicalFaceVector) = length(fv.surfaces)
Base.size(fv::PhysicalFaceVector) = (length(fv),)

@inline function Base.getindex(fv::PhysicalFaceVector, i::Integer)
    @boundscheck 1 <= i <= nfaces(fv.dh) || throw(BoundsError(fv, i))
    @inline begin
        i1 = fv.dh.face_offsets[i] + 1
        i2 = fv.dh.face_offsets[i+1]
        return PhysicalFace(
            view(fv.coords, i1:i2),
            view(fv.frames, i1:i2),
            view(fv.J, i1:i2),
            view(fv.M, i),
            view(fv.surfaces, i),
        )
    end
end

function PhysicalFaceVector(stdvec, dh, mesh)
    coords = []
    framevec = []
    Jvec = []
    Mvec = []
    surfacevec = []

    for iface in eachface(mesh)
        ielem = mesh.faces[iface].eleminds[1]
        face = PhysicalFace(stdvec[get_stdid(dh, ielem)], mesh, iface)
        append!(coords, face.coords)
        append!(framevec, face.frames)
        append!(Jvec, face.J)
        push!(Mvec, face.M)
        push!(surfacevec, face.surface)
    end

    coords = [coords...]
    framevec = [framevec...]
    Jvec = [Jvec...]
    Mvec = [Mvec...]
    surfacevec = [surfacevec...]

    return PhysicalFaceVector(dh, coords, framevec, Jvec, Mvec, surfacevec)
end

@inline function integrate(f::AbstractVector, face::PhysicalFace)
    return sum(face.M[] * f)
end

function PhysicalFace(_, mesh::CartesianMesh{1,RT}, iface) where {RT}
    pos = mesh.faces[iface].elempos[1]
    nind = mesh.faces[iface].nodeinds[1]
    x = [mesh.nodes[nind]]
    J = [one(RT)]
    if pos == 1
        frames = [
            ReferenceFrame(
                SVector(-one(RT)),
                SVector(zero(RT)),
                SVector(zero(RT)),
            )
        ]
    else # pos == 2
        frames = [
            ReferenceFrame(
                SVector(one(RT)),
                SVector(zero(RT)),
                SVector(zero(RT)),
            )
        ]
    end
    M = SMatrix{1,1}(zero(RT))
    surf = one(RT)
    return PhysicalFace(x, frames, J, M, surf)
end

function PhysicalFace(std, mesh::CartesianMesh{2,RT}, iface) where {RT}
    (; Δx) = mesh
    pos = mesh.faces[iface].elempos[1]
    fstd = std.faces[pos]
    if pos == 1 || pos == 2  # Vertical
        J = fill(Δx[2] / 2, ndofs(fstd))
        if pos == 1
            frames = [
                ReferenceFrame(
                    SVector(-one(RT), zero(RT)),
                    SVector(zero(RT), -one(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        else # pos == 2
            frames = [
                ReferenceFrame(
                    SVector(one(RT), zero(RT)),
                    SVector(zero(RT), one(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        end

    else  # pos == 3 || pos == 4  # Horizontal
        J = fill(Δx[1] / 2, ndofs(fstd))
        if pos == 3
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), -one(RT)),
                    SVector(one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        else # pos == 4
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), one(RT)),
                    SVector(-one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        end
    end

    xy = [face_phys_coords(ξ, mesh, iface) for ξ in fstd.ξ]
    M = massmatrix(fstd, J)
    surf = sum(M)
    return PhysicalFace(xy, frames, J, M, surf)
end

function PhysicalFace(std, mesh::CartesianMesh{3,RT}, iface) where {RT}
    (; Δx) = mesh
    pos = mesh.nodes[iface].elempos[1]
    fstd = std.faces[pos]
    if pos == 1 || pos == 2  # X faces
        J = fill(Δx[2] * Δx[3] / 4, ndofs(fstd))
        if pos == 1
            frames = [
                ReferenceFrame(
                    SVector(-one(RT), zero(RT), zero(RT)),
                    SVector(zero(RT), -one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT), one(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        else # pos == 2
            frames = [
                ReferenceFrame(
                    SVector(one(RT), zero(RT), zero(RT)),
                    SVector(zero(RT), one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT), one(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        end
    elseif pos == 3 || pos == 4  # Y faces
        J = fill(Δx[1] * Δx[3] / 4, ndofs(fstd))
        if pos == 3
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), -one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT), -one(RT)),
                    SVector(one(RT), zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        else # pos == 4
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT), one(RT)),
                    SVector(one(RT), zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        end
    else # pos == 5 || pos == 6  # Z faces
        J = fill(Δx[1] * Δx[2] / 4, ndofs(fstd))
        if pos == 5
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), zero(RT), -one(RT)),
                    SVector(-one(RT), zero(RT), zero(RT)),
                    SVector(zero(RT), one(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        else # pos == 6
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), zero(RT), one(RT)),
                    SVector(one(RT), zero(RT), zero(RT)),
                    SVector(zero(RT), one(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        end
    end

    xy = [face_phys_coords(ξ, mesh, iface) for ξ in fstd.ξ]
    M = massmatrix(fstd, J)
    surf = sum(M)
    return PhysicalFace(xy, frames, J, M, surf)
end

function PhysicalFace(std, mesh::StepMesh{RT}, iface) where {RT}
    (; Δx) = mesh
    ie = mesh.faces[iface].eleminds[1]
    ireg = get_region(mesh, ie)
    pos = mesh.faces[iface].elempos[1]
    fstd = std.faces[pos]

    if pos == 1 || pos == 2  # Vertical
        J = fill(Δx[ireg][2] / 2, ndofs(fstd))
        xy = [face_phys_coords(ξ, mesh, iface) for ξ in fstd.ξ]
        if pos == 1
            frames = [
                ReferenceFrame(
                    SVector(-one(RT), zero(RT)),
                    SVector(zero(RT), -one(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        else # pos == 2
            frames = [
                ReferenceFrame(
                    SVector(one(RT), zero(RT)),
                    SVector(zero(RT), one(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        end

    else  # pos == 3 || pos == 4  # Horizontal
        J = fill(Δx[ireg][1] / 2, ndofs(fstd))
        xy = [face_phys_coords(ξ, mesh, iface) for ξ in fstd.ξ]
        if pos == 3
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), -one(RT)),
                    SVector(one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        else # pos == 4
            frames = [
                ReferenceFrame(
                    SVector(zero(RT), one(RT)),
                    SVector(-one(RT), zero(RT)),
                    SVector(zero(RT), zero(RT)),
                )
                for _ in 1:ndofs(fstd)
            ]
        end
    end

    M = massmatrix(fstd, J)
    surf = sum(M)
    return PhysicalFace(xy, frames, J, M, surf)
end

function PhysicalFace(_, mesh::UnstructuredMesh{1,RT}, iface) where {RT}
    face = mesh.faces[iface]
    ielem = face.eleminds[1]
    pos = face.elempos[1]
    x = Vector{SVector{1,RT}}(undef, 1)
    frames = Vector{ReferenceFrame{1,RT}}(undef, 1)
    J = Vector{RT}(undef, 1)
    if pos == 1
        ξ = SVector(-one(RT))
        x[1] = phys_coords(ξ, mesh, ielem)
        main = map_basis(ξ, mesh, ielem)
        dual = map_dual_basis(main, mesh, ielem)
        s = map_jacobian(main, mesh, ielem) |> sign
        n = -dual[1] * s
        t = SVector(zero(RT))
        b = SVector(zero(RT))
        J[1] = norm(n)
        n /= J[1]
        frames[1] = ReferenceFrame(n, t, b)
    else # pos == 2
        ξ = SVector(one(RT))
        x[1] = phys_coords(ξ, mesh, ielem)
        main = map_basis(ξ, mesh, ielem)
        dual = map_dual_basis(main, mesh, ielem)
        s = map_jacobian(main, mesh, ielem) |> sign
        n = dual[1] * s
        t = SVector(zero(RT))
        b = SVector(zero(RT))
        J[1] = norm(n)
        n /= J[1]
        frames[1] = ReferenceFrame(n, t, b)
    end
    M = SMatrix{1,1}(zero(RT))
    surf = one(RT)
    return PhysicalFace(x, frames, J, M, surf)
end

function PhysicalFace(std, mesh::UnstructuredMesh{2,RT}, iface) where {RT}
    face = mesh.faces[iface]
    ielem = face.eleminds[1]
    pos = face.elempos[1]
    fstd = std.faces[pos]

    xy = Vector{SVector{2,RT}}(undef, ndofs(fstd))
    frames = Vector{ReferenceFrame{2,RT}}(undef, ndofs(fstd))
    J = Vector{RT}(undef, ndofs(fstd))
    if pos == 1  # Left
        for i in eachindex(fstd)
            ξ = SVector(-one(RT), fstd.ξ[i][1])
            xy[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            s = map_jacobian(main, mesh, ielem) |> sign
            n = -dual[1] * s
            t = -normalize(main[2]) * s
            b = SVector(zero(RT), zero(RT))
            J[i] = norm(n)
            n /= J[i]
            frames[i] = ReferenceFrame(n, t, b)
        end
    elseif pos == 2  # Right
        for i in eachindex(fstd)
            ξ = SVector(one(RT), fstd.ξ[i][1])
            xy[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            s = map_jacobian(main, mesh, ielem) |> sign
            n = dual[1] * s
            t = normalize(main[2]) * s
            b = SVector(zero(RT), zero(RT))
            J[i] = norm(n)
            n /= J[i]
            frames[i] = ReferenceFrame(n, t, b)
        end
    elseif pos == 3  # Bottom
        for i in eachindex(fstd)
            ξ = SVector(fstd.ξ[i][1], -one(RT))
            xy[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            s = map_jacobian(main, mesh, ielem) |> sign
            n = -dual[2] * s
            t = normalize(main[1]) * s
            b = SVector(zero(RT), zero(RT))
            J[i] = norm(n)
            n /= J[i]
            frames[i] = ReferenceFrame(n, t, b)
        end
    else # pos == 4  # Top
        for i in eachindex(fstd)
            ξ = SVector(fstd.ξ[i][1], one(RT))
            xy[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            s = map_jacobian(main, mesh, ielem) |> sign
            n = dual[2] * s
            t = -normalize(main[1]) * s
            b = SVector(zero(RT), zero(RT))
            J[i] = norm(n)
            n /= J[i]
            frames[i] = ReferenceFrame(n, t, b)
        end
    end

    M = massmatrix(fstd, J)
    surf = sum(M)
    return PhysicalFace(xy, frames, J, M, surf)
end

function PhysicalFace(std, mesh::UnstructuredMesh{3,RT}, iface) where {RT}
    face = mesh.faces[iface]
    ielem = face.eleminds[1]
    pos = face.elempos[1]
    fstd = std.faces[pos]

    xyz = Vector{SVector{3,RT}}(undef, ndofs(fstd))
    frames = Vector{ReferenceFrame{3,RT}}(undef, ndofs(fstd))
    J = Vector{RT}(undef, ndofs(fstd))
    if pos == 1  # -X face
        for i in eachindex(fstd)
            ξ = SVector(-one(RT), fstd.ξ[i][1], fstd.ξ[i][2])
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n = -dual[1]
            t = -normalize(main[2])
            b = normalize(cross(n, t))
            J[i] = norm(n)
            n /= J[i]
            frames[i] = ReferenceFrame(n, t, b)
        end
    elseif pos == 2  # +X face
        for i in eachindex(fstd)
            ξ = SVector(one(RT), fstd.ξ[i][1], fstd.ξ[i][2])
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n = dual[1]
            t = normalize(main[2])
            b = normalize(cross(n, t))
            J[i] = norm(n)
            n /= J[i]
            frames[i] = ReferenceFrame(n, t, b)
        end
    elseif pos == 3  # -Y face
        for i in eachindex(fstd)
            ξ = SVector(fstd.ξ[i][1], -one(RT), fstd.ξ[i][2])
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n = -dual[2]
            t = -normalize(main[3])
            b = normalize(cross(n, t))
            J[i] = norm(n)
            n /= J[i]
            frames[i] = ReferenceFrame(n, t, b)
        end
    elseif pos == 4  # +Y face
        for i in eachindex(fstd)
            ξ = SVector(fstd.ξ[i][1], one(RT), fstd.ξ[i][2])
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n = dual[2]
            t = normalize(main[3])
            b = normalize(cross(n, t))
            J[i] = norm(n)
            n /= J[i]
            frames[i] = ReferenceFrame(n, t, b)
        end
    elseif pos == 5  # -Z face
        for i in eachindex(fstd)
            ξ = SVector(fstd.ξ[i][1], fstd.ξ[i][2], -one(RT))
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n = -dual[3]
            t = -normalize(main[1])
            b = normalize(cross(n, t))
            J[i] = norm(n)
            n /= J[i]
            frames[i] = ReferenceFrame(n, t, b)
        end
    else # pos == 6  # +Z face
        for i in eachindex(fstd)
            ξ = SVector(fstd.ξ[i][1], fstd.ξ[i][2], one(RT))
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n = dual[3]
            t = normalize(main[1])
            b = normalize(cross(n, t))
            J[i] = norm(n)
            n /= J[i]
            frames[i] = ReferenceFrame(n, t, b)
        end
    end

    M = massmatrix(fstd, J)
    surf = sum(M)
    return PhysicalFace(xyz, frames, J, M, surf)
end

#==========================================================================================#
#                                         Geometry                                         #

struct Geometry{PE,PF,SG}
    elements::PE
    faces::PF
    subgrids::SG
end

function Geometry(stdvec, dh::DofHandler, mesh::AbstractMesh, subgrid=false)
    elemvec = PhysicalElementVector(stdvec, dh, mesh)
    facevec = PhysicalFaceVector(stdvec, dh, mesh)
    subgridvec = if subgrid
        PhysicalSubgridVector(stdvec, dh, mesh)
    else
        nothing
    end
    return Geometry(elemvec, facevec, subgridvec)
end

nelements(v::PhysicalSubgridVector) = nelements(v.dh)
nelements(v::PhysicalElementVector) = nelements(v.dh)
nelements(g::Geometry) = nelements(g.elements)
nfaces(v::PhysicalFaceVector) = nfaces(v.dh)
nfaces(g::Geometry) = nfaces(g.faces)
ndofs(v::PhysicalElementVector) = ndofs(v.dh)
ndofs(g::Geometry) = ndofs(g.elements)

function contravariant(F, Ja::SMatrix{1,1})
    return SVector(F[1] * Ja[1, 1])
end

function contravariant(F, Ja::SMatrix{2,2})
    return SVector(
        F[1] * Ja[1, 1] + F[2] * Ja[2, 1],
        F[1] * Ja[1, 2] + F[2] * Ja[2, 2],
    )
end

function contravariant(F, Ja::SMatrix{3,3})
    return SVector(
        F[1] * Ja[1, 1] + F[2] * Ja[2, 1] + F[3] * Ja[3, 1],
        F[1] * Ja[1, 2] + F[2] * Ja[2, 2] + F[3] * Ja[3, 2],
        F[1] * Ja[1, 3] + F[2] * Ja[2, 3] + F[3] * Ja[3, 3],
    )
end

