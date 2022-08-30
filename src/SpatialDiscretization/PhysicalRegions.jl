struct PhysicalElement{ND,RT,MM,MF,ND2,SG}
    coords::Vector{SVector{ND,RT}}
    M::MM
    Mfac::MF
    Ja::Vector{SMatrix{ND,ND,RT,ND2}}
    volume::RT
    subgrid::SG
end

function PhysicalElement(std, mesh::CartesianMesh{ND,RT}, ie, sub=false) where {ND,RT}
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

    # Subgrid
    subgrid = if sub
        PhysicalSubgrid(std, mesh, ie)
    else
        nothing
    end

    return PhysicalElement(xyz, M, Mfac, Ja, vol, subgrid)
end

function PhysicalElement(std, mesh::StepMesh{RT}, ie, sub=false) where {RT}
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

    # Subgrid
    subgrid = if sub
        PhysicalSubgrid(std, mesh, ie)
    else
        nothing
    end

    return PhysicalElement(xyz, M, Mfac, Ja, vol, subgrid)
end

function PhysicalElement(std, mesh::UnstructuredMesh{ND,RT}, ie, sub=false) where {ND,RT}
    # Coordinates
    xyz = [phys_coords(ξ, mesh, ie) for ξ in std.ξ]

    # Jacobian and dual basis
    J = Vector{RT}(undef, ndofs(std))
    Ja = Vector{SMatrix{ND,ND,RT,ND*ND}}(undef, ndofs(std))
    for i in eachindex(std)
        main = map_basis(std.ξ[i], mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        if ND == 1
            Ja[i] = SMatrix{1,1}(dual[1])
            J[i] = main[1][1]
        elseif ND == 2
            Ja[i] = SMatrix{2,2}(vcat(dual[1], dual[2]))
            J[i] = dual[1][1] * dual[2][2] - dual[1][2] * dual[2][1]
        else # ND == 3
            Ja[i] = SMatrix{3,3}(vcat(dual[1], dual[2], dual[3]))
            J[i] = dot(main[1], cross(dual[2], dual[3]))
        end
    end

    # Mass matrix
    M = massmatrix(std, J)
    Mfac = M |> factorize

    # Volume
    vol = sum(M)

    # Subgrid
    subgrid = if sub
        PhysicalSubgrid(std, mesh, ie)
    else
        nothing
    end

    return PhysicalElement(xyz, M, Mfac, Ja, vol, subgrid)
end

struct PhysicalSubgrid{ND,RT}
    n::NTuple{ND,Array{SVector{ND,RT},ND}}
    t::NTuple{ND,Array{SVector{ND,RT},ND}}
    b::NTuple{ND,Array{SVector{ND,RT},ND}}
    Jf::NTuple{ND,Array{RT,ND}}
end

function PhysicalSubgrid(std, ::CartesianMesh{1,RT}, _) where {RT}
    nx = length(std) + 1
    n = (fill(SVector(one(RT)), nx),)
    t = (fill(SVector(zero(RT)), nx),)
    b = (fill(SVector(zero(RT)), nx),)
    Jf = (fill(one(RT), nx),)
    return PhysicalSubgrid(n, t, b, Jf)
end

function PhysicalSubgrid(std, mesh::CartesianMesh{2,RT}, _) where {RT}
    (; Δx) = mesh
    nx, ny = size(std)
    n = (
        fill(SVector(one(RT), zero(RT)), nx + 1, ny),
        fill(SVector(zero(RT), one(RT)), nx, ny + 1),
    )
    t = (
        fill(SVector(zero(RT), one(RT)), nx + 1, ny),
        fill(SVector(-one(RT), zero(RT)), nx, ny + 1),
    )
    b = (
        fill(SVector(zero(RT), zero(RT)), nx + 1, ny),
        fill(SVector(zero(RT), zero(RT)), nx, ny + 1),
    )
    Jf = (
        fill(Δx[2] / 2, nx + 1, ny),
        fill(Δx[1] / 2, nx, ny + 1),
    )
    return PhysicalSubgrid(n, t, b, Jf)
end

function PhysicalSubgrid(std, mesh::CartesianMesh{3,RT}, _) where {RT}
    (; Δx) = mesh
    nx, ny, nz = size(std)
    n = (
        fill(SVector(one(RT), zero(RT), zero(RT)), nx + 1, ny, nz),
        fill(SVector(zero(RT), one(RT), zero(RT)), nx, ny + 1, nz),
        fill(SVector(zero(RT), zero(RT), one(RT)), nx, ny, nz + 1),
    )
    t = (
        fill(SVector(zero(RT), one(RT), zero(RT)), nx + 1, ny, nz),
        fill(SVector(zero(RT), zero(RT), one(RT)), nx, ny + 1, nz),
        fill(SVector(one(RT), zero(RT), zero(RT)), nx, ny, nz + 1),
    )
    b = (
        fill(SVector(zero(RT), zero(RT), one(RT)), nx + 1, ny, nz),
        fill(SVector(one(RT), zero(RT), zero(RT)), nx, ny + 1, nz),
        fill(SVector(zero(RT), one(RT), zero(RT)), nx, ny, nz + 1),
    )
    Jf = (
        fill(Δx[2] * Δx[3] / 4, nx + 1, ny, nz),
        fill(Δx[1] * Δx[3] / 4, nx, ny + 1, nz),
        fill(Δx[1] * Δx[2] / 4, nx, ny, nz + 1),
    )
    return PhysicalElement(n, t, b, Jf)
end

function PhysicalSubgrid(std, mesh::StepMesh{RT}, ie) where {RT}
    (; Δx) = mesh
    ireg = get_region(mesh, ie)
    nx, ny = size(std)
    n = (
        fill(SVector(one(RT), zero(RT)), nx + 1, ny),
        fill(SVector(zero(RT), one(RT)), nx, ny + 1),
    )
    t = (
        fill(SVector(zero(RT), one(RT)), nx + 1, ny),
        fill(SVector(-one(RT), zero(RT)), nx, ny + 1),
    )
    b = (
        fill(SVector(zero(RT), zero(RT)), nx + 1, ny),
        fill(SVector(zero(RT), zero(RT)), nx, ny + 1),
    )
    Jf = (
        fill(Δx[ireg][2] / 2, nx + 1, ny),
        fill(Δx[ireg][1] / 2, nx, ny + 1),
    )
    return PhysicalSubgrid(n, t, b, Jf)
end

function PhysicalSubgrid(_, ::UnstructuredMesh, _)
    error("Not implemented")
end

struct PhysicalFace{ND,RT,MM}
    coords::Vector{SVector{ND,RT}}
    n::Vector{SVector{ND,RT}}
    t::Vector{SVector{ND,RT}}
    b::Vector{SVector{ND,RT}}
    J::Vector{RT}
    M::MM
    surface::RT
end

function PhysicalFace(_, mesh::CartesianMesh{1,RT}, iface) where {RT}
    pos = get_face(mesh, iface).elempos[1]
    nind = get_face(mesh, iface).nodeinds[1]
    x = [get_vertex(mesh, nind)]
    J = [one(RT)]
    if pos == 1
        n = [SVector(-one(RT))]
        t = [SVector(zero(RT))]
        b = [SVector(zero(RT))]
    else # pos == 2
        n = [SVector(one(RT))]
        t = [SVector(zero(RT))]
        b = [SVector(zero(RT))]
    end
    M = SMatrix{1,1}(zero(RT))
    surf = one(RT)
    return PhysicalFace(x, n, t, b, J, M, surf)
end

function PhysicalFace(std, mesh::CartesianMesh{2,RT}, iface) where {RT}
    (; Δx) = mesh
    pos = get_face(mesh, iface).elempos[1]
    if pos == 1 || pos == 2  # Vertical
        fstd = get_face(std, 2)
        J = fill(Δx[2] / 2, ndofs(fstd))
        if pos == 1
            n = fill(SVector(-one(RT), zero(RT)), ndofs(fstd))
            t = fill(SVector(zero(RT), -one(RT)), ndofs(fstd))
            b = fill(SVector(zero(RT), zero(RT)), ndofs(fstd))
        else # pos == 2
            n = fill(SVector(one(RT), zero(RT)), ndofs(fstd))
            t = fill(SVector(zero(RT), one(RT)), ndofs(fstd))
            b = fill(SVector(zero(RT), zero(RT)), ndofs(fstd))
        end

    else  # pos == 3 || pos == 4  # Horizontal
        fstd = get_face(std, 1)
        J = fill(Δx[1] / 2, ndofs(fstd))
        if pos == 3
            n = fill(SVector(zero(RT), -one(RT)), ndofs(fstd))
            t = fill(SVector(one(RT), zero(RT)), ndofs(fstd))
            b = fill(SVector(zero(RT), zero(RT)), ndofs(fstd))
        else # pos == 4
            n = fill(SVector(zero(RT), one(RT)), ndofs(fstd))
            t = fill(SVector(-one(RT), zero(RT)), ndofs(fstd))
            b = fill(SVector(zero(RT), zero(RT)), ndofs(fstd))
        end
    end

    xy = [face_phys_coords(ξ, mesh, iface) for ξ in fstd.ξ]
    M = massmatrix(fstd, J)
    surf = sum(M)
    return PhysicalFace(xy, n, t, b, J, M, surf)
end

function PhysicalFace(_, ::CartesianMesh{3,RT}, _) where {RT}
    error("Not implemented yet!")
end

function PhysicalFace(std, mesh::StepMesh{RT}, iface) where {RT}
    (; Δx) = mesh
    ie = get_face(mesh, iface).eleminds[1]
    ireg = get_region(mesh, ie)
    pos = get_face(mesh, iface).elempos[1]

    if pos == 1 || pos == 2  # Vertical
        fstd = get_face(std, 2)
        J = fill(Δx[ireg][2] / 2, ndofs(fstd))
        xy = [face_phys_coords(ξ, mesh, iface) for ξ in fstd.ξ]
        if pos == 1
            n = fill(SVector(-one(RT), zero(RT)), ndofs(fstd))
            t = fill(SVector(zero(RT), -one(RT)), ndofs(fstd))
            b = fill(SVector(zero(RT), zero(RT)), ndofs(fstd))
        else # pos == 2
            n = fill(SVector(one(RT), zero(RT)), ndofs(fstd))
            t = fill(SVector(zero(RT), one(RT)), ndofs(fstd))
            b = fill(SVector(zero(RT), zero(RT)), ndofs(fstd))
        end

    else  # pos == 3 || pos == 4  # Horizontal
        fstd = get_face(std, 1)
        J = fill(Δx[ireg][1] / 2, ndofs(fstd))
        xy = [face_phys_coords(ξ, mesh, iface) for ξ in fstd.ξ]
        if pos == 3
            n = fill(SVector(zero(RT), -one(RT)), ndofs(fstd))
            t = fill(SVector(one(RT), zero(RT)), ndofs(fstd))
            b = fill(SVector(zero(RT), zero(RT)), ndofs(fstd))
        else # pos == 4
            n = fill(SVector(zero(RT), one(RT)), ndofs(fstd))
            t = fill(SVector(-one(RT), zero(RT)), ndofs(fstd))
            b = fill(SVector(zero(RT), zero(RT)), ndofs(fstd))
        end
    end

    M = massmatrix(fstd, J)
    surf = sum(M)
    return PhysicalFace(xy, n, t, b, J, M, surf)
end

function PhysicalFace(_, mesh::UnstructuredMesh{1,RT}, iface) where {RT}
    pos = get_face(mesh, iface).elempos[1]
    nind = get_face(mesh, iface).nodeinds[1]
    x = [get_vertex(mesh, nind)]
    J = [one(RT)]
    if pos == 1
        n = [SVector(-one(RT))]
        t = [SVector(zero(RT))]
        b = [SVector(zero(RT))]
    else # pos == 2
        n = [SVector(one(RT))]
        t = [SVector(zero(RT))]
        b = [SVector(zero(RT))]
    end
    M = SMatrix{1,1}(zero(RT))
    surf = one(RT)
    return PhysicalFace(x, n, t, b, J, M, surf)
end

function PhysicalFace(std, mesh::UnstructuredMesh{2,RT}, iface) where {RT}
    face = get_face(mesh, iface)
    ielem = face.eleminds[1]
    pos = face.elempos[1]
    if pos == 1 || pos == 2  # Vertical
        fstd = get_face(std, 2)
    else # pos == 3 || pos == 4  # Horizontal
        fstd = get_face(std, 1)
    end

    xy = Vector{SVector{2,RT}}(undef, ndofs(fstd))
    n = Vector{SVector{2,RT}}(undef, ndofs(fstd))
    t = Vector{SVector{2,RT}}(undef, ndofs(fstd))
    b = Vector{SVector{2,RT}}(undef, ndofs(fstd))
    J = Vector{RT}(undef, ndofs(fstd))
    if pos == 1  # Left
        for i in eachindex(fstd)
            ξ = SVector(-one(RT), fstd.ξ[i][1])
            xy[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n[i] = -dual[1]
            t[i] = -main[2] / norm(main[2])
            b[i] = SVector(zero(RT), zero(RT))
            J[i] = norm(n[i])
            n[i] /= J[i]
        end
    elseif pos == 2  # Right
        for i in eachindex(fstd)
            ξ = SVector(one(RT), fstd.ξ[i][1])
            xy[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n[i] = dual[1]
            t[i] = main[2] / norm(main[2])
            b[i] = SVector(zero(RT), zero(RT))
            J[i] = norm(n[i])
            n[i] /= J[i]
        end
    elseif pos == 3  # Bottom
        for i in eachindex(fstd)
            ξ = SVector(fstd.ξ[i][1], -one(RT))
            xy[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n[i] = -dual[2]
            t[i] = main[1] / norm(main[1])
            b[i] = SVector(zero(RT), zero(RT))
            J[i] = norm(n[i])
            n[i] /= J[i]
        end
    else # pos == 4  # Top
        for i in eachindex(fstd)
            ξ = SVector(fstd.ξ[i][1], one(RT))
            xy[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n[i] = dual[2]
            t[i] = -main[1] / norm(main[1])
            b[i] = SVector(zero(RT), zero(RT))
            J[i] = norm(n[i])
            n[i] /= J[i]
        end
    end

    M = massmatrix(fstd, J)
    surf = sum(M)
    return PhysicalFace(xy, n, t, b, J, M, surf)
end

function PhysicalFace(_, ::UnstructuredMesh{3,RT}, _) where {RT}
    error("Not implemented yet!")
end

function compute_metric_terms(stdvec, dh, mesh, subgrid=false)
    # Element geometry
    physvec = [
        PhysicalElement(
            stdvec[loc2reg(dh, ie).first],
            mesh,
            ie,
            subgrid,
        )
        for ie in eachelement(dh)
    ]
    ev = StructVector(physvec, unwrap=T -> T <: PhysicalSubgrid)

    # Face geometry
    facevec = []
    for iface in eachface(mesh)
        ielem = get_face(mesh, iface).eleminds[1]
        ireg = loc2reg(dh, ielem).first
        push!(
            facevec,
            PhysicalFace(stdvec[ireg], mesh, iface),
        )
    end
    fv = StructVector([facevec...])
    return ev, fv
end

get_elements(v::StructVector{<:PhysicalElement}) = LazyRows(v)
elementgrids(v::StructVector{<:PhysicalElement}) = LazyRows(v.subgrid)
get_faces(v::StructVector{<:PhysicalFace}) = LazyRows(v)

get_element(v::StructVector{<:PhysicalElement}, i) = LazyRow(v, i)
elementgrid(v::StructVector{<:PhysicalElement}, i) = LazyRow(v.subgrid, i)
get_face(v::StructVector{<:PhysicalFace}, i) = LazyRow(v, i)

phys_coords(e::PhysicalElement) = e.coords
phys_coords(v::StructVector{<:PhysicalElement}, i) = LazyRow(v, i).coords
phys_coords(f::PhysicalFace) = f.coords
phys_coords(v::StructVector{<:PhysicalFace}, i) = LazyRow(v, i).coords

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

function covariant(F̃, Ja::SMatrix{1,1})
    return SVector(F̃ / Ja[1, 1])
end

function covariant(F̃, Ja::SMatrix{2,2})
    error("Not implemented yet!")
end

function covariant(F̃, Ja::SMatrix{3,3})
    error("Not implemented yet!")
end
