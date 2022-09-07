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
    nx = ndofs(std) + 1
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
    return PhysicalSubgrid(n, t, b, Jf)
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

function PhysicalSubgrid(std, mesh::UnstructuredMesh{1,RT}, ie) where {RT}
    nx = ndofs(std)
    n = (Vector{SVector{1,RT}}(undef, nx + 1),)
    t = (Vector{SVector{1,RT}}(undef, nx + 1),)
    b = (Vector{SVector{1,RT}}(undef, nx + 1),)
    Jf = (Vector{RT}(undef, nx + 1),)
    for i in 1:(nx + 1)
        ξ = std.ξc[1][i]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        s = map_jacobian(main, mesh, ie) |> sign
        n[1][i] = dual[1] * s
        t[1][i] = SVector(zero(RT))
        b[1][i] = SVector(zero(RT))
        Jf[1][i] = one(RT)
    end
    return PhysicalSubgrid(n, t, b, Jf)
end

function PhysicalSubgrid(std, mesh::UnstructuredMesh{2,RT}, ie) where {RT}
    nx, ny = size(std)
    n = (Matrix{SVector{2,RT}}(undef, nx + 1, ny), Matrix{SVector{2,RT}}(undef, nx, ny + 1))
    t = (Matrix{SVector{2,RT}}(undef, nx + 1, ny), Matrix{SVector{2,RT}}(undef, nx, ny + 1))
    b = (Matrix{SVector{2,RT}}(undef, nx + 1, ny), Matrix{SVector{2,RT}}(undef, nx, ny + 1))
    Jf = (Matrix{RT}(undef, nx + 1, ny), Matrix{RT}(undef, nx, ny + 1))
    for j in 1:ny, i in 1:(nx + 1)  # Vertical faces
        ξ = std.ξc[1][i, j]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        s = map_jacobian(main, mesh, ie) |> sign
        n[1][i, j] = dual[1] * s
        t[1][i, j] = normalize(main[2]) * s
        b[1][i, j] = SVector(zero(RT), zero(RT))
        Jf[1][i, j] = norm(n[1][i, j])
        n[1][i, j] /= Jf[1][i, j]
    end
    for j in 1:(ny + 1), i in 1:nx  # Horizontal faces
        ξ = std.ξc[2][i, j]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        s = map_jacobian(main, mesh, ie) |> sign
        n[2][i, j] = dual[2] * s
        t[2][i, j] = -normalize(main[1]) * s
        b[2][i, j] = SVector(zero(RT), zero(RT))
        Jf[2][i, j] = norm(n[2][i, j])
        n[2][i, j] /= Jf[2][i, j]
    end
    return PhysicalSubgrid(n, t, b, Jf)
end

function PhysicalSubgrid(std, mesh::UnstructuredMesh{3,RT}, ie) where {RT}
    nx, ny, nz = size(std)
    n = (
        Array{SVector{3,RT},3}(undef, nx + 1, ny, nz),
        Array{SVector{3,RT},3}(undef, nx, ny + 1, nz),
        Array{SVector{3,RT},3}(undef, nx, ny, nz + 1),
    )
    t = (
        Array{SVector{3,RT},3}(undef, nx + 1, ny, nz),
        Array{SVector{3,RT},3}(undef, nx, ny + 1, nz),
        Array{SVector{3,RT},3}(undef, nx, ny, nz + 1),
    )
    b = (
        Array{SVector{3,RT},3}(undef, nx + 1, ny, nz),
        Array{SVector{3,RT},3}(undef, nx, ny + 1, nz),
        Array{SVector{3,RT},3}(undef, nx, ny, nz + 1),
    )
    Jf = (
        Array{RT,3}(undef, nx + 1, ny, nz),
        Array{RT,3}(undef, nx, ny + 1, nz),
        Array{RT,3}(undef, nx, ny, nz + 1),
    )
    for k in 1:nz, j in 1:ny, i in 1:(nx + 1)  # X faces
        ξ = std.ξc[1][i, j, k]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        n[1][i, j, k] = dual[1]
        t[1][i, j, k] = normalize(main[2])
        b[1][i, j, k] = normalize(cross(n[1][i, j, k], t[1][i, j, k]))
        Jf[1][i, j, k] = norm(n[1][i, j, k])
        n[1][i, j, k] /= Jf[1][i, j, k]
    end
    for k in 1:nz, j in 1:(ny + 1), i in 1:nx  # Y faces
        ξ = std.ξc[2][i, j, k]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        n[2][i, j, k] = dual[2]
        t[2][i, j, k] = normalize(main[3])
        b[2][i, j, k] = normalize(cross(n[2][i, j, k], t[2][i, j, k]))
        Jf[2][i, j, k] = norm(n[2][i, j, k])
        n[2][i, j, k] /= Jf[2][i, j, k]
    end
    for k in 1:(nz + 1), j in 1:ny, i in 1:nx  # Z faces
        ξ = std.ξc[3][i, j, k]
        main = map_basis(ξ, mesh, ie)
        dual = map_dual_basis(main, mesh, ie)
        n[3][i, j, k] = dual[3]
        t[3][i, j, k] = normalize(main[1])
        b[3][i, j, k] = normalize(cross(n[3][i, j, k], t[3][i, j, k]))
        Jf[3][i, j, k] = norm(n[3][i, j, k])
        n[3][i, j, k] /= Jf[3][i, j, k]
    end
    return PhysicalSubgrid(n, t, b, Jf)
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
        fstd = get_face(std, 1)
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
        fstd = get_face(std, 2)
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

function PhysicalFace(std, mesh::CartesianMesh{3,RT}, iface) where {RT}
    (; Δx) = mesh
    pos = get_face(mesh, iface).elempos[1]
    if pos == 1 || pos == 2  # X faces
        fstd = get_face(std, 1)
        J = fill(Δx[2] * Δx[3] / 4, ndofs(fstd))
        if pos == 1
            n = fill(SVector(-one(RT), zero(RT), zero(RT)), ndofs(fstd))
            t = fill(SVector(zero(RT), -one(RT), zero(RT)), ndofs(fstd))
            b = fill(SVector(zero(RT), zero(RT), one(RT)), ndofs(fstd))
        else # pos == 2
            n = fill(SVector(one(RT), zero(RT), zero(RT)), ndofs(fstd))
            t = fill(SVector(zero(RT), one(RT), zero(RT)), ndofs(fstd))
            b = fill(SVector(zero(RT), zero(RT), one(RT)), ndofs(fstd))
        end
    elseif pos == 3 || pos == 4  # Y faces
        fstd = get_face(std, 2)
        J = fill(Δx[1] * Δx[3] / 4, ndofs(fstd))
        if pos == 3
            n = fill(SVector(zero(RT), -one(RT), zero(RT)), ndofs(fstd))
            t = fill(SVector(zero(RT), zero(RT), -one(RT)), ndofs(fstd))
            b = fill(SVector(one(RT), zero(RT), zero(RT)), ndofs(fstd))
        else # pos == 4
            n = fill(SVector(zero(RT), one(RT), zero(RT)), ndofs(fstd))
            t = fill(SVector(zero(RT), zero(RT), one(RT)), ndofs(fstd))
            b = fill(SVector(one(RT), zero(RT), zero(RT)), ndofs(fstd))
        end
    else # pos == 5 || pos == 6  # Z faces
        fstd = get_face(std, 3)
        J = fill(Δx[1] * Δx[2] / 4, ndofs(fstd))
        if pos == 5
            n = fill(SVector(zero(RT), zero(RT), -one(RT)), ndofs(fstd))
            t = fill(SVector(-one(RT), zero(RT), zero(RT)), ndofs(fstd))
            b = fill(SVector(zero(RT), one(RT), zero(RT)), ndofs(fstd))
        else # pos == 6
            n = fill(SVector(zero(RT), zero(RT), one(RT)), ndofs(fstd))
            t = fill(SVector(one(RT), zero(RT), zero(RT)), ndofs(fstd))
            b = fill(SVector(zero(RT), one(RT), zero(RT)), ndofs(fstd))
        end
    end

    xy = [face_phys_coords(ξ, mesh, iface) for ξ in fstd.ξ]
    M = massmatrix(fstd, J)
    surf = sum(M)
    return PhysicalFace(xy, n, t, b, J, M, surf)
end

function PhysicalFace(std, mesh::StepMesh{RT}, iface) where {RT}
    (; Δx) = mesh
    ie = get_face(mesh, iface).eleminds[1]
    ireg = get_region(mesh, ie)
    pos = get_face(mesh, iface).elempos[1]

    if pos == 1 || pos == 2  # Vertical
        fstd = get_face(std, 1)
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
        fstd = get_face(std, 2)
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

function PhysicalFace(std, mesh::UnstructuredMesh{1,RT}, iface) where {RT}
    face = get_face(mesh, iface)
    ielem = face.eleminds[1]
    pos = face.elempos[1]
    x = Vector{SVector{1,RT}}(undef, 1)
    n = Vector{SVector{1,RT}}(undef, 1)
    t = Vector{SVector{1,RT}}(undef, 1)
    b = Vector{SVector{1,RT}}(undef, 1)
    J = Vector{RT}(undef, 1)
    if pos == 1
        ξ = SVector(-one(RT))
        x[1] = phys_coords(ξ, mesh, ielem)
        main = map_basis(ξ, mesh, ielem)
        dual = map_dual_basis(main, mesh, ielem)
        s = map_jacobian(main, mesh, ielem) |> sign
        n[1] = -dual[1] * s
        t[1] = SVector(zero(RT))
        b[1] = SVector(zero(RT))
        J[1] = norm(n[1])
        n[1] /= J[1]
    else # pos == 2
        ξ = SVector(one(RT))
        x[1] = phys_coords(ξ, mesh, ielem)
        main = map_basis(ξ, mesh, ielem)
        dual = map_dual_basis(main, mesh, ielem)
        s = map_jacobian(main, mesh, ielem) |> sign
        n[1] = dual[1] * s
        t[1] = SVector(zero(RT))
        b[1] = SVector(zero(RT))
        J[1] = norm(n[1])
        n[1] /= J[1]
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
        fstd = get_face(std, 1)
    else # pos == 3 || pos == 4  # Horizontal
        fstd = get_face(std, 2)
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
            s = map_jacobian(main, mesh, ielem) |> sign
            n[i] = -dual[1] * s
            t[i] = -normalize(main[2]) * s
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
            s = map_jacobian(main, mesh, ielem) |> sign
            n[i] = dual[1] * s
            t[i] = normalize(main[2]) * s
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
            s = map_jacobian(main, mesh, ielem) |> sign
            n[i] = -dual[2] * s
            t[i] = normalize(main[1]) * s
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
            s = map_jacobian(main, mesh, ielem) |> sign
            n[i] = dual[2] * s
            t[i] = -normalize(main[1]) * s
            b[i] = SVector(zero(RT), zero(RT))
            J[i] = norm(n[i])
            n[i] /= J[i]
        end
    end

    M = massmatrix(fstd, J)
    surf = sum(M)
    return PhysicalFace(xy, n, t, b, J, M, surf)
end

function PhysicalFace(std, mesh::UnstructuredMesh{3,RT}, iface) where {RT}
    face = get_face(mesh, iface)
    ielem = face.eleminds[1]
    pos = face.elempos[1]
    if pos == 1 || pos == 2  # X faces
        fstd = get_face(std, 1)
    elseif pos == 3 || pos == 4  # Y faces
        fstd = get_face(std, 2)
    else # pos == 5 || pos == 6  # Z faces
        fstd = get_face(std, 3)
    end

    xyz = Vector{SVector{3,RT}}(undef, ndofs(fstd))
    n = Vector{SVector{3,RT}}(undef, ndofs(fstd))
    t = Vector{SVector{3,RT}}(undef, ndofs(fstd))
    b = Vector{SVector{3,RT}}(undef, ndofs(fstd))
    J = Vector{RT}(undef, ndofs(fstd))
    if pos == 1  # -X face
        for i in eachindex(fstd)
            ξ = SVector(-one(RT), fstd.ξ[i][1], fstd.ξ[i][2])
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n[i] = -dual[1]
            t[i] = -normalize(main[2])
            b[i] = normalize(cross(n[i], t[i]))
            J[i] = norm(n[i])
            n[i] /= J[i]
        end
    elseif pos == 2  # +X face
        for i in eachindex(fstd)
            ξ = SVector(one(RT), fstd.ξ[i][1], fstd.ξ[i][2])
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n[i] = dual[1]
            t[i] = normalize(main[2])
            b[i] = normalize(cross(n[i], t[i]))
            J[i] = norm(n[i])
            n[i] /= J[i]
        end
    elseif pos == 3  # -Y face
        for i in eachindex(fstd)
            ξ = SVector(fstd.ξ[i][1], -one(RT), fstd.ξ[i][2])
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n[i] = -dual[2]
            t[i] = -normalize(main[3])
            b[i] = normalize(cross(n[i], t[i]))
            J[i] = norm(n[i])
            n[i] /= J[i]
        end
    elseif pos == 4  # +Y face
        for i in eachindex(fstd)
            ξ = SVector(fstd.ξ[i][1], one(RT), fstd.ξ[i][2])
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n[i] = dual[2]
            t[i] = normalize(main[3])
            b[i] = normalize(cross(n[i], t[i]))
            J[i] = norm(n[i])
            n[i] /= J[i]
        end
    elseif pos == 5  # -Z face
        for i in eachindex(fstd)
            ξ = SVector(fstd.ξ[i][1], fstd.ξ[i][2], -one(RT))
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n[i] = -dual[3]
            t[i] = -normalize(main[1])
            b[i] = normalize(cross(n[i], t[i]))
            J[i] = norm(n[i])
            n[i] /= J[i]
        end
    else # pos == 6  # +Z face
        for i in eachindex(fstd)
            ξ = SVector(fstd.ξ[i][1], fstd.ξ[i][2], one(RT))
            xyz[i] = phys_coords(ξ, mesh, ielem)
            main = map_basis(ξ, mesh, ielem)
            dual = map_dual_basis(main, mesh, ielem)
            n[i] = dual[3]
            t[i] = normalize(main[1])
            b[i] = normalize(cross(n[i], t[i]))
            J[i] = norm(n[i])
            n[i] /= J[i]
        end
    end

    M = massmatrix(fstd, J)
    surf = sum(M)
    return PhysicalFace(xyz, n, t, b, J, M, surf)
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
