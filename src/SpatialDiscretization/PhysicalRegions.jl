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
    xyz = [coords(ξ, mesh, ie) for ξ in std.ξ]

    # Jacobian and dual basis
    J = fill(prod(Δx) / (2^ND), ndofs(std))
    Ja = if ND == 1
        fill(SMatrix{1,1}(one(RT)), ndofs(std))
    elseif ND == 2
        fill(
            SMatrix{2,2}(
                Δx[2]/2, 0,
                0, Δx[1]/2,
            ),
            ndofs(std),
        )
    else # ND == 3
        fill(
            SMatrix{3,3}(
                Δx[2]*Δx[3]/4, 0, 0,
                0, Δx[1]*Δx[3]/4, 0,
                0, 0, Δx[1]*Δx[2]/4,
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
    ireg = region(mesh, ie)

    # Coordinates
    xy = [coords(ξ, mesh, ie) for ξ in std.ξ]

    # Jacobian and dual basis
    J = fill(prod(Δx[ireg]) / 4, ndofs(std))
    Ja = fill(
        SMatrix{2,2}(
            Δx[ireg][2]/2, 0,
            0, Δx[ireg][1]/2,
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

    return PhysicalElement(xy, M, Mfac, Ja, vol, subgrid)
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
    error("Not implemented yet!")
end

function PhysicalSubgrid(std, mesh::StepMesh{RT}, ie) where {RT}
    (; Δx) = mesh
    ireg = region(mesh, ie)
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
    pos = face(mesh, iface).elempos[1]
    nind = face(mesh, iface).nodeinds[1]
    x = [vertex(mesh, nind)]
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

function PhysicalFace(stdvec, mesh::CartesianMesh{2,RT}, iface) where {RT}
    (; Δx) = mesh
    std = stdvec[1]
    pos = face(mesh, iface).elempos[1]
    ninds = face(mesh, iface).nodeinds
    nodes = [vertex(mesh, i) for i in ninds]
    mapping = mesh.mappings[face(mesh, iface).mapind]

    if pos == 1 || pos == 2  # Vertical
        fstd = face(std, 2)
        J = fill(Δx[2] / 2, ndofs(fstd))
        xy = [coords(ξ, nodes, mapping) for ξ in fstd.ξ]
        if pos == 1
            n = [SVector(-one(RT), zero(RT)) for _ in eachindex(fstd)]
            t = [SVector(zero(RT), -one(RT)) for _ in eachindex(fstd)]
            b = [SVector(zero(RT), zero(RT)) for _ in eachindex(fstd)]
        else # pos == 2
            n = [SVector(one(RT), zero(RT)) for _ in eachindex(fstd)]
            t = [SVector(zero(RT), one(RT)) for _ in eachindex(fstd)]
            b = [SVector(zero(RT), zero(RT)) for _ in eachindex(fstd)]
        end

    else  # pos == 3 || pos == 4  # Horizontal
        fstd = face(std, 1)
        J = fill(Δx[1] / 2, ndofs(fstd))
        xy = [coords(ξ, nodes, mapping) for ξ in fstd.ξ]
        if pos == 3
            n = [SVector(zero(RT), -one(RT)) for _ in eachindex(fstd)]
            t = [SVector(one(RT), zero(RT)) for _ in eachindex(fstd)]
            b = [SVector(zero(RT), zero(RT)) for _ in eachindex(fstd)]
        else # pos == 4
            n = [SVector(zero(RT), one(RT)) for _ in eachindex(fstd)]
            t = [SVector(-one(RT), zero(RT)) for _ in eachindex(fstd)]
            b = [SVector(zero(RT), zero(RT)) for _ in eachindex(fstd)]
        end
    end

    M = massmatrix(fstd, J)
    surf = sum(M)
    return PhysicalFace(xy, n, t, b, J, M, surf)
end

function PhysicalFace(stdvec, mesh::CartesianMesh{3,RT}, iface) where {RT}
    error("Not implemented yet!")
end

function PhysicalFace(stdvec, mesh::StepMesh{RT}, iface) where {RT}
    (; Δx) = mesh
    ie = face(mesh, iface).eleminds[1]
    ireg = region(mesh, ie)
    std = stdvec[1]
    pos = face(mesh, iface).elempos[1]
    ninds = face(mesh, iface).nodeinds
    nodes = [vertex(mesh, i) for i in ninds]
    mapping = mesh.mappings[face(mesh, iface).mapind]

    if pos == 1 || pos == 2  # Vertical
        fstd = face(std, 2)
        J = fill(Δx[ireg][2] / 2, ndofs(fstd))
        xy = [coords(ξ, nodes, mapping) for ξ in fstd.ξ]
        if pos == 1
            n = [SVector(-one(RT), zero(RT)) for _ in eachindex(fstd)]
            t = [SVector(zero(RT), -one(RT)) for _ in eachindex(fstd)]
            b = [SVector(zero(RT), zero(RT)) for _ in eachindex(fstd)]
        else # pos == 2
            n = [SVector(one(RT), zero(RT)) for _ in eachindex(fstd)]
            t = [SVector(zero(RT), one(RT)) for _ in eachindex(fstd)]
            b = [SVector(zero(RT), zero(RT)) for _ in eachindex(fstd)]
        end

    else  # pos == 3 || pos == 4  # Horizontal
        fstd = face(std, 1)
        J = fill(Δx[ireg][1] / 2, ndofs(fstd))
        xy = [coords(ξ, nodes, mapping) for ξ in fstd.ξ]
        if pos == 3
            n = [SVector(zero(RT), -one(RT)) for _ in eachindex(fstd)]
            t = [SVector(one(RT), zero(RT)) for _ in eachindex(fstd)]
            b = [SVector(zero(RT), zero(RT)) for _ in eachindex(fstd)]
        else # pos == 4
            n = [SVector(zero(RT), one(RT)) for _ in eachindex(fstd)]
            t = [SVector(-one(RT), zero(RT)) for _ in eachindex(fstd)]
            b = [SVector(zero(RT), zero(RT)) for _ in eachindex(fstd)]
        end
    end

    M = massmatrix(fstd, J)
    surf = sum(M)
    return PhysicalFace(xy, n, t, b, J, M, surf)
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
        ie1, ie2 = face(mesh, iface).eleminds
        ir1 = loc2reg(dh, ie1).first
        ir2 = ie2 != 0 ? loc2reg(dh, ie2).first : ir1
        push!(
            facevec,
            PhysicalFace((stdvec[ir1], stdvec[ir2]), mesh, iface),
        )
    end
    fv = StructVector([facevec...])
    return ev, fv
end

elements(v::StructVector{<:PhysicalElement}) = LazyRows(v)
elementgrids(v::StructVector{<:PhysicalElement}) = LazyRows(v.subgrid)
faces(v::StructVector{<:PhysicalFace}) = LazyRows(v)

element(v::StructVector{<:PhysicalElement}, i) = LazyRow(v, i)
elementgrid(v::StructVector{<:PhysicalElement}, i) = LazyRow(v.subgrid, i)
face(v::StructVector{<:PhysicalFace}, i) = LazyRow(v, i)

coords(e::PhysicalElement) = e.coords
coords(v::StructVector{<:PhysicalElement}, i) = LazyRow(v, i).coords
coords(f::PhysicalFace) = f.coords
coords(v::StructVector{<:PhysicalFace}, i) = LazyRow(v, i).coords

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
