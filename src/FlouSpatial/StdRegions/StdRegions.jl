include("ApproximationBases.jl")
include("Reconstruction.jl")

abstract type AbstractStdRegion{ND} end

"""
    basis(std)

Return the approximation basis of `std`.
"""
function basis(s::AbstractStdRegion)
    return s.basis
end

"""
    dofsize(std, i)

Return the number of nodes in the `i` direction (equivalent to `dofsize(std)` with
non-tensor-product regions).
"""
function dofsize(s::AbstractStdRegion, _::Integer)
    return s |> basis |> nnodes
end

"""
    dofsize(std)

Return the number of nodes in each spatial direction (for non-tensor-product regions simply
return the total number of nodes).

See also [`ndofs`](@ref), [`eachdof`](@ref), [`equisize`](@ref).
"""
function dofsize(s::AbstractStdRegion)
    nd = spatialdim(s)
    ntuple(_ -> s |> basis |> nnodes, nd)
end

"""
    lineardofs(std)

Return a `LinearIndices` object useful to convert indices from tensor to linear form.

See also [`cartesiandofs`](@ref), [`tpdofs`](@ref).
"""
function lineardofs(s::AbstractStdRegion)
    return LinearIndices(dofsize(s))
end

"""
    cartesiandofs(std)

Return a `CartesianIndices` object useful to convert indices from linear to tensor form.

See also [`lineardofs`](@ref), [`tpdofs`](@ref).
"""
function cartesiandofs(s::AbstractStdRegion)
    return CartesianIndices(dofsize(s))
end

"""
    tpdofs(std, ::Val(dir))

Return the tensor-product indices of the nodes in `std`. For example, for a 2D element
`tpdofs(std, Val(1))` returns the indices of the nodes in the first direction for each
row of nodes in the second dimension.

See also [`tpdofs_subgrid`](@ref).
"""
function tpdofs(::AbstractStdRegion, _)
    # Specific implementations for each type
    return nothing
end

"""
    ndofs(std, i)

Return the number of nodes in the `i` direction (equivalent to `ndofs(std)` in
non-tensor-product elements).
"""
function ndofs(s::AbstractStdRegion, i::Integer)
    return dofsize(s, i)
end

"""
    ndofs(std)

Return the number of nodes.

See also [`eachdof`](@ref), [`dofsize`](@ref).
"""
function ndofs(s::AbstractStdRegion)
    return dofsize(s) |> prod
end

"""
    eachdof(std, i)

Return a range that covers all the indices of the nodes in the `i` direction (equivalent to
`eachdof(std)` in non-tensor-product regions).
"""
function eachdof(s::AbstractStdRegion, i::Integer)
    return Base.OneTo(ndofs(s, i))
end

"""
    eachdof(std)

Return a range that covers all the indices of the nodes.

See also [`ndofs`](@ref), [`dofsize`](@ref).
"""
function eachdof(s::AbstractStdRegion)
    return Base.OneTo(ndofs(s))
end

"""
    nequispaced(std)

Return the number of equispaced nodes, useful for plotting and/or saving to files.
"""
function nequispaced(s::AbstractStdRegion)
    return length(s.ξe)
end


"""
    equisize(std, i)

Return the number of equispaced nodes in direction `i`, useful for plotting and/or saving
to files.
"""
function equisize(s::AbstractStdRegion{ND}, _) where {ND}
    n = nequispaced(s)
    return ND == 1 ? n : (ND == 2 ? Int(sqrt(n)) : Int(cbrt(n)))
end

"""
    equisize(std)

Return the number of equispaced nodes in each spatial direction, useful for plotting
and/or saving to files (for non-tensor-product regions simply return the total number of
nodes).
"""
function equisize(s::AbstractStdRegion{ND}) where {ND}
    return ntuple(i -> equisize(s, i), ND)
end

"""
    tpdofs_subgrid(std, ::Val(dir))

Return the tensor-product indices of the subrid nodes in `std`. For example, for a 2D
element `tpdofs_subgrid(std, Val(1))` returns the indices of the nodes in the first
direction for each row of nodes in the second dimension.

See also [`tpdofs`](@ref).
"""
function tpdofs_subgrid(::AbstractStdRegion, _)
    # Specific implementations for each type
    return nothing
end

"""
    spatialdim(std)

Return the spatial dimension of `std`.
"""
function FlouCommon.spatialdim(::AbstractStdRegion{ND}) where {ND}
    return ND
end

"""
    eachdim(std)

Return a range that covers all the spatial dimensions of `std`.
"""
function FlouCommon.eachdim(s::AbstractStdRegion)
    return Base.OneTo(spatialdim(s))
end

"""
    ndirections(std)

Return the number of directions of `std`. In tensor-product regions it is equivalent to
`spatialdim(std)`.

See also [`eachdirection`](@ref).
"""
function ndirections(s::AbstractStdRegion)
    return spatialdim(s)
end

"""
    eachdirection(std)

Return a range that covers all the directions of `std`.

See also [`ndirections`](@ref).
"""
function eachdirection(s::AbstractStdRegion)
    return Base.OneTo(ndirections(s))
end

"""
    is_tensor_product(std)

Return `true` if `std` is a tensor-product region.
"""
function is_tensor_product(::AbstractStdRegion)
    return true
end

"""
    project2equispaced!(Qe, Q, std)

Project the nodal solution values `Q` in region `std` to equispaced nodes, `Qe`.
"""
function project2equispaced!(
    Qe::AbstractVecOrMat,
    Q::AbstractVecOrMat,
    s::AbstractStdRegion,
)
    return mul!(Qe, s.node2eq, Q)
end

"""
    reconstruction(std)

Return the reconstruction used in `std`.
"""
function reconstruction(s::AbstractStdRegion)
    return s.reconstruction
end

#==========================================================================================#
#                                     Temporary cache                                      #

struct StdRegionCache{NV,S,B}
    scalar::Vector{NTuple{3,StateVector{1,S}}}
    state::Vector{NTuple{3,StateVector{NV,S}}}
    block::Vector{NTuple{3,BlockVector{NV,B}}}
    sharp::Vector{NTuple{3,BlockVector{NV,B}}}
    subcell::Vector{NTuple{3,StateVector{NV,S}}}
end

function StdRegionCache{NV}(nd, np, ftype=Float64) where {NV}
    nthr = Threads.nthreads()
    npts = np^nd
    nsub = np^(nd - 1) * (np + 1)
    tmps = [
        ntuple(_ -> StateVector{1}(undef, npts, ftype), 3)
        for _ in 1:nthr
    ]
    tmpst = [
        ntuple(_ -> StateVector{NV}(undef, npts, ftype), 3)
        for _ in 1:nthr
    ]
    tmpb = [
        ntuple(_ -> BlockVector{NV}(undef, npts, nd, ftype), 3)
        for _ in 1:nthr
    ]
    tmp♯ = [
        ntuple(i -> BlockVector{NV}(undef, np, npts, ftype), 3)
        for _ in 1:nthr
    ]
    tmpsubcell = [
        ntuple(i -> StateVector{NV}(undef, nsub, ftype), 3)
        for _ in 1:nthr
    ]
    return StdRegionCache(tmps, tmpst, tmpb, tmp♯, tmpsubcell)
end

#==========================================================================================#
#                                     Implementations                                      #

include("StdPoint.jl")
include("StdSegment.jl")
include("StdQuad.jl")
include("StdHex.jl")
