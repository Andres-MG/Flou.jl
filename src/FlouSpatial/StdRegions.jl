abstract type AbstractStdRegion{ND,NP,NE} end

"""
    dofsize(std; equispaced=Val(false))

Return the number of solution nodes in each spatial direction (for non-tensor-product
regions simply return the total number of nodes).

See also [`ndofs`](@ref), [`eachdof`](@ref).
"""
function dofsize(
    ::AbstractStdRegion{ND,NP,NE};
    equispaced::Val{E}=Val(false),
) where {
    ND,
    NP,
    NE,
    E,
}
    return E ? ntuple(_ -> NE, ND) : ntuple(_ -> NP, ND)
end

"""
    dofsize(std, i; equispaced=Val(false))

Return the number of solution nodes in the `i` direction (equivalent to `dofsize(std)` with
non-tensor-product regions).
"""
function dofsize(s::AbstractStdRegion, i::Integer; equispaced=Val(false))
    return dofsize(s; equispaced)[i]
end

"""
    lineardofs(std; equispaced=Val(false))

Return a `LinearIndices` object useful to convert indices from tensor to linear form.

See also [`cartesiandofs`](@ref).
"""
function lineardofs(s::AbstractStdRegion; equispaced=Val(false))
    return LinearIndices(dofsize(s; equispaced))
end

"""
    cartesiandofs(std; equispaced=Val(false))

Return a `CartesianIndices` object useful to convert indices from linear to tensor form.

See also [`lineardofs`](@ref).
"""
function cartesiandofs(s::AbstractStdRegion; equispaced=Val(false))
    return CartesianIndices(dofsize(s; equispaced))
end

"""
    ndofs(std; equispaced=Val(false))

Return the number of solution nodes.

See also [`eachdof`](@ref), [`dofsize`](@ref).
"""
function ndofs(s::AbstractStdRegion; equispaced=Val(false))
    return dofsize(s; equispaced) |> prod
end

"""
    ndofs(std, i; equispaced=Val(false))

Return the number of solution nodes in the `i` direction (equivalent to `ndofs(std)` in
non-tensor-product elements).
"""
function ndofs(s::AbstractStdRegion, i::Integer; equispaced=Val(false))
    return dofsize(s, i; equispaced)
end

"""
    eachdof(std; equispaced=Val(false))

Return a range that covers all the indices of the solution nodes.

See also [`ndofs`](@ref), [`dofsize`](@ref).
"""
function eachdof(s::AbstractStdRegion; equispaced=Val(false))
    return Base.OneTo(ndofs(s; equispaced))
end

"""
    eachdof(std, i; equispaced=Val(false))

Return a range that covers all the indices of the solution nodes in the `i` direction
(equivalent to `eachdof(std)` in non-tensor-product regions).
"""
function eachdof(s::AbstractStdRegion, i::Integer; equispaced=Val(false))
    return Base.OneTo(ndofs(s, i; equispaced))
end

"""
    spatialdim(std)

Return the spatial dimension of `std`.
"""
function FlouCommon.spatialdim(::AbstractStdRegion{ND}) where {ND}
    return ND
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
