# Copyright (C) 2023 Andrés Mateo Gabín
#
# This file is part of Flou.jl.
#
# Flou.jl is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# Flou.jl is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Flou.jl. If
# not, see <https://www.gnu.org/licenses/>.

abstract type AbstractReconstruction end

"""
    reconstruction(type, ξ)

Return a tuple containing the derivative of the two reconstruction functions,
(``g_l``, ``g_r``) of `type` at the positions defined by `ξ`. When using nodal bases, `ξ` is
optional and the nodes of the basis will be used.
"""
function reconstruction end

"""
    basis(reconstruction)

Return the nodal basis used in `reconstruction` or `nothing`.
"""
function basis end

"""
    reconstruction_name(reconstruction)

Return a string representing the name of `reconstruction`.
"""
function reconstruction_name end

#==========================================================================================#
#                                          Nodal                                           #

struct NodalRec{B<:AbstractNodalBasis} <: AbstractReconstruction
    basis::B
    function NodalRec(basis::B) where {B<:AbstractNodalBasis}
        hasboundaries(basis) || throw(ArgumentError(
            "Nodal reconstruction requires a nodal basis with boundaries."
        ))
        return new{B}(basis)
    end
end

function reconstruction(r::NodalRec, _)
    D = derivative_matrix(r.basis.ξ, r.basis)
    return D[:, begin], D[:, end]
end

function basis(r::NodalRec)
    return r.basis
end

function reconstruction_name(r::NodalRec)
    bname = r |> basis |> basisname
    nname = r |> basis |> nodesname
    np = r |> basis |> nnodes
    return bname * " with " * string(np) * " " * nname * " nodes"
end

#==========================================================================================#
#                                           VCJH                                           #

struct VCJHrec{RT} <: AbstractReconstruction
    p::Int  # Order of the reconstruction
    η::RT
    name::String
end

function VCJHrec(p::Integer, η::Real)
    return VCJHrec(p, η, "VCJH (η=$η, p=$p)")
end

function VCJHrec{RT}(p::Integer, η::Symbol) where {RT}
    name = "VCJH" * string(η) * " (p=$p)"
    if η == :DGSEM_GL
        η = zero(RT)
    elseif η == :SD
        η = convert(RT, (p - 1) / p)
    elseif η == :Huynh
        η = convert(RT, p / (p - 1))
    else
        throw(ArgumentError("Reconstruction of type $η is not implemented."))
    end
    return VCJHrec(p, η, name)
end

function reconstruction(r::VCJHrec{RT}, ξ::AbstractVector) where {RT}
    (; p, η) = r
    Lkm = SpecialPolynomials.basis(Legendre, p - 2)
    Lk = SpecialPolynomials.basis(Legendre, p - 1)
    Lkp = SpecialPolynomials.basis(Legendre, p)
    gl = (-one(RT))^(p - 1) * (Lk - (η * Lkm + Lkp) / (one(RT) + η)) / 2
    gr = (Lk + (η * Lkm + Lkp) / (one(RT) + η)) / 2
    return (
        Polynomials.derivative(gl).(ξ),
        Polynomials.derivative(gr).(ξ),
    )
end

function basis(::VCJHrec)
    return nothing
end

function reconstruction_name(r::VCJHrec)
    return r.name
end

#==========================================================================================#
#                                          DGSEM                                           #

struct DGSEMrec{B<:AbstractNodalBasis} <: AbstractReconstruction
    basis::B
end

function reconstruction(r::DGSEMrec, _)
    (; basis) = r
    rt = eltype(basis.ξ)
    n = nnodes(basis)
    P = interp_matrix([-one(rt), one(rt)], basis)
    return (
        -P[1, 1:n] ./ basis.ω,
        +P[2, 1:n] ./ basis.ω,
    )
end

function basis(r::DGSEMrec)
    return r.basis
end

function reconstruction_name(r::DGSEMrec)
    bname = r |> basis |> basisname
    nname = r |> basis |> nodesname
    np = r |> basis |> nnodes
    return "DGSEM -> " * bname * " with " * string(np) * " " * nname * " nodes"
end
