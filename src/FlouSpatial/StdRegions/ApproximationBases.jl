# RBF interpolants
include("RBF.jl")

abstract type AbstractNodalBasis{B} end

"""
    hasboundaries(basis)

Return `true` if the nodes of `basis` include ``ξ=-1`` and ``ξ=+1``.
"""
function hasboundaries(::AbstractNodalBasis{B}) where {B}
    return B
end

"""
    nnodes(basis)

Return the number of nodes in the approximation `basis`.
"""
function nnodes(b::AbstractNodalBasis)
    return length(b.ξ)
end

"""
    basisname(basis)

Return a string representing the name of `basis`.
"""
function basisname(b::AbstractNodalBasis)
    return b.bname
end

"""
    nodesname(basis)

Return a string representing the name of the nodes that define `basis`.
"""
function nodesname(b::AbstractNodalBasis)
    return b.nname
end

#==========================================================================================#
#                                         Lagrange                                         #

struct LagrangeBasis{B,RT,L} <: AbstractNodalBasis{B}
    ξ::Vector{RT}
    ω::Vector{RT}
    basis::L
    bname::String
    nname::String
end

function LagrangeBasis(nodetype, nnodes, floattype=Float64)
    ξ, bounds, name = _nodes_from_name(nnodes, nodetype)
    basis = _approximation_basis(ξ)
    ω = _weights_from_basis(basis, floattype)
    return LagrangeBasis{
        bounds,
        floattype,
        typeof(basis),
    }(
        floattype.(ξ),
        ω,
        basis,
        "Lagrange",
        name,
    )
end

function interp_matrix(x::AbstractVector{RT}, b::LagrangeBasis) where {RT}
    return [f(xi) for xi in x, f in b.basis]
end

function derivative_matrix(x::AbstractVector{RT}, b::LagrangeBasis) where {RT}
    return [Polynomials.derivative(f)(xi) for xi in x, f in b.basis]
end

#==========================================================================================#
#                                      RBF-Polynomial                                      #

struct RBFpolyBasis{B,RT,R} <: AbstractNodalBasis{B}
    ξ::Vector{RT}
    ω::Vector{RT}
    Δ::Int
    C::Matrix{RT}
    basis::R
    bname::String
    nname::String
end

function RBFpolyBasis(nodetype, nnodes, Δ, k, floattype=Float64)
    ξ, bounds, name = _nodes_from_name(nnodes, nodetype)
    basis = _approximation_basis(ξ, nnodes - Δ, k)
    C = _modal2nodal(basis, ξ)
    ω = _weights_from_basis(basis, floattype)
    return RBFpolyBasis{
        bounds,
        floattype,
        typeof(basis),
    }(
        floattype.(ξ),
        C' * ω,
        Δ,
        C,
        basis,
        "RBF-Polynomial (Δ=$Δ, k=$k)",
        name,
    )
end

function interp_matrix(x::AbstractVector{RT}, b::RBFpolyBasis) where {RT}
    return [f(xi) for xi in x, f in b.basis] * b.C
end

function derivative_matrix(x::AbstractVector{RT}, b::RBFpolyBasis) where {RT}
    D = [Polynomials.derivative(f)(xi) for xi in x, f in b.basis]
    return D * b.C
end

function _nodes_from_name(n, name)
    if name == :Gauss || name == :GL
        ξ, _ = gausslegendre(n)
        str = "Gauss"
        bounds = false
    elseif name == :GaussLobatto || name == :GLL
        ξ, _ = gausslobatto(n)
        str = "Gauss-Lobatto"
        bounds = true
    elseif name == :ChebyshevGauss || name == :CGL
        ξ, _ = gausschebyshev(n)
        str = "Chebyshev-Gauss"
        bounds = false
    else
        throw(ArgumentError("Nodes of type $(name) cannot be used in Lagrange bases."))
    end
    return ξ, bounds, str
end

function _approximation_basis(ξ)
    rt = eltype(ξ)
    p = length(ξ)
    poly = Polynomial{rt,:x}[]
    y = zeros(rt, p)
    for i in 1:p
        y[i] = one(rt)
        push!(poly, Polynomials.fit(ξ, y))
        y[i] = zero(rt)
    end
    return poly
end

function _approximation_basis(ξ, p, k)
    rt = eltype(ξ)
    rbf = [Phs(k, x) for x in ξ]
    poly = Polynomial{rt,:x}[]
    for i in 1:p
        coeffs = zeros(rt, i)
        coeffs[end] = one(rt)
        push!(poly, Polynomial(coeffs))
    end
    return [rbf..., poly...]
end

function _weights_from_basis(basis, rt)
    ω = Vector{rt}(undef, length(basis))
    for (i, f) in enumerate(basis)
        int = Polynomials.integrate(f)
        ω[i] = int(one(rt)) - int(-one(rt))
    end
    return ω
end

function _modal2nodal(basis, ξ)
    rt = eltype(ξ)
    nt = length(basis)
    n = length(ξ)

    M = zeros(rt, nt, nt)
    for (i, x) in enumerate(ξ)
        for (j, f) in enumerate(basis)
            M[i, j] = f(x)
        end
    end
    for i in (n + 1):nt
        M[i, :] = M[:, i]
    end

    return M \ diagm(nt, n, ones(rt, n))
end
