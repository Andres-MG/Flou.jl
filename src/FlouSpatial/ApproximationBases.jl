module ApproximationBases

using FastGaussQuadrature: FastGaussQuadrature, gausslegendre, gausschebyshev, gausslobatto
using Polynomials: Polynomials, Polynomial, derivative
using LinearAlgebra: diagm

export AbstractNodeDistribution
export GaussNodes, GaussLobattoNodes, GaussChebyshevNodes
export GL, GCL, GLL

export npoly, ndelta, nnodes
export interp_matrix, derivative_matrix

# RBF interpolants
include("RBF.jl")

struct Basis{R,P,RT} <: AbstractVector{Union{P,R}}
    p::Int
    Δ::Int
    rbf::Vector{R}
    poly::Vector{P}
    C::Matrix{RT}
end

function Basis(ξ::AbstractVector, p::Int, k=3)
    Δ = length(ξ) - p
    rbf, poly = approximation_basis(ξ, p, k)
    C = modal2nodal([rbf..., poly...], ξ)
    return Basis(p, Δ, rbf, poly, C)
end

function Base.size(b::Basis)
    return (2b.p + b.Δ,)
end

Base.@propagate_inbounds function Base.getindex(b::Basis, i)
    nt = b.p + b.Δ
    return if i > nt
        b.poly[i - nt]
    else
        b.rbf[i]
    end
end

Base.@propagate_inbounds function Base.setindex(b::Basis, i, f)
    nt = b.p + b.Δ
    return if i > nt
        b.poly[i - nt] = f
    else
        b.rbf[i] = f
    end
end

function approximation_basis(ξ, p, k)
    rt = eltype(ξ)
    rbf = [Phs(k, x) for x in ξ]
    poly = []
    for i in 1:p
        coeffs = zeros(rt, i)
        coeffs[end] = one(rt)
        push!(
            poly,
            Polynomial(coeffs),
        )
    end
    poly = [poly...]
    return rbf, poly
end

function modal2nodal(basis, ξ)
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

abstract type AbstractNodeDistribution{RT} end

struct GaussNodes{RT,B} <: AbstractNodeDistribution{RT}
    ξ::Vector{RT}
    ω::Vector{RT}
    basis::B
end

function GaussNodes{RT}(p, Δ=0) where {RT}
    ξ, ω = gausslegendre(p + Δ)
    ξ, ω = convert.(RT, ξ), convert.(RT, ξ)
    basis = Basis(ξ, p)
    return GaussNodes(ξ, ω, basis)
end

struct GaussChebyshevNodes{RT,B} <: AbstractNodeDistribution{RT}
    ξ::Vector{RT}
    ω::Vector{RT}
    basis::B
end

function GaussChebyshevNodes{RT}(p, Δ=0) where {RT}
    ξ, ω = gausschebyshev(p + Δ)
    ξ, ω = convert.(RT, ξ), convert.(RT, ξ)
    basis = Basis(ξ, p)
    return GaussChebyshevNodes(ξ, ω, basis)
end

struct GaussLobattoNodes{RT,B} <: AbstractNodeDistribution{RT}
    ξ::Vector{RT}
    ω::Vector{RT}
    basis::B
end

function GaussLobattoNodes{RT}(p, Δ=0) where {RT}
    ξ, ω = gausslobatto(p + Δ)
    ξ, ω = convert.(RT, ξ), convert.(RT, ξ)
    basis = Basis(ξ, p)
    return GaussLobattoNodes(ξ, ω, basis)
end

const GL = GaussNodes
const GCL = GaussChebyshevNodes
const GLL = GaussLobattoNodes

"""
    ndelta(nodes)

Return the number of additional nodes in the `nodes` distribution to introduce RBF
stabilization.
"""
function ndelta(nd::AbstractNodeDistribution)
    return nd.basis.Δ
end


"""
    npoly(nodes)

Return the number of polynomials in the `nodes` distribution (Equivalent to `nnodes(nodes)`
when RBFs are not used).
"""
function npoly(nd::AbstractNodeDistribution)
    return nd.basis.p
end

"""
    nnodes(nodes)

Return the number of nodes in the `nodes` distribution.
"""
function nnodes(nd::AbstractNodeDistribution)
    return npoly(nd) + ndelta(nd)
end

function interp_matrix(x, nd::AbstractNodeDistribution{RT}) where {RT}
    nb = length(nd.basis)
    nx = length(x)

    P = Matrix{RT}(undef, nx, nb)
    for (i, f) in enumerate(nd.basis)
        P[:, i] = f.(x)
    end

    return P * nd.basis.C
end

function derivative_matrix(x, nd::AbstractNodeDistribution{RT}) where {RT}
    nb = length(nd.basis)
    nx = length(x)

    D = Matrix{RT}(undef, nx, nb)
    for (i, f) in enumerate(nd.basis)
        D[:, i] = derivative(f).(x)
    end

    return D * nd.basis.C
end

end # ApproximationBases