abstract type AbstractReconstruction{RT} end

"""
    reconstruction(ξ, type)

Return a tuple containing the two reconstruction functions, (``g_l``, ``g_r``) of `type`
at the positions defined by `ξ`.
"""
function reconstruction end

function reconstruction_name(r::AbstractReconstruction)
    return r.name
end

struct VCJH{RT} <: AbstractReconstruction{RT}
    p::Int  # Order of the reconstruction
    η::RT
    name::String
end

function VCJH{RT}(p::Integer, η::Union{Symbol,Real}) where {RT}
    if η isa Symbol
        name = string(η)
        if η == :DGSEM_GL
            η = zero(RT)
        elseif η == :SD
            η = (p - 1) / p
        elseif η == :Huynh
            η = p / (p - 1)
        else
            throw(ArgumentError("Reconstruction of type $η is not implemented."))
        end
    else
        name = "VCJH ($η)"
    end
    return VCJH(p, η, name)
end

function reconstruction(ξ::AbstractVector{RT}, r::VCJH{RT}) where {RT}
    (; p, η) = r
    Lkm = SpecialPolynomials.basis(Legendre, p - 2)
    Lk = SpecialPolynomials.basis(Legendre, p - 1)
    Lkp = SpecialPolynomials.basis(Legendre, p)
    gl = (-one(RT))^p * (Lk - (η * Lkm + Lkp) / (one(RT) + η)) / 2
    gr = (Lk + (η * Lkm + Lkp) / (one(RT) + η)) / 2
    return (
        Polynomials.derivative(gl).(ξ),
        Polynomials.derivative(gr).(ξ),
    )
end
