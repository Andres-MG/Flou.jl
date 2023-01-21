function vcjh_reconstruction(ξ, ηk)
    rt = eltype(ξ)
    k = length(ξ) - 1
    Lk = SpecialPolynomials.basis(Legendre, k)
    Lkp = SpecialPolynomials.basis(Legendre, k + 1)
    Lkm = SpecialPolynomials.basis(Legendre, k - 1)
    gl = (-one(rt))^k * (Lk - (ηk * Lkm + Lkp) / (one(rt) + ηk)) / 2
    gr = (Lk + (ηk * Lkm + Lkp) / (one(rt) + ηk)) / 2
    return (
        -Polynomials.derivative(gl).(ξ),
        Polynomials.derivative(gr).(ξ),
    )
end

function dgsem_reconstruction(ξ)
    ηk = zero(eltype(ξ))
    return vcjh_reconstruction(ξ, ηk)
end

function sd_reconstruction(ξ)
    k = length(ξ) - 1
    ηk = k / (k + 1)
    return vcjh_reconstruction(ξ, ηk)
end

function huynh_reconstruction(ξ)
    k = length(ξ) - 1
    ηk = (k + 1) / k
    return vcjh_reconstruction(ξ, ηk)
end