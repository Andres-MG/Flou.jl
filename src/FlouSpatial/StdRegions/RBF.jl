#==========================================================================================#
#                                   Polyharmonic splines                                   #

struct Phs{RT}
    e::Int
    x0::RT
    even::Bool
end

function Phs(e, x0)
    return Phs(e, x0, iseven(e))
end

function (p::Phs)(x)
    r = abs(x - p.x0)
    k = p.e
    return if p.even
        r ^ (k - 1) * log(r ^ r)
    else
        r ^ k
    end
end

struct PhsDerivative{RT}
    p::Phs{RT}
end

function Polynomials.derivative(p::Phs)
    return PhsDerivative(p)
end

function (d::PhsDerivative)(x)
    p = d.p
    xx = x - p.x0
    s = sign(xx)
    r = s * xx
    k = p.e
    return if p.even
        s * r ^ (k - 2) * (k * log(r ^ r) + r)
    else
        s * k * r ^ (k - 1)
    end
end

struct PhsIntegral{RT}
    p::Phs{RT}
end

function Polynomials.integrate(p::Phs)
    return PhsIntegral(p)
end

function (int::PhsIntegral{RT})(x) where {RT}
    p = int.p
    xx = x - p.x0
    s = sign(xx)
    r = s * xx
    k = p.e
    return if p.even
        s * r * (k + 1) / (k + 1)^2 * ((k + 1) * log(r) - one(RT))
    else
        s * r ^ (k + 1) / (k + 1)
    end
end
