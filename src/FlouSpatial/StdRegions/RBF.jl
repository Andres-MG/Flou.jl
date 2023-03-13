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
