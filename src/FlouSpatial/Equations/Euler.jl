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

function rotate2face(Qf, frame, ::EulerEquation{1})
    return SVector(Qf[1], Qf[2] * frame.n[1], Qf[3])
end

function rotate2phys(Qrot, frame, ::EulerEquation{1})
    return SVector(Qrot[1], Qrot[2] * frame.n[1], Qrot[3])
end

function rotate2face(Qf, frame, ::EulerEquation{2})
    (; n, t) = frame
    return SVector(
        Qf[1],
        Qf[2] * n[1] + Qf[3] * n[2],
        Qf[2] * t[1] + Qf[3] * t[2],
        Qf[4],
    )
end

function rotate2phys(Qrot, frame, ::EulerEquation{2})
    (; n, t) = frame
    return SVector(
        Qrot[1],
        Qrot[2] * n[1] + Qrot[3] * t[1],
        Qrot[2] * n[2] + Qrot[3] * t[2],
        Qrot[4],
    )
end

function rotate2face(Qf, frame, ::EulerEquation{3})
    (; n, t, b) = frame
    return SVector(
        Qf[1],
        Qf[2] * n[1] + Qf[3] * n[2] + Qf[4] * n[3],
        Qf[2] * t[1] + Qf[3] * t[2] + Qf[4] * t[3],
        Qf[2] * b[1] + Qf[3] * b[2] + Qf[4] * b[3],
        Qf[5],
    )
end

function rotate2phys(Qrot, frame, ::EulerEquation{3})
    (; n, t, b) = frame
    return SVector(
        Qrot[1],
        Qrot[2] * n[1] + Qrot[3] * t[1] + Qrot[4] * b[1],
        Qrot[2] * n[2] + Qrot[3] * t[2] + Qrot[4] * b[2],
        Qrot[2] * n[3] + Qrot[3] * t[3] + Qrot[4] * b[3],
        Qrot[5],
    )
end

#==========================================================================================#
#                                   Boundary Conditions                                    #

struct EulerInflowBC{RT,NV} <: AbstractBC
    Qext::SVector{NV,RT}
    function EulerInflowBC(Qext)
        nvar = length(Qext)
        3 <= nvar <= 5 || throw(ArgumentError("`Qext` must have a length of 3, 4 or 5."))
        return new{eltype(Qext),nvar}(SVector{nvar}(Qext))
    end
end

function (bc::EulerInflowBC)(_, _, _, _, ::EulerEquation)
    return bc.Qext
end

struct EulerOutflowBC <: AbstractBC end

function (::EulerOutflowBC)(Qin, _, _, _, eq::EulerEquation)
    return SVector{nvariables(eq)}(Qin)
end

struct EulerSlipBC <: AbstractBC end

function (::EulerSlipBC)(Qin, _, frame, _, eq::EulerEquation)
    Qn = rotate2face(Qin, frame, eq) |> MVector
    Qn[2] = -Qn[2]
    return rotate2phys(SVector(Qn), frame, eq)
end

#==========================================================================================#
#                                     Numerical fluxes                                     #

function numericalflux(Ql, Qr, _, eq::EulerEquation{1}, ::StdAverage)
    _, ρul, ρel = Ql
    _, ρur, ρer = Qr
    _, ul, pl = vars_cons2prim(Ql, eq)
    _, ur, pr = vars_cons2prim(Qr, eq)
    return SVector(
        (ρul + ρur) / 2,
        (ρul * ul + pl + ρur * ur + pr) / 2,
        ((ρel + pl) * ul + (ρer + pr) * ur) / 2,
    )
end

function numericalflux(Ql ,Qr, _, eq::EulerEquation{2}, ::StdAverage)
    _, ρul, ρvl, ρel = Ql
    _, ρur, ρvr, ρer = Qr
    _, ul, _, pl = vars_cons2prim(Ql, eq)
    _, ur, _, pr = vars_cons2prim(Qr, eq)
    return SVector(
        (ρul + ρur) / 2,
        (ρul * ul + pl + ρur * ur + pr) / 2,
        (ρvl * ul + ρvr * ur) / 2,
        ((ρel + pl) * ul + (ρer + pr) * ur) / 2,
    )
end

function numericalflux(Ql, Qr, _, eq::EulerEquation{3}, ::StdAverage)
    _, ρul, ρvl, ρwl, ρel = Ql
    _, ρur, ρvr, ρwr, ρer = Qr
    _, ul, _, _, pl = vars_cons2prim(Ql, eq)
    _, ur, _, _, pr = vars_cons2prim(Qr, eq)
    return SVector(
        (ρul + ρur) / 2,
        (ρul * ul + pl + ρur * ur + pr) / 2,
        (ρvl * ul + ρvr * ur) / 2,
        (ρwl * ul + ρwr * ur) / 2,
        ((ρel + pl) * ul + (ρer + pr) * ur) / 2,
    )
end

function numericalflux(
    Ql,
    Qr,
    n,
    eq::EulerEquation{ND},
    nf::LxF,
) where {
    ND,
}
    # Average
    Fn = numericalflux(Ql, Qr, n, eq, nf.avg)

    # Dissipation
    if ND == 1
        ρl, ul, pl = vars_cons2prim(Ql, eq)
        ρr, ur, pr = vars_cons2prim(Qr, eq)
    elseif ND == 2
        ρl, ul, _, pl = vars_cons2prim(Ql, eq)
        ρr, ur, _, pr = vars_cons2prim(Qr, eq)
    else # ND == 3
        ρl, ul, _, _, pl = vars_cons2prim(Ql, eq)
        ρr, ur, _, _, pr = vars_cons2prim(Qr, eq)
    end
    al = soundvelocity(ρl, pl, eq)
    ar = soundvelocity(ρr, pr, eq)
    λ = max(abs(ul) + al, abs(ur) + ar)
    return SVector(Fn + λ * (Ql - Qr) / 2 * nf.intensity)
end

struct ChandrasekharAverage <: AbstractNumericalFlux end

function numericalflux(
    Ql,
    Qr,
    _,
    eq::EulerEquation{ND},
    ::ChandrasekharAverage,
) where {
    ND,
}
    # Variables
    if ND == 1
        ρl, ul, pl = vars_cons2prim(Ql, eq)
        ρr, ur, pr = vars_cons2prim(Qr, eq)
        u = (ul + ur) / 2
    elseif ND == 2
        ρl, ul, vl, pl = vars_cons2prim(Ql, eq)
        ρr, ur, vr, pr = vars_cons2prim(Qr, eq)
        u, v = (ul + ur) / 2, (vl + vr) / 2
    else # ND == 3
        ρl, ul, vl, wl, pl = vars_cons2prim(Ql, eq)
        ρr, ur, vr, wr, pr = vars_cons2prim(Qr, eq)
        u, v, w = (ul + ur) / 2, (vl + vr) / 2, (wl + wr) / 2
    end

    # Averages
    βl, βr = ρl / 2pl, ρr / 2pr
    ρ = logarithmic_mean(ρl, ρr)
    p = (ρl + ρr) / (2 * (βl + βr))
    β = logarithmic_mean(βl, βr)

    # Fluxes
    if ND == 1
        h = 1 / (2β * (eq.γ - 1)) - (ul^2 + ur^2) / 4 + p/ρ + u^2
        return SVector(
            ρ * u,
            ρ * u^2 + p,
            ρ * u * h,
        )
    elseif ND == 2
        h = 1 / (2β * (eq.γ - 1)) - (ul^2 + vl^2 + ur^2 + vr^2) / 4 + p/ρ + u^2 + v^2
        return SVector(
            ρ * u,
            ρ * u^2 + p,
            ρ * u * v,
            ρ * u * h,
        )
    else # ND == 3
        h = 1 / (2β * (eq.γ - 1)) - (ul^2 + vl^2 + wl^2 + ur^2 + vr^2 + wl^2) / 4 +
            p/ρ + u^2 + v^2 + w^2
        return SVector(
            ρ * u,
            ρ * u^2 + p,
            ρ * u * v,
            ρ * u * w,
            ρ * u * h,
        )
    end
end

struct ScalarDissipation{T,RT} <: AbstractNumericalFlux
    avg::T
    intensity::RT
end

function numericalflux(
    Ql,
    Qr,
    n,
    eq::EulerEquation{ND},
    nf::ScalarDissipation,
) where {
    ND,
}
    # Variables
    if ND == 1
        ρl, ul, pl = vars_cons2prim(Ql, eq)
        ρul = Ql[2]
        ρr, ur, pr = vars_cons2prim(Qr, eq)
        ρur = Qr[2]
        u = (ul + ur) / 2
    elseif ND == 2
        ρl, ul, vl, pl = vars_cons2prim(Ql, eq)
        ρul, ρvl = Ql[2], Ql[3]
        ρr, ur, vr, pr = vars_cons2prim(Qr, eq)
        ρur, ρvr = Qr[2], Qr[3]
        u, v = (ul + ur) / 2, (vl + vr) / 2
    else # ND == 3
        ρl, ul, vl, wl, pl = vars_cons2prim(Ql, eq)
        ρul, ρvl, ρwl = Qr[2], Qr[3], Qr[4]
        ρr, ur, vr, wr, pr = vars_cons2prim(Qr, eq)
        ρur, ρvr, ρwr = Qr[2], Qr[3], Qr[4]
        u, v, w = (ul + ur) / 2, (vl + vr) / 2, (wl + wr) / 2
    end

    # Averages
    ρ = (ρl + ρr) / 2
    βl, βr = ρl / 2pl, ρr / 2pr
    β = logarithmic_mean(βl, βr)
    al = soundvelocity(ρl, pl, eq)
    ar = soundvelocity(ρr, pr, eq)

    # Averaging term
    Fn = numericalflux(Ql, Qr, n, eq, nf.avg)

    # Dissipative term
    λ = max(abs(ul) + al, abs(ur) + ar)
    𝓓 = if ND == 1
        SVector(
            ρr - ρl,
            ρur - ρul,
            (1 / β / (eq.γ - 1) + ul * ur) * (ρr - ρl) / 2 +
                ρ * (u * (ur - ul) + (1/βr - 1/βl) / 2(eq.γ - 1)),
        )
    elseif ND == 2
        SVector(
            ρr - ρl,
            ρur - ρul,
            ρvr - ρvl,
            (1 / β / (eq.γ - 1) + ul * ur + vl * vr) * (ρr - ρl) / 2 +
                ρ * (u * (ur - ul) + v * (vr - vl) + (1/βr - 1/βl) / 2(eq.γ - 1)),
        )
    else # ND == 3
        SVector(
            ρr - ρl,
            ρur - ρul,
            ρvr - ρvl,
            ρwr - ρwl,
            (1 / β / (eq.γ - 1) + ul * ur + vl * vr + wl * wr) * (ρr - ρl) / 2 +
                ρ * (u * (ur - ul) + v * (vr - vl) + w * (wr - wl) + (1/βr - 1/βl) / 2(eq.γ - 1)),
        )
    end
    return SVector(Fn - λ / 2 * 𝓓 * nf.intensity)
end

struct MatrixDissipation{T,RT} <: AbstractNumericalFlux
    avg::T
    intensity::RT
end

function numericalflux(
    Ql,
    Qr,
    n,
    eq::EulerEquation{ND},
    nf::MatrixDissipation,
) where {
    ND,
}
    # Variables
    if ND == 1
        ρl, ul, pl = vars_cons2prim(Ql, eq)
        ρr, ur, pr = vars_cons2prim(Qr, eq)
        u = (ul + ur) / 2
        v2 = 2 * u^2 - (ul^2 + ur^2) / 2
    elseif ND == 2
        ρl, ul, vl, pl = vars_cons2prim(Ql, eq)
        ρr, ur, vr, pr = vars_cons2prim(Qr, eq)
        u, v = (ul + ur) / 2, (vl + vr) / 2
        v2 = 2 * (u^2 + v^2) - (ul^2 + vl^2 + ur^2 + vr^2) / 2
    else # ND == 3
        ρl, ul, vl, wl, pl = vars_cons2prim(Ql, eq)
        ρr, ur, vr, wr, pr = vars_cons2prim(Qr, eq)
        u, v, w = (ul + ur) / 2, (vl + vr) / 2, (wl + wr) / 2
        v2 = 2 * (u^2 + v^2 + w^2) - (ul^2 + vl^2 + wl^2 + ur^2 + vr^2 + wr^2) / 2
    end

    # Averages
    βl, βr = ρl / 2pl, ρr / 2pr
    ρ = logarithmic_mean(ρl, ρr)
    p = (ρl + ρr) / (2 * (βl + βr))
    β = logarithmic_mean(βl, βr)
    a = soundvelocity(ρ, p, eq)
    h = eq.γ / 2β / (eq.γ - 1) + v2 / 2

    # Averaging term
    Fn = numericalflux(Ql, Qr, n, eq, nf.avg)

    # Dissipative term
    rt = eltype(Fn)
    Wl = vars_cons2entropy(Ql, eq)
    Wr = vars_cons2entropy(Qr, eq)

    if ND == 1
        Λ = SDiagonal{3}((u - a, u, u + a) .|> abs)
        T = SDiagonal{3}((ρ / 2eq.γ, (eq.γ - 1) * ρ / eq.γ, ρ / 2eq.γ))
        R = SMatrix{3,3}(
            one(rt),  u - a, h - u * a,
            one(rt),  u,     v2 / 2,
            one(rt),  u + a, h + u * a,
        )
    elseif ND == 2
        Λ = SDiagonal{4}((u - a, u, u, u + a) .|> abs)
        T = SDiagonal{4}((ρ / 2eq.γ, (eq.γ - 1) * ρ / eq.γ, p, ρ / 2eq.γ))
        R = SMatrix{4,4}(
            one(rt),  u - a,    v,       h - u * a,
            one(rt),  u,        v,       v2 / 2,
            zero(rt), zero(rt), one(rt), v,
            one(rt),  u + a,    v,       h + u * a,
        )
    else # ND == 3
        Λ = SDiagonal{5}((u - a, u, u, u, u + a) .|> abs)
        T = SDiagonal{5}((ρ / 2eq.γ, (eq.γ - 1) * ρ / eq.γ, p, p, ρ / 2eq.γ))
        R = SMatrix{5,5}(
            one(rt),  u - a,    v,        w,        h - u * a,
            one(rt),  u,        v,        w,        v2 / 2,
            zero(rt), zero(rt), one(rt),  zero(rt), v,
            zero(rt), zero(rt), zero(rt), one(rt),  w,
            one(rt),  u + a,    v,        w,        h + u * a,
        )
    end
    return SVector(Fn + R * Λ * T * R' * (Wl - Wr) / 2 * nf.intensity)
end

#==========================================================================================#
#                                     Two-point fluxes                                     #

function twopointflux(Q1, Q2, Ja1, Ja2, eq::EulerEquation{1}, ::StdAverage)
    _, ρu1, ρe1 = Q1
    _, ρu2, ρe2 = Q2
    _, u1, p1 = vars_cons2prim(Q1, eq)
    _, u2, p2 = vars_cons2prim(Q2, eq)

    n = (Ja1[1] + Ja2[1]) / 2
    f1 = (ρu1 + ρu2) / 2
    f2 = (ρu1 * u1 + p1 + ρu2 * u2 + p2) / 2
    f3 = ((ρe1 + p1) * u1 + (ρe2 + p2) * u2) / 2
    return SVector(
        f1 * n,
        f2 * n,
        f3 * n,
    )
end

function twopointflux(Q1, Q2, Ja1, Ja2, eq::EulerEquation{2}, ::StdAverage)
    _, ρu1, ρv1, ρe1 = Q1
    _, ρu2, ρv2, ρe2 = Q2
    _, u1, v1, p1 = vars_cons2prim(Q1, eq)
    _, u2, v2, p2 = vars_cons2prim(Q2, eq)

    n = SVector((Ja1 .+ Ja2) ./ 2)
    f1 = SVector(
        (ρu1 + ρu2) / 2,
        (ρv1 + ρv2) / 2,
    )
    f2 = SVector(
        (ρu1 * u1 + p1 + ρu2 * u2 + p2) / 2,
        (ρu1 * v1 + ρu2 * v2) / 2,
    )
    f3 = SVector(
        (ρv1 * u1 + ρv2 * u2) / 2,
        (ρv1 * v1 + p1 + ρv2 * v2 + p2) / 2,
    )
    f4 = SVector(
        ((ρe1 + p1) * u1 + (ρe2 + p2) * u2) / 2,
        ((ρe1 + p1) * v1 + (ρe2 + p2) * v2) / 2,
    )
    return SVector(
        f1[1] * n[1] + f1[2] * n[2],
        f2[1] * n[1] + f2[2] * n[2],
        f3[1] * n[1] + f3[2] * n[2],
        f4[1] * n[1] + f4[2] * n[2],
    )
end

function twopointflux(Q1, Q2, Ja1, Ja2, eq::EulerEquation{3}, ::StdAverage)
    _, ρu1, ρv1, ρw1, ρe1 = Q1
    _, ρu2, ρv2, ρw2, ρe2 = Q2
    _, u1, v1, w1, p1 = vars_cons2prim(Q1, eq)
    _, u2, v2, w2, p2 = vars_cons2prim(Q2, eq)

    n = SVector((Ja1 .+ Ja2) ./ 2)
    f1 = SVector(
        (ρu1 + ρu2) / 2,
        (ρv1 + ρv2) / 2,
        (ρw1 + ρw2) / 2,
    )
    f2 = SVector(
        (ρu1 * u1 + p1 + ρu2 * u2 + p2) / 2,
        (ρu1 * v1 + ρu2 * v2) / 2,
        (ρu1 * w1 + ρu2 * w2) / 2,
    )
    f3 = SVector(
        (ρv1 * u1 + ρv2 * u2) / 2,
        (ρv1 * v1 + p1 + ρv2 * v2 + p2) / 2,
        (ρv1 * w1 + ρv2 * w2) / 2,
    )
    f4 = SVector(
        (ρw1 * u1 + ρw2 * u2) / 2,
        (ρw1 * v1 + ρw2 * v2) / 2,
        (ρw1 * w1 + p1 + ρw2 * w2 + p2) / 2,
    )
    f5 = SVector(
        ((ρe1 + p1) * u1 + (ρe2 + p2) * u2) / 2,
        ((ρe1 + p1) * v1 + (ρe2 + p2) * v2) / 2,
        ((ρe1 + p1) * w1 + (ρe2 + p2) * w2) / 2,
    )
    return SVector(
        f1[1] * n[1] + f1[2] * n[2] + f1[3] * n[3],
        f2[1] * n[1] + f2[2] * n[2] + f2[3] * n[3],
        f3[1] * n[1] + f3[2] * n[2] + f3[3] * n[3],
        f4[1] * n[1] + f4[2] * n[2] + f4[3] * n[3],
        f5[1] * n[1] + f5[2] * n[2] + f5[3] * n[3],
    )
end

function twopointflux(
    Q1,
    Q2,
    Ja1,
    Ja2,
    eq::EulerEquation{ND},
    ::ChandrasekharAverage,
) where {
    ND,
}
    # Variables
    if ND == 1
        ρ1, u1, p1 = vars_cons2prim(Q1, eq)
        ρ2, u2, p2 = vars_cons2prim(Q2, eq)
        u = (u1 + u2) / 2
    elseif ND == 2
        ρ1, u1, v1, p1 = vars_cons2prim(Q1, eq)
        ρ2, u2, v2, p2 = vars_cons2prim(Q2, eq)
        u, v = (u1 + u2) / 2, (v1 + v2) / 2
    else # ND == 3
        ρ1, u1, v1, w1, p1 = vars_cons2prim(Q1, eq)
        ρ2, u2, v2, w2, p2 = vars_cons2prim(Q2, eq)
        u, v, w = (u1 + u2) / 2, (v1 + v2) / 2, (w1 + w2) / 2
    end

    # Averages
    β1, β2 = ρ1 / 2p1, ρ2 / 2p2
    ρ = logarithmic_mean(ρ1, ρ2)
    p = (ρ1 + ρ2) / (2 * (β1 + β2))
    β = logarithmic_mean(β1, β2)

    # Fluxes
    if ND == 1
        h = 1 / (2β * (eq.γ - 1)) - (u1^2 + u2^2) / 4 + p/ρ + u^2
        n = (Ja1[1] + Ja2[1]) / 2
        return SVector(
            (ρ * u) * n,
            (ρ * u^2 + p) * n,
            (ρ * u * h) * n,
        )
    elseif ND == 2
        h = 1 / (2β * (eq.γ - 1)) - (u1^2 + v1^2 + u2^2 + v2^2) / 4 + p/ρ + u^2 + v^2
        n = SVector((Ja1 .+ Ja2) ./ 2)
        return SVector(
            (ρ * u) * n[1] + (ρ * v) * n[2],
            (ρ * u^2 + p) * n[1] + (ρ * u * v) * n[2],
            (ρ * u * v) * n[1] + (ρ * v^2 + p) * n[2],
            (ρ * u * h) * n[1] + (ρ * v * h) * n[2],
        )
    else # ND == 3
        h = 1 / (2β * (eq.γ - 1)) - (u1^2 + v1^2 + w1^2 + u2^2 + v2^2 + w2^2) / 4 +
            p/ρ + u^2 + v^2 + w^2
        n = SVector((Ja1 .+ Ja2) ./ 2)
        return SVector(
            (ρ * u) * n[1] + (ρ * v) * n[2] + (ρ * w) * n[3],
            (ρ * u^2 + p) * n[1] + (ρ * u * v) * n[2] + (ρ * u * w) * n[3],
            (ρ * u * v) * n[1] + (ρ * v^2 + p) * n[2] + (ρ * v * w) * n[3],
            (ρ * u * w) * n[1] + (ρ * v * w) * n[2] + (ρ * w^2 + p) * n[3],
            (ρ * u * h) * n[1] + (ρ * v * h) * n[2] + (ρ * w * h) * n[3],
        )
    end
    return nothing
end

#==========================================================================================#
#                                    Monitors                                              #

function FlouCommon.list_monitors(::MultielementDisc, ::EulerEquation)
    return (:kinetic_energy, :entropy,)
end

function FlouCommon.get_monitor(
    disc::MultielementDisc,
    equation::EulerEquation,
    name::Symbol,
    _,
)
    if name == :energy
        return kinetic_energy_monitor(disc, equation)
    elseif name == :entropy
        return entropy_monitor(disc, equation)
    else
        error("Unknown monitor '$(name)'.")
    end
end

function kinetic_energy_monitor(::MultielementDisc, ::EulerEquation)
    return (_Q, disc, equation) -> begin
        Q = GlobalStateVector(_Q, disc.dofhandler)
        s = zero(datatype(Q))
        @flouthreads for ie in eachelement(disc)
            Qe = Q.elements[ie]
            svec = disc.std.cache.scalar[Threads.threadid()][1].vars[1]
            @inbounds for (i, Qi) in enumerate(Qe.dofs)
                svec[i] = kinetic_energy(Qi, equation)
            end
            s += integrate(svec, disc.geometry.elements[ie])
        end
        return s
    end
end

function entropy_monitor(::MultielementDisc, ::EulerEquation)
    return (_Q, disc, equation) -> begin
        Q = GlobalStateVector(_Q, disc.dofhandler)
        s = zero(datatype(Q))
        @flouthreads for ie in eachelement(disc)
            Qe = Q.elements[ie]
            svec = disc.std.cache.scalar[Threads.threadid()][1].vars[1]
            @inbounds for (i, Qi) in enumerate(Qe.dofs)
                svec[i] = math_entropy(Qi, equation)
            end
            s += integrate(svec, disc.geometry.elements[ie])
        end
        return s
    end
end
