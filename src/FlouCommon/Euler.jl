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

struct EulerEquation{ND,NV,RT} <: HyperbolicEquation{ND,NV}
    γ::RT
    function EulerEquation{ND}(γ) where {ND}
        1 <= ND <= 3 || throw(ArgumentError(
            "The Euler equations are only implemented in 1D, 2D and 3D."
        ))
        return new{ND,ND + 2,typeof(γ)}(γ)
    end
end

function Base.show(io::IO, ::MIME"text/plain", eq::EulerEquation{ND}) where {ND}
    @nospecialize
    println(io, ND, "D Euler equation:")
    print(io, " γ: ", eq.γ)
    return nothing
end

function variablenames(::EulerEquation{ND}; unicode=false) where {ND}
    return if unicode
        if ND == 1
            ("ρ", "ρu", "ρe")
        elseif ND == 2
            ("ρ", "ρu", "ρv", "ρe")
        else # ND == 3
            ("ρ", "ρu", "ρv", "ρw", "ρe")
        end
    else
        if ND == 1
            ("rho", "rhou", "rhoe")
        elseif ND == 2
            ("rho", "rhou", "rhov", "rhoe")
        else # ND == 3
            ("rho", "rhou", "rhov", "rhow", "rhoe")
        end
    end
    return nothing
end

function volumeflux(Q, eq::EulerEquation{1})
    ρ, ρu, ρe = Q
    u = ρu/ρ
    p = pressure(Q, eq)
    return (
        SVector{3}(
            ρu,
            ρu * u + p,
            (ρe + p) * u,
        ),
    )
end

function volumeflux(Q, eq::EulerEquation{2})
    ρ, ρu, ρv, ρe = Q
    u, v = ρu/ρ, ρv/ρ
    p = pressure(Q, eq)
    return (
        SVector{4}(
            ρu,
            ρu * u + p,
            ρu * v,
            (ρe + p) * u,
        ),
        SVector{4}(
            ρv,
            ρv * u,
            ρv * v + p,
            (ρe + p) * v,
        ),
    )
end

function volumeflux(Q, eq::EulerEquation{3})
    ρ, ρu, ρv, ρw, ρe = Q
    u, v, w = ρu/ρ, ρv/ρ, ρw/ρ
    p = pressure(Q, eq)
    return (
        SVector{5}(
            ρu,
            ρu * u + p,
            ρu * v,
            ρu * w,
            (ρe + p) * u,
        ),
        SVector{5}(
            ρv,
            ρv * u,
            ρv * v + p,
            ρv * w,
            (ρe + p) * v,
        ),
        SVector{5}(
            ρw,
            ρw * u,
            ρw * v,
            ρw * w + p,
            (ρe + p) * w,
        ),
    )
end

function get_max_dt(Q, Δx::Real, cfl::Real, eq::EulerEquation{1})
    c = soundvelocity(Q, eq)
    ρ, ρu, _ = Q
    u = ρu / ρ
    return cfl * Δx / (abs(u) + c)
end

function get_max_dt(Q, Δx::Real, cfl::Real, eq::EulerEquation{2})
    c = soundvelocity(Q, eq)
    ρ, ρu, ρv, _ = Q
    u, v = ρu / ρ, ρv / ρ
    return cfl * Δx / (sqrt(u^2 + v^2) + c)
end

function get_max_dt(Q, Δx::Real, cfl::Real, eq::EulerEquation{3})
    c = soundvelocity(Q, eq)
    ρ, ρu, ρv, ρw, _ = Q
    u, v, w = ρu / ρ, ρv / ρ, ρw / ρ
    return cfl * Δx / (sqrt(u^2 + v^2 + w^2) + c)
end

"""
    pressure(Q, eq::EulerEquation)

Compute the pressure from the *conservative* variables `Q`.
"""
function pressure(Q, eq::EulerEquation{1})
    ρ, ρu, ρe = Q
    return (eq.γ - 1) * (ρe - ρu^2 / 2ρ)
end

function pressure(Q, eq::EulerEquation{2})
    ρ, ρu, ρv, ρe = Q
    return (eq.γ - 1) * (ρe - (ρu^2 + ρv^2) / 2ρ)
end

function pressure(Q, eq::EulerEquation{3})
    ρ, ρu, ρv, ρw, ρe = Q
    return (eq.γ - 1) * (ρe - (ρu^2 + ρv^2 + ρw^2) / 2ρ)
end

"""
    kinetic_energy(Q, eq::EulerEquation)

Compute the kinetic energy from the *conservative* variables `Q`.
"""
function kinetic_energy(Q, ::EulerEquation{1})
    ρ, ρu = view(Q, 1:2)
    return ρu^2 / 2ρ
end

function kinetic_energy(Q, ::EulerEquation{2})
    ρ, ρu, ρv = view(Q, 1:3)
    return (ρu^2 + ρv^2) / 2ρ
end

function kinetic_energy(Q, ::EulerEquation{3})
    ρ, ρu, ρv, ρw = view(Q, 1:4)
    return (ρu^2 + ρv^2 + ρw^2) / 2ρ
end

"""
    energy(Q, eq::EulerEquation)

Compute the energy, `ρe`, from the *primitive* variables `Q`.
"""
function energy(Q, eq::EulerEquation{1})
    ρ, u, p = Q
    return p / (eq.γ - 1) + ρ * u^2 / 2
end

function energy(Q, eq::EulerEquation{2})
    ρ, u, v, p = Q
    return p / (eq.γ - 1) + ρ * (u^2 + v^2) / 2
end

function energy(Q, eq::EulerEquation{3})
    ρ, u, v, w, p = Q
    return p / (eq.γ - 1) + ρ * (u^2 + v^2 + w^2) / 2
end

"""
    entropy(Q, eq::EulerEquation)

Compute the physical entropy, `σ`, from the *conservative* variables `Q`.
"""
function entropy(Q, eq::EulerEquation)
    ρ = Q[1]
    p = pressure(Q, eq)
    return log(p) - eq.γ * log(ρ)
end

"""
    math_entropy(Q, eq::EulerEquation)

Compute the mathematical entropy, `-ρσ/(γ-1)`, from the *conservative* variables `Q`.
"""
function math_entropy(Q, eq::EulerEquation)
    ρ = Q[1]
    s = entropy(Q, eq)
    return - ρ * s / (eq.γ - 1)
end

function soundvelocity_sqr(Q, eq::EulerEquation)
    ρ = Q[1]
    p = pressure(Q, eq)
    return soundvelocity_sqr(ρ, p, eq)
end

function soundvelocity_sqr(ρ, p, eq::EulerEquation)
    return eq.γ * p / ρ
end

function soundvelocity(Q, eq::EulerEquation)
    return soundvelocity_sqr(Q, eq) |> sqrt
end

function soundvelocity(ρ, p, eq::EulerEquation)
    return soundvelocity_sqr(ρ, p, eq) |> sqrt
end

function vars_cons2prim(Q, eq::EulerEquation{1})
    ρ, ρu, _ = Q
    p = pressure(Q, eq)
    return SVector(ρ, ρu/ρ, p)
end

function vars_cons2prim(Q, eq::EulerEquation{2})
    ρ, ρu, ρv, _ = Q
    p = pressure(Q, eq)
    return SVector(ρ, ρu/ρ, ρv/ρ, p)
end

function vars_cons2prim(Q, eq::EulerEquation{3})
    ρ, ρu, ρv, ρw, _ = Q
    p = pressure(Q, eq)
    return SVector(ρ, ρu/ρ, ρv/ρ, ρw/ρ, p)
end

function vars_prim2cons(Q, eq::EulerEquation{1})
    ρ, u, _ = Q
    ρe = energy(Q, eq)
    return SVector(ρ, ρ * u, ρe)
end

function vars_prim2cons(Q, eq::EulerEquation{2})
    ρ, u, v, _ = Q
    ρe = energy(Q, eq)
    return SVector(ρ, ρ * u, ρ * v, ρe)
end

function vars_prim2cons(Q, eq::EulerEquation{3})
    ρ, u, v, w, _ = Q
    ρe = energy(Q, eq)
    return SVector(ρ, ρ * u, ρ * v, ρ * w, ρe)
end

function vars_cons2entropy(Q, eq::EulerEquation{1})
    ρ, ρu, _ = Q
    p = pressure(Q, eq)
    s = entropy(Q, eq)
    return SVector(
        (eq.γ - s) / (eq.γ - 1) - ρu^2 / ρ / 2p,
        ρu / p,
        -ρ / p,
    )
end

function vars_cons2entropy(Q, eq::EulerEquation{2})
    ρ, ρu, ρv, _ = Q
    p = pressure(Q, eq)
    s = entropy(Q, eq)
    return SVector(
        (eq.γ - s) / (eq.γ - 1) - (ρu^2 + ρv^2) / ρ / 2p,
        ρu / p,
        ρv / p,
        -ρ / p,
    )
end

function vars_cons2entropy(Q, eq::EulerEquation{3})
    ρ, ρu, ρv, ρw, _ = Q
    p = pressure(Q, eq)
    s = entropy(Q, eq)
    return SVector(
        (eq.γ - s) / (eq.γ - 1) - (ρu^2 + ρv^2 + ρw^2) / ρ / 2p,
        ρu / p,
        ρv / p,
        ρw / p,
        -ρ / p,
    )
end

function vars_entropy2prim(W, eq::EulerEquation{1})
    u = -W[2] / W[3]
    s = eq.γ - (eq.γ - 1) * (W[1] - W[3] * u^2 / 2)
    p = ((-W[3])^eq.γ * exp(s))^(1 / (1 - eq.γ))
    ρ = -p * W[3]
    return SVector(ρ, u, p)
end

function vars_entropy2prim(W, eq::EulerEquation{2})
    u = -W[2] / W[4]
    v = -W[3] / W[4]
    s = eq.γ - (eq.γ - 1) * (W[1] - W[4] * (u^2 + v^2) / 2)
    p = ((-W[4])^eq.γ * exp(s))^(1 / (1 - eq.γ))
    ρ = -p * W[4]
    return SVector(ρ, u, v, p)
end

function vars_entropy2prim(W, eq::EulerEquation{3})
    u = -W[2] / W[5]
    v = -W[3] / W[5]
    w = -W[4] / W[5]
    s = eq.γ - (eq.γ - 1) * (W[1] - W[5] * (u^2 + v^2 + w^2) / 2)
    p = ((-W[5])^eq.γ * exp(s))^(1 / (1 - eq.γ))
    ρ = -p * W[5]
    return SVector(ρ, u, v, w, p)
end

function vars_entropy2cons(W, eq::EulerEquation)
    P = vars_entropy2prim(W, eq)
    return vars_prim2cons(P, eq)
end

"""
    normal_shockwave(ρ0, u0, p0, eq::EulerEquation)

Compute `ρ1`, `u1` and `p1` downstream of a normal shock-wave.
"""
function normal_shockwave(ρ0, u0, p0, eq::EulerEquation)
    # Inflow
    a = soundvelocity(ρ0, p0, eq)
    M0 = u0 / a

    # Outflow
    ρ1 = ρ0 * M0^2 * (eq.γ + 1) / ((eq.γ - 1) * M0^2 + 2)
    p1 = p0 * (2 * eq.γ * M0^2 - (eq.γ - 1)) / (eq.γ + 1)
    M1 = sqrt(((eq.γ - 1) * M0^2 + 2) / (2 * eq.γ * M0^2 - (eq.γ - 1)))
    a = soundvelocity(ρ1, p1, eq)
    u1 = M1 * a

    return SVector(ρ1, u1, p1)
end
