struct EulerEquation{ND,NV,RT} <: HyperbolicEquation{ND,NV}
    γ::RT
end

function EulerEquation{ND}(γ) where {ND}
    1 <= ND <= 3 || throw(ArgumentError(
        "The Euler equations are only implemented in 1D, 2D and 3D."
    ))
    return EulerEquation{ND,ND + 2,typeof(γ)}(γ)
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
function pressure end

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
    energy(Q, eq::EulerEquation)

Compute the energy, `ρe`, from the *primitive* variables `Q`.
"""
function energy end

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

function numericalflux(
    Ql,
    Qr,
    _,
    eq::EulerEquation{ND},
    ::StdAverageNumericalFlux,
) where {
    ND,
}
    if ND == 1
        _, ρul, ρel = Ql
        _, ρur, ρer = Qr
        _, ul, pl = vars_cons2prim(Ql, eq)
        _, ur, pr = vars_cons2prim(Qr, eq)
        return SVector(
            (ρul + ρur) / 2,
            (ρul * ul + pl + ρur * ur + pr) / 2,
            ((ρel + pl) * ul + (ρer + pr) * ur) / 2,
        )

    elseif ND == 2
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

    else # ND == 3
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
end

function numericalflux(
    Ql,
    Qr,
    n,
    eq::EulerEquation{ND},
    nf::LxFNumericalFlux,
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
