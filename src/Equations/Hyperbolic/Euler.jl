struct EulerEquation{ND,NV,RT,DV} <: HyperbolicEquation{NV}
    operators::Tuple{DV}
    γ::RT
end

function EulerEquation{ND}(div_operator, γ) where {ND}
    1 <= ND <= 3 ||
        throw(ArgumentError("The Euler equations are only implemented in 1D, 2D and 3D."))
    EulerEquation{ND,ND + 2,typeof(γ),typeof(div_operator)}((div_operator,), γ)
end

function Base.show(io::IO, m::MIME"text/plain", eq::EulerEquation)
    @nospecialize
    println(io, eq |> typeof, ":")
    print(io, " Advection operator: "); show(io, m, eq.operators[1]); println(io, "")
    print(io, " γ: ", eq.γ)
end

function variablenames(::EulerEquation{ND}; unicode=false) where {ND}
    return if unicode
        if ND == 1
            (:ρ, :ρu, :ρe)
        elseif ND == 2
            (:ρ, :ρu, :ρv, :ρe)
        else # ND == 3
            (:ρ, :ρu, :ρv, :ρw, :ρe)
        end
    else
        if ND == 1
            (:rho, :rhou, :rhoe)
        elseif ND == 2
            (:rho, :rhou, :rhov, :rhoe)
        else # ND == 3
            (:rho, :rhou, :rhov, :rhow, :rhoe)
        end
    end
end

function volumeflux!(F, Q, eq::EulerEquation{1})
    ρ, ρu, ρe = Q
    u = ρu/ρ
    p = pressure(Q, eq)
    F[1, 1] = ρu
    F[1, 2] = ρu * u + p
    F[1, 3] = (ρe + p) * u
    return nothing
end

function volumeflux!(F, Q, eq::EulerEquation{2})
    ρ, ρu, ρv, ρe = Q
    u, v = ρu/ρ, ρv/ρ
    p = pressure(Q, eq)
    F[1, 1] = ρu           ; F[2, 1] = ρv
    F[1, 2] = ρu * u + p   ; F[2, 2] = ρu * v
    F[1, 3] = ρv * u       ; F[2, 3] = ρv * v + p
    F[1, 4] = (ρe + p) * u ; F[2, 4] = (ρe + p) * v
    return nothing
end

function volumeflux!(F, Q, eq::EulerEquation{3})
    ρ, ρu, ρv, ρw, ρe = Q
    u, v, w = ρu/ρ, ρv/ρ, ρw/ρ
    p = pressure(Q, eq)
    F[1, 1] = ρu           ; F[2, 1] = ρv           ; F[3, 1] = ρw
    F[1, 2] = ρu * u + p   ; F[2, 2] = ρu * v       ; F[3, 2] = ρu * w
    F[1, 3] = ρv * u       ; F[2, 3] = ρv * v + p   ; F[3, 3] = ρv * w
    F[1, 4] = ρw * u       ; F[2, 4] = ρw * v       ; F[3, 4] = ρw * w + p
    F[1, 5] = (ρe + p) * u ; F[2, 5] = (ρe + p) * v ; F[3, 5] = (ρe + p) * w
    return nothing
end

function rotate2face!(Qrot, Qf, n, t, b, ::EulerEquation{1})
    Qrot[1] = Qf[1]
    Qrot[2] = Qf[2] * n[1]
    Qrot[3] = Qf[3]
    return nothing
end

function rotate2phys!(Qf, Qrot, n, t, b, ::EulerEquation{1})
    Qf[1] = Qrot[1]
    Qf[2] = Qrot[2] * n[1]
    Qf[3] = Qrot[3]
    return nothing
end

function rotate2face!(Qrot, Qf, n, t, b, ::EulerEquation{2})
    Qrot[1] = Qf[1]
    Qrot[2] = Qf[2] * n[1] + Qf[3] * n[2]
    Qrot[3] = Qf[2] * t[1] + Qf[3] * t[2]
    Qrot[4] = Qf[4]
    return nothing
end

function rotate2phys!(Qf, Qrot, n, t, b, ::EulerEquation{2})
    Qf[1] = Qrot[1]
    Qf[2] = Qrot[2] * n[1] + Qrot[3] * t[1]
    Qf[3] = Qrot[2] * n[2] + Qrot[3] * t[2]
    Qf[4] = Qrot[4]
    return nothing
end

function rotate2face!(Qrot, Qf, n, t, b, ::EulerEquation{3})
    Qrot[1] = Qf[1]
    Qrot[2] = Qf[2] * n[1] + Qf[3] * n[2] + Qf[4] * n[3]
    Qrot[3] = Qf[2] * t[1] + Qf[3] * t[2] + Qf[4] * t[3]
    Qrot[4] = Qf[2] * b[1] + Qf[3] * b[2] + Qf[4] * b[3]
    Qrot[5] = Qf[5]
    return nothing
end

function rotate2phys!(Qf, Qrot, n, t, b, ::EulerEquation{3})
    Qf[1] = Qrot[1]
    Qf[2] = Qrot[2] * n[1] + Qrot[3] * t[1] + Qrot[4] * b[1]
    Qf[3] = Qrot[2] * n[2] + Qrot[3] * t[2] + Qrot[4] * b[2]
    Qf[4] = Qrot[2] * n[2] + Qrot[3] * t[2] + Qrot[4] * b[3]
    Qf[5] = Qrot[5]
    return nothing
end

"""
    pressure(Q, eq::EulerEquation)

Compute the pressure from the *conservative* variables `Q`.
"""
function pressure end

function pressure(Q, eq::EulerEquation{1})
    ρ, ρu, ρe = Q
    return (eq.γ-1) * (ρe - ρu^2 / 2ρ)
end

function pressure(Q, eq::EulerEquation{2})
    ρ, ρu, ρv, ρe = Q
    return (eq.γ-1) * (ρe - (ρu^2 + ρv^2) / 2ρ)
end

function pressure(Q, eq::EulerEquation{3})
    ρ, ρu, ρv, ρw, ρe = Q
    return (eq.γ-1) * (ρe - (ρu^2 + ρv^2 + ρw^2) / 2ρ)
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

function soundvelocity(ρ, p, eq::EulerEquation)
    return sqrt(eq.γ * p / ρ)
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
    s = log(p) - eq.γ * log(ρ)
    return SVector(
        (eq.γ - s) / (eq.γ - 1) - ρu^2 / ρ / 2p,
        ρu / p,
        -ρ / p,
    )
end

function vars_cons2entropy(Q, eq::EulerEquation{2})
    ρ, ρu, ρv, _ = Q
    p = pressure(Q, eq)
    s = log(p) - eq.γ * log(ρ)
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
    s = log(p) - eq.γ * log(ρ)
    return SVector(
        (eq.γ - s) / (eq.γ - 1) - (ρu^2 + ρv^2 + ρw^2) / ρ / 2p,
        ρu / p,
        ρv / p,
        ρw / p,
        -ρ / p,
    )
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
        3 <= nvar <= 5 ||
            throw(ArgumentError("'Qext' must have a length of 3, 4 or 5."))
        return new{eltype(Qext),nvar}(SVector{nvar}(Qext))
    end
end

function stateBC!(Q, x, n, t, b, time, ::EulerEquation, bc::EulerInflowBC)
    copy!(Q, bc.Qext)
    return nothing
end

struct EulerOutflowBC <: AbstractBC end

function stateBC!(Q, x, n, t, b, time, ::EulerEquation, ::EulerOutflowBC)
    return nothing
end

struct EulerSlipBC <: AbstractBC end

function stateBC!(Q, x, n, t, b, time, eq::EulerEquation, ::EulerSlipBC)
    Qn = similar(Q)
    rotate2face!(Qn, Q, n, t, b, eq)
    Qn[2] = -Qn[2]
    rotate2phys!(Q, Qn, n, t, b, eq)
    return nothing
end

#==========================================================================================#
#                                     Numerical fluxes                                     #

function numericalflux!(
    Fn,
    Ql,
    Qr,
    n,
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
        Fn[1] = (ρul + ρur) / 2
        Fn[2] = (ρul * ul + pl + ρur * ur + pr) / 2
        Fn[3] = ((ρel + pl) * ul + (ρer + pr) * ur) / 2

    elseif ND == 2
        _, ρul, ρvl, ρel = Ql
        _, ρur, ρvr, ρer = Qr
        _, ul, _, pl = vars_cons2prim(Ql, eq)
        _, ur, _, pr = vars_cons2prim(Qr, eq)
        Fn[1] = (ρul + ρur) / 2
        Fn[2] = (ρul * ul + pl + ρur * ur + pr) / 2
        Fn[3] = (ρvl * ul + ρvr * ur) / 2
        Fn[4] = ((ρel + pl) * ul + (ρer + pr) * ur) / 2

    else # ND == 3
        _, ρul, ρvl, ρwl, ρel = Ql
        _, ρur, ρvr, ρwr, ρer = Qr
        _, ul, _, _, pl = vars_cons2prim(Ql, eq)
        _, ur, _, _, pr = vars_cons2prim(Qr, eq)
        Fn[1] = (ρul + ρur) / 2
        Fn[2] = (ρul * ul + pl + ρur * ur + pr) / 2
        Fn[3] = (ρvl * ul + ρvr * ur) / 2
        Fn[4] = (ρwl * ul + ρwr * ur) / 2
        Fn[5] = ((ρel + pl) * ul + (ρer + pr) * ur) / 2
    end
    return nothing
end

function numericalflux!(
    Fn,
    Ql,
    Qr,
    n,
    eq::EulerEquation{ND},
    nf::LxFNumericalFlux,
) where {
    ND,
}
    # Average
    numericalflux!(Fn, Ql, Qr, n, eq, nf.avg)

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
    @inbounds for ivar in eachvariable(eq)
        Fn[ivar] += λ * (Ql[ivar] - Qr[ivar]) / 2 * nf.intensity
    end
    return nothing
end

struct ChandrasekharAverage <: AbstractNumericalFlux end

function numericalflux!(
    Fn,
    Ql,
    Qr,
    n,
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
        ρr, ur, vr, wl, pr = vars_cons2prim(Qr, eq)
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
        Fn[1] = ρ * u
        Fn[2] = ρ * u^2 + p
        Fn[3] = ρ * u * h
    elseif ND == 2
        h = 1 / (2β * (eq.γ - 1)) - (ul^2 + vl^2 + ur^2 + vr^2) / 4 + p/ρ + u^2 + v^2
        Fn[1] = ρ * u
        Fn[2] = ρ * u^2 + p
        Fn[3] = ρ * u * v
        Fn[4] = ρ * u * h
    else # ND == 3
        h = 1 / (2β * (eq.γ - 1)) - (ul^2 + vl^2 + wl^2 + ur^2 + vr^2 + wl^2) / 4 +
            p/ρ + u^2 + v^2 + w^2
        Fn[1] = ρ * u
        Fn[2] = ρ * u^2 + p
        Fn[3] = ρ * u * v
        Fn[4] = ρ * u * w
        Fn[5] = ρ * u * h
    end
    return nothing
end

struct MatrixDissipation{T,RT} <: AbstractNumericalFlux
    avg::T
    intensity::RT
end

function numericalflux!(
    Fn,
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
    h = eq.γ / (eq.γ - 1) / 2β + v2 / 2

    # Averaging term
    numericalflux!(Fn, Ql, Qr, n, eq, nf.avg)

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
    Fn .+= R * Λ * T * R' * (Wl .- Wr) ./ 2 .* nf.intensity
    return nothing
end

#==========================================================================================#
#                                     Two-point fluxes                                     #

function twopointflux!(
    F♯,
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
        F♯[1] = (ρ * u) * n
        F♯[2] = (ρ * u^2 + p) * n
        F♯[3] = (ρ * u * h) * n
    elseif ND == 2
        h = 1 / (2β * (eq.γ - 1)) - (u1^2 + v1^2 + u2^2 + v2^2) / 4 + p/ρ + u^2 + v^2
        n = SVector((Ja1 .+ Ja2) ./ 2)
        F♯[1] = (ρ * u) * n[1] + (ρ * v) * n[2]
        F♯[2] = (ρ * u^2 + p) * n[1] + (ρ * u * v) * n[2]
        F♯[3] = (ρ * u * v) * n[1] + (ρ * v^2 + p) * n[2]
        F♯[4] = (ρ * u * h) * n[1] + (ρ * v * h) * n[2]
    else # ND == 3
        h = 1 / (2β * (eq.γ - 1)) - (u1^2 + v1^2 + w1^2 + u2^2 + v2^2 + w2^2) / 4 +
            p/ρ + u^2 + v^2 + w^2
        n = SVector((Ja1 .+ Ja2) ./ 2)
        F♯[1] = (ρ * u) * n[1] + (ρ * v) * n[2] + (ρ * w) * n[3]
        F♯[2] = (ρ * u^2 + p) * n[1] + (ρ * u * v) * n[2] + (ρ * u * w) * n[3]
        F♯[3] = (ρ * u * v) * n[1] + (ρ * v^2 + p) * n[2] + (ρ * v * w) * n[3]
        F♯[4] = (ρ * u * w) * n[1] + (ρ * v * w) * n[2] + (ρ * w^2 + p) * n[3]
        F♯[5] = (ρ * u * h) * n[1] + (ρ * v * h) * n[2] + (ρ * w * h) * n[3]
    end
    return nothing
end
