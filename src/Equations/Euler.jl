struct EulerEquation{ND,NV,RT,DV} <: AbstractHyperbolicEquation{NV}
    div_operator::DV
    γ::RT
end

function EulerEquation{ND}(div_operator, γ) where {ND}
    1 <= ND <= 3 ||
        throw(ArgumentError("The Euler equations are only implemented in 1D, 2D and 3D."))
    EulerEquation{ND,ND + 2,typeof(γ),typeof(div_operator)}(div_operator, γ)
end

function Base.show(io::IO, m::MIME"text/plain", eq::EulerEquation)
    @nospecialize
    println(io, eq |> typeof, ":")
    print(io, " Advection operator: "); show(io, m, eq.div_operator); println(io, "")
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
    Qrot[2] = Qf[2]
    Qrot[3] = Qf[3]
    return nothing
end

function rotate2phys!(Qf, Qrot, n, t, b, ::EulerEquation{1})
    Qf[1] = Qrot[1]
    Qf[2] = Qrot[2]
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

function energy(Q, p, eq::EulerEquation{1})
    ρ, ρu, _ = Q
    return p / (eq.γ - 1) + ρu^2 / 2ρ
end

function energy(Q, p, eq::EulerEquation{2})
    ρ, ρu, ρv, _ = Q
    return p / (eq.γ - 1) + (ρu^2 + ρv^2) / 2ρ
end

function energy(Q, p, eq::EulerEquation{3})
    ρ, ρu, ρv, ρw, _ = Q
    return p / (eq.γ - 1) + (ρu^2 + ρv^2 + ρw^2) / 2ρ
end

function soundvelocity(Q, eq::EulerEquation)
    ρ = Q[1]
    p = pressure(Q, eq)
    return √(eq.γ * p / ρ)
end

function entropyvariables!(W, Q, eq::EulerEquation{ND}) where {ND}
    Wt = entropyvariables(Q, eq)
    copy!(W, Wt)
    return nothing
end

function entropyvariables(Q, eq::EulerEquation{1})
    ρ, ρu, _ = Q
    p = pressure(Q, eq)
    s = log(p) - eq.γ * log(ρ)
    return SVector{3}(
        (eq.γ - s) / (eq.γ - 1) - ρu^2 / ρ / 2p,
        ρu / p,
        -ρ / p,
    )
end

function entropyvariables(Q, eq::EulerEquation{2})
    ρ, ρu, ρv, _ = Q
    p = pressure(Q, eq)
    s = log(p) - eq.γ * log(ρ)
    return SVector{4}(
        (eq.γ - s) / (eq.γ - 1) - (ρu^2 + ρv^2) / ρ / 2p,
        ρu / p,
        ρv / p,
        -ρ / p,
    )
end

function entropyvariables(Q, eq::EulerEquation{3})
    ρ, ρu, ρv, ρw, _ = Q
    p = pressure(Q, eq)
    s = log(p) - eq.γ * log(ρ)
    return SVector{5}(
        (eq.γ - s) / (eq.γ - 1) - (ρu^2 + ρv^2 + ρw^2) / ρ / 2p,
        ρu / p,
        ρv / p,
        ρw / p,
        -ρ / p,
    )
end

#==========================================================================================#
#                                   Boundary Conditions                                    #

struct EulerInflowBC{RT} <: AbstractBC
    Qext::Vector{RT}
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

function numericalflux!(Fn, Ql, Qr, n, eq::EulerEquation, ::StdAverageNumericalFlux)
    ρl, ρul, ρvl, ρel = Ql
    ul = ρul/ρl
    pl = pressure(Ql, eq)
    ρr, ρur, ρvr, ρer = Qr
    ur = ρur/ρr
    pr = pressure(Qr, eq)
    Fn[1] = (ρul + ρur) / 2
    Fn[2] = (ρul * ul + pl + ρur * ur + pr) / 2
    Fn[3] = (ρvl * ul + ρvr * ur) / 2
    Fn[4] = ((ρel + pl) * ul + (ρer + pr) * ur) / 2
    return nothing
end

function numericalflux!(Fn, Ql, Qr, n, eq::EulerEquation, nf::LxFNumericalFlux)
    # Average
    numericalflux!(Fn, Ql, Qr, n, eq, nf.avg)

    # Dissipation
    ul, ur = Ql[2]/Ql[1], Qr[2]/Qr[1]
    al = soundvelocity(Ql, eq)
    ar = soundvelocity(Qr, eq)
    λ = max(abs(ul) + al, abs(ur) + ar)
    @inbounds for ivar in eachvariable(eq)
        Fn[ivar] += λ * (Ql[ivar] - Qr[ivar]) / 2 * nf.intensity
    end
    return nothing
end

struct ChandrasekharAverage <: AbstractNumericalFlux end

function numericalflux!(Fn, Ql, Qr, n, eq::EulerEquation, ::ChandrasekharAverage)
    # Unpacking
    ρl, ρul, ρvl, _ = Ql
    ρr, ρur, ρvr, _ = Qr

    # Variables
    ur, vr = ρur/ρr, ρvr/ρr
    ul, vl = ρul/ρl, ρvl/ρl
    pl = pressure(Ql, eq)
    pr = pressure(Qr, eq)
    βl, βr = ρl / 2pl, ρr / 2pr

    # Averages
    ρ = logarithmic_mean(ρl, ρr)
    u, v = (ul + ur) / 2, (vl + vr) / 2
    p = (ρl + ρr) / (2 * (βl + βr))
    β = logarithmic_mean(βl, βr)
    h = 1 / (2β * (eq.γ - 1)) - (ul^2 + vl^2 + ur^2 + vr^2) / 4 + p/ρ + u^2 + v^2

    # Fluxes
    Fn[1] = ρ * u
    Fn[2] = ρ * u^2 + p
    Fn[3] = ρ * u * v
    Fn[4] = ρ * u * h
    return nothing
end

struct MatrixDissipation{T,RT} <: AbstractNumericalFlux
    avg::T
    intensity::RT
end

function numericalflux!(Fn, Ql, Qr, n, eq::EulerEquation, nf::MatrixDissipation)
    # Unpacking
    ρl, ρul, ρvl, _ = Ql
    ρr, ρur, ρvr, _ = Qr

    # Variables
    ur, vr = ρur/ρr, ρvr/ρr
    ul, vl = ρul/ρl, ρvl/ρl
    pl = pressure(Ql, eq)
    pr = pressure(Qr, eq)
    βl, βr = ρl / 2pl, ρr / 2pr

    # Averages
    ρ = logarithmic_mean(ρl, ρr)
    u, v = (ul + ur) / 2, (vl + vr) / 2
    p = (ρl + ρr) / (2 * (βl + βr))
    β = logarithmic_mean(βl, βr)
    a = √(eq.γ * p / ρ)
    v2 = 2 * (u^2 + v^2) - (ul^2 + vl^2 + ur^2 + vl^2) / 2
    h = eq.γ / (eq.γ - 1) / 2β + v2 / 2

    # Averaging term
    numericalflux!(Fn, Ql, Qr, n, eq, nf.avg)

    # Dissipative term
    rt = eltype(Fn)
    Wl = entropyvariables(Ql, eq)
    Wr = entropyvariables(Qr, eq)
    Λ = SDiagonal{4}((u - a, u, u, u + a) .|> abs)
    T = SDiagonal{4}((ρ / 2eq.γ, (eq.γ - 1) * ρ / eq.γ, p, ρ / 2eq.γ))
    R = SMatrix{4,4}(
        one(rt),  u - a,    v,       h - u * a,
        one(rt),  u,        v,       v2 / 2,
        zero(rt), zero(rt), one(rt), v,
        one(rt),  u + a,    v,       h + u * a,
    )
    Fn .+= R * Λ * T * R' * (Wl .- Wr) ./ 2 .* nf.intensity
    return nothing
end

#==========================================================================================#
#                                     Two-point fluxes                                     #

function twopointflux!(F♯, Q1, Q2, Ja1, Ja2, eq::EulerEquation, ::ChandrasekharAverage)
    # Unpacking
    ρ1, ρu1, ρv1, _ = Q1
    ρ2, ρu2, ρv2, _ = Q2

    # Variables
    u1, v1 = ρu1/ρ1, ρv1/ρ1
    u2, v2 = ρu2/ρ2, ρv2/ρ2
    p1 = pressure(Q1, eq)
    p2 = pressure(Q2, eq)
    β1, β2 = ρ1 / 2p1, ρ2 / 2p2

    # Averages
    ρ = logarithmic_mean(ρ1, ρ2)
    u, v = (u1 + u2) / 2, (v1 + v2) / 2
    p = (ρ1 + ρ2) / (2 * (β1 + β2))
    β = logarithmic_mean(β1, β2)
    h = 1 / (2β * (eq.γ - 1)) - (u1^2 + v1^2 + u2^2 + v2^2) / 4 + p/ρ + u^2 + v^2

    # Fluxes
    n = SVector{2}((Ja1 .+ Ja2) ./ 2)
    F♯[1] = (ρ * u) * n[1] + (ρ * v) * n[2]
    F♯[2] = (ρ * u^2 + p) * n[1] + (ρ * u * v) * n[2]
    F♯[3] = (ρ * u * v) * n[1] + (ρ * v^2 + p) * n[2]
    F♯[4] = (ρ * u * h) * n[1] + (ρ * v * h) * n[2]
    return nothing
end
