function Advection1D()
    Δt = 1e-3
    tf = 0.5
    solver = ORK256(;williamson_condition=false)

    equation = LinearAdvection(2.0)

    std = StdSegment{Float64}(5, GL(), nvariables(equation))
    mesh = CartesianMesh{1,Float64}(0, 1, 20)
    apply_periodicBCs!(mesh, "1" => "2")

    ∇ = WeakDivOperator(LxFNumericalFlux(StdAverageNumericalFlux(), 1.0))
    DG = DGSEM(mesh, std, equation, ∇, ())

    x0 = 0.5
    sx = 0.1
    h = 1.0
    Q = StateVector{nvariables(equation),Float64}(undef, DG.dofhandler)
    for i in eachdof(DG)
        x = DG.geometry.elements.coords[i][1]
        Q.data[i] = Flou.gaussian_bump(x, 0.0, 0.0, x0, 0.0, 0.0, sx, 1.0, 1.0, h)
    end

    sol, _ = timeintegrate(
        Q.data.flat, DG, equation, solver, tf;
        saveat=(0, tf), adaptive=false, dt=Δt, alias_u0=true,
    )
    return sol
end

function Advection2D()
    Δt = 1e-3
    tf = 0.5
    solver = ORK256(;williamson_condition=false)

    equation = LinearAdvection(3.0, 4.0)

    std = StdQuad{Float64}(5, GL(), nvariables(equation))
    mesh = CartesianMesh{2,Float64}((0, 0), (1.5, 2), (20, 10))
    apply_periodicBCs!(mesh, "1" => "2", "3" => "4")

    ∇ = WeakDivOperator(LxFNumericalFlux(StdAverageNumericalFlux(), 1.0))
    DG = DGSEM(mesh, std, equation, ∇, ())

    x0, y0 = 0.75, 1.0
    sx, sy = 0.2, 0.2
    h = 1.0
    Q = StateVector{nvariables(equation),Float64}(undef, DG.dofhandler)
    for i in eachdof(DG)
        x, y = DG.geometry.elements.coords[i]
        Q.data[i] = Flou.gaussian_bump(x, y, 0.0, x0, y0, 0.0, sx, sy, 1.0, h)
    end

    sol, _ = timeintegrate(
        Q.data.flat, DG, equation, solver, tf;
        saveat=(0, tf), adaptive=false, dt=Δt, alias_u0=true,
    )
    return sol
end

function SodTube1D()
    Δt = 1e-4
    tf = 0.018
    solver = ORK256(;williamson_condition=false)

    equation = EulerEquation{1}(1.4)

    std = StdSegment{Float64}(4, GLL(), nvariables(equation))
    mesh = CartesianMesh{1,Float64}(0, 1, 20)

    function Qext(_, x, _, _, eq)
        P = if x[1] < 0.5
            SVector(1.0, 0.0, 100.0)
        else
            SVector(0.125, 0.0, 10.0)
        end
        return Flou.vars_prim2cons(P, eq)
    end
    ∂Ω = Dict(
        "1" => GenericBC(Qext),
        "2" => GenericBC(Qext),
    )

    ∇ = SplitDivOperator(
        ChandrasekharAverage(),
        MatrixDissipation(ChandrasekharAverage(), 1.0),
    )
    DG = DGSEM(mesh, std, equation, ∇, ∂Ω)

    Q = StateVector{nvariables(equation),Float64}(undef, DG.dofhandler)
    for i in eachdof(DG)
        x = DG.geometry.elements.coords[i]
        Q.data[i] = Qext((), x, (), 0.0, equation)
    end

    sol, _ = timeintegrate(
        Q.data.flat, DG, equation, solver, tf;
        saveat=(0, tf), adaptive=false, dt=Δt, alias_u0=true,
    )
    return sol
end

function Shockwave2D()
    Δt = 1e-2
    tf = 1.0
    solver = ORK256(;williamson_condition=false)

    equation = EulerEquation{2}(1.4)

    std = StdQuad{Float64}(6, GLL(), nvariables(equation))
    mesh = CartesianMesh{2,Float64}((-1, 0), (1, 1), (11, 3))
    apply_periodicBCs!(mesh, "3" => "4")

    ρ0, M0, p0 = 1.0, 2.0, 1.0
    a0 = Flou.soundvelocity(ρ0, p0, equation)
    u0 = M0 * a0
    ρ1, u1, p1 = Flou.normal_shockwave(ρ0, u0, p0, equation)
    Q0 = Flou.vars_prim2cons((ρ0, u0, 0.0, p0), equation)
    Q1 = Flou.vars_prim2cons((ρ1, u1, 0.0, p1), equation)

    function Qext(_, xy, _, _, _)
        x = xy[1]
        return (x < 0) ? Q0 : Q1
    end
    ∂Ω = Dict(
        "1" => GenericBC(Qext),
        "2" => GenericBC(Qext),
    )

    ∇ = SSFVDivOperator(
        ChandrasekharAverage(),
        LxFNumericalFlux(
            StdAverageNumericalFlux(),
            1.0,
        ),
        1e+0,
        MatrixDissipation(ChandrasekharAverage(), 1.0),
    )
    DG = DGSEM(mesh, std, equation, ∇, ∂Ω)

    Q = StateVector{nvariables(equation),Float64}(undef, DG.dofhandler)
    for i in eachdof(DG)
        xy = DG.geometry.elements.coords[i]
        Q.data[i] = Qext((), xy, (), 0.0, equation)
    end

    sol, _ = timeintegrate(
        Q.data.flat, DG, equation, solver, tf;
        saveat=(0, tf), adaptive=false, dt=Δt, alias_u0=true,
    )
    return sol
end

# https://www.math.ntnu.no/conservation/2001/049.pdf
function Implosion2D()
    Δt = 1e-4
    tf = 50Δt # 0.045
    solver = ORK256(;williamson_condition=false)

    equation = EulerEquation{2}(1.4)

    std = StdQuad{Float64}(4, GLL(), nvariables(equation))
    mesh = CartesianMesh{2,Float64}((0, 0), (0.3, 0.3), (100, 100))
    ∂Ω = Dict(
        "1" => EulerSlipBC(),
        "2" => EulerSlipBC(),
        "3" => EulerSlipBC(),
        "4" => EulerSlipBC(),
    )

    ∇ = SplitDivOperator(
        ChandrasekharAverage(),
        MatrixDissipation(ChandrasekharAverage(), 1.0),
    )
    DG = DGSEM(mesh, std, equation, ∇, ∂Ω)

    Q = StateVector{nvariables(equation),Float64}(undef, DG.dofhandler)
    for i in eachdof(DG)
        x, y = DG.geometry.elements.coords[i]
        if x <= 0.15 && y <= 0.15 - x
            Q.data[i] = (0.125, 0.0, 0.0, 0.14 / 0.4)
        else
            Q.data[i] = (1.0, 0.0, 0.0, 1.0 / 0.4)
        end
    end

    sol, _ = timeintegrate(
        Q.data.flat, DG, equation, solver, tf;
        saveat=(0, tf), adaptive=false, dt=Δt, alias_u0=true,
    )
    return sol
end

function ForwardFacingStep2D()
    Δt = 1e-4
    tf = 0.1 # 2.0
    solver = ORK256(;williamson_condition=false)

    equation = EulerEquation{2}(1.4)

    std = StdQuad{Float64}(8, GLL(), nvariables(equation))
    mesh = StepMesh{Float64}((0,0), (3, 1), 0.6, 0.2, ((10, 5), (10, 20), (40, 20)))

    M0 = 3.0
    a0 = soundvelocity(1.0, 1.0, equation)
    Q0 = Flou.vars_prim2cons((1.0, M0*a0, 0.0, 1.0), equation)
    ∂Ω = Dict(
        "1" => EulerInflowBC(Q0),
        "2" => EulerOutflowBC(),
        "3" => EulerSlipBC(),
        "4" => EulerSlipBC(),
        "5" => EulerSlipBC(),
        "6" => EulerSlipBC(),
    )

    ∇ = SSFVDivOperator(
        ChandrasekharAverage(),
        LxFNumericalFlux(
            StdAverageNumericalFlux(),
            0.2,
        ),
        0.1^2,
        MatrixDissipation(ChandrasekharAverage(), 1.0),
    )
    DG = DGSEM(mesh, std, equation, ∇, ∂Ω)

    Q = StateVector{nvariables(equation),Float64}(undef, DG.dofhandler)
    Q.data .= (Q0,)

    sol, _ = timeintegrate(
        Q.data.flat, DG, equation, solver, tf;
        saveat=(0, tf), adaptive=false, dt=Δt, alias_u0=true,
    )
    return sol
end
