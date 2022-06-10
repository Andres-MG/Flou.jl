function Advection1D()
    Δt = 1e-3
    tf = 0.5
    solver = ORK256(;williamson_condition=false)

    order = 4
    qtype = GL()
    div = WeakDivOperator()
    numflux = LxFNumericalFlux(
        StdAverageNumericalFlux(),
        1.0,
    )

    mesh = CartesianMesh{1,Float64}(0, 1, 20)
    apply_periodicBCs!(mesh, 1 => 2)
    equation = LinearAdvection(div, 2.0)
    DG = DGSEM(mesh, order + 1, qtype, equation, (), numflux)

    x0 = 0.5
    sx = 0.1
    h = 1.0
    Q = StateVector{Float64}(undef, DG.dofhandler, DG.stdvec, nvariables(equation))
    for ie in eachelement(mesh)
        for i in eachindex(DG.stdvec[1])
            x = coords(DG.physelem, ie)[i][1]
            Q[1][i, 1, ie] = Flou.gaussian_bump(x, 0.0, 0.0, x0, 0.0, 0.0, sx, 1.0, 1.0, h)
        end
    end

    sol, _ = integrate(
        Q, DG, solver, tf;
        saveat=(0, tf), adaptive=false, dt=Δt, alias_u0=true,
    )
    return sol
end

function Advection2D()
    Δt = 1e-3
    tf = 0.5
    solver = ORK256(;williamson_condition=false)

    order = (4, 4)
    qtype = (GL(), GL())
    div = WeakDivOperator()
    numflux = LxFNumericalFlux(
        StdAverageNumericalFlux(),
        1.0,
    )

    mesh = CartesianMesh{2,Float64}((0, 0), (1.5, 2), (20, 10))
    apply_periodicBCs!(mesh, 1 => 2, 3 => 4)
    equation = LinearAdvection(div, 3.0, 4.0)
    DG = DGSEM(mesh, order .+ 1, qtype, equation, (), numflux)

    x0, y0 = 0.75, 1.0
    sx, sy = 0.2, 0.2
    h = 1.0
    Q = StateVector{Float64}(undef, DG.dofhandler, DG.stdvec, nvariables(equation))
    for ie in eachelement(mesh)
        for i in eachindex(DG.stdvec[1])
            x, y = coords(DG.physelem, ie)[i]
            Q[1][i, 1, ie] = Flou.gaussian_bump(x, y, 0.0, x0, y0, 0.0, sx, sy, 1.0, h)
        end
    end

    sol, _ = integrate(
        Q, DG, solver, tf;
        saveat=(0, tf), adaptive=false, dt=Δt, alias_u0=true,
    )
    return sol
end

function SodTube1D()
    Δt = 1e-4
    tf = 0.018
    solver = ORK256(;williamson_condition=false)

    order = 3
    qtype = GLL()
    div = SplitDivOperator(
        ChandrasekharAverage(),
    )
    numflux = MatrixDissipation(
        ChandrasekharAverage(),
        1.0,
    )

    mesh = CartesianMesh{1,Float64}(0, 1, 20)
    function Q!(Q, x, n, t, b, time, eq)
        P = if x[1] < 0.5
            SVector(1.0, 0.0, 100.0)
        else
            SVector(0.125, 0.0, 10.0)
        end
        Q .= vars_prim2cons(P, eq)
        return nothing
    end
    ∂Ω = [
        1 => DirichletBC(Q!),
        2 => DirichletBC(Q!),
    ]
    equation = EulerEquation{1}(div, 1.4)
    DG = DGSEM(mesh, order + 1, qtype, equation, ∂Ω, numflux)

    Q = StateVector{Float64}(undef, DG.dofhandler, DG.stdvec, nvariables(equation))
    for ie in eachelement(mesh)
        for i in eachindex(DG.stdvec[1])
            x = coords(DG.physelem, ie)[i]
            Q!(view(Q[1], i, :, ie), x, [], [], [], 0.0, equation)
        end
    end

    sol, _ = integrate(
        Q, DG, solver, tf;
        saveat=(0, tf), adaptive=false, dt=Δt, alias_u0=true,
    )
    return sol
end

# https://www.math.ntnu.no/conservation/2001/049.pdf
function Implosion2D()
    Δt = 1e-4
    tf = 50Δt # 0.045
    solver = ORK256(;williamson_condition=false)

    order = (3, 3)
    qtype = (GLL(), GLL())
    div = SplitDivOperator(
        ChandrasekharAverage(),
    )
    numflux = MatrixDissipation(
        ChandrasekharAverage(),
        1.0,
    )

    mesh = CartesianMesh{2,Float64}((0, 0), (0.3, 0.3), (100, 100))
    ∂Ω = [
        1 => EulerSlipBC(),
        2 => EulerSlipBC(),
        3 => EulerSlipBC(),
        4 => EulerSlipBC(),
    ]
    equation = EulerEquation{2}(div, 1.4)
    DG = DGSEM(mesh, order .+ 1, qtype, equation, ∂Ω, numflux)

    Q = StateVector{Float64}(undef, DG.dofhandler, DG.stdvec, nvariables(equation))
    for ie in eachelement(mesh)
        for i in eachindex(DG.stdvec[1])
            x, y = coords(DG.physelem, ie)[i]
            if x <= 0.15 && y <= 0.15 - x
                Q[1][i, 1, ie] = 0.125
                Q[1][i, 2, ie] = 0.0
                Q[1][i, 3, ie] = 0.0
                Q[1][i, 4, ie] = 0.14 / 0.4
            else
                Q[1][i, 1, ie] = 1.0
                Q[1][i, 2, ie] = 0.0
                Q[1][i, 3, ie] = 0.0
                Q[1][i, 4, ie] = 1.0 / 0.4
            end
        end
    end

    sol, _ = integrate(
        Q, DG, solver, tf;
        saveat=(0, tf), adaptive=false, dt=Δt, alias_u0=true,
    )
    return sol
end
