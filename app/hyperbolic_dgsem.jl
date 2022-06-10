using Flou
using OrdinaryDiffEq
using LinearAlgebra: BLAS

# Progress of ODEsolver
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

# Header
print_flou_header()

# BLAS threading pool clashes with Julia's one 😢
if Threads.nthreads() > 1
    BLAS.set_num_threads(1)
end

# Discretization
Δt = 5e-5
tf = Δt # 0.05
solver = ORK256(;williamson_condition=false)

order = (5, 5)
qtype = (GLL(), GLL())
# div = SplitDivOperator(
#     ChandrasekharAverage(),
# )
div = SSFVDivOperator(
    ChandrasekharAverage(),
    LxFNumericalFlux(
        StdAverageNumericalFlux(),
        1.0,
    ),
    1e-10,
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

display(DG)

sb = get_save_callback("../results/solution", 0:0.01:tf)

@info "Starting simulation..."

_, exetime = integrate(Q, DG, solver, tf; save_everystep=false, alias_u0=true,
    adaptive=false, dt=Δt, callback=sb, progress=true, progress_steps=50)

@info "Elapsed time: $(exetime) s"
@info "Time per iteration and DOF: $(exetime / (tf/Δt) / ndofs(DG)) s"
