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
Δt = 2e-6
tf = 0.01
# solver = ORK256(williamson_condition=false)
# solver = SSPRK53()
# solver = DGLDDRK73_C()
solver = CarpenterKennedy2N54(williamson_condition=false)

std = StdQuad{Float64}((8, 8), (GLL(), GLL()))
div = SSFVDivOperator(
    ChandrasekharAverage(),
    LxFNumericalFlux(
        StdAverageNumericalFlux(),
        0.2,
    ),
    # MatrixDissipation(
    #     ChandrasekharAverage(),
	# 1.0,
    # ),
    0.1,
)
numflux = MatrixDissipation(
    ChandrasekharAverage(),
    1.0,
)

mesh = StepMesh{Float64}((0,0), (2, 2), 1.5, 1.0, ((30, 20), (30, 20), (10, 20)))

equation = EulerEquation{2}(div, 1.4)

Q0 = Flou.vars_prim2cons((5.997, -98.5914, 0.0, 11_666.5), equation)
Q1 = Flou.vars_prim2cons((1.0, 0.0, 0.0, 1.0), equation)
∂Ω = [
    1 => EulerOutflowBC(),
    2 => EulerInflowBC(Q0),
    3 => EulerSlipBC(),
    4 => EulerSlipBC(),
    5 => EulerSlipBC(),
    6 => EulerSlipBC(),
]
DG = DGSEM(mesh, std, equation, ∂Ω, numflux)

Q = StateVector{Float64}(undef, DG.dofhandler, DG.stdvec, nvariables(equation))
for ie in eachelement(mesh)
    for i in eachindex(DG.stdvec[1])
        x = coords(DG.physelem, ie)[i][1]
        if x > 1.5
            Q[1][i, :, ie] .= Q0
        else
            Q[1][i, :, ie] .= Q1
        end
    end
end

display(DG)
println()

sb = get_save_callback("../results/solution", range(0, tf, 30))

@info "Starting simulation..."

_, exetime = integrate(Q, DG, solver, tf; save_everystep=false, alias_u0=true,
    adaptive=false, dt=Δt, callback=sb, progress=true, progress_steps=50)

@info "Elapsed time: $(exetime) s"
@info "Time per iteration and DOF: $(exetime / (tf/Δt) / ndofs(DG)) s"
