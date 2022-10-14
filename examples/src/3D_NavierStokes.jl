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
Δt = 1e-4
tf = 0.01
save_steps = range(0, 100, 11)
solver = ORK256(williamson_condition=false)

div = SplitDivOperator(ChandrasekharAverage())
equation = EulerEquation{3}(div, 1.4)

std = StdHex{Float64}((4, 4, 4), GLL(), nvariables(equation))
mesh = UnstructuredMesh{3,Float64}("../test/meshes/mesh3d_refined.msh")

Q0 = Flou.vars_prim2cons((1.0, 2.0, 0.0, 0.0, 1.0), equation)
∂Ω = Dict(
    "Inlet" => EulerInflowBC(Q0),
    "Outlet" => EulerOutflowBC(),
    "Body" => EulerSlipBC(),
    "Interior" => EulerSlipBC(),
    "Wall" => EulerSlipBC(),
    "Left" => EulerSlipBC(),
    "Right" => EulerSlipBC(),
)

numflux = MatrixDissipation(ChandrasekharAverage(), 1.0)
DG = DGSEM(mesh, std, equation, ∂Ω, numflux)

Q = StateVector{Float64}(undef, nvariables(equation), DG.dofhandler)
Q.data .= Q0'

display(DG)
println()

mb, mvals = get_monitor_callback(Float64, DG, :entropy)
sb = get_save_callback("../results/solution", save_steps)
cb = make_callback_list(mb, sb)

@info "Starting simulation..."

_, exetime = timeintegrate(Q.data, DG, solver, tf;
    save_everystep=false, alias_u0=true, adaptive=false, dt=Δt, callback=cb,
    progress=true, progress_steps=5)

@info "Elapsed time: $(exetime) s"
@info "Time per iteration and DOF: $(exetime / (tf/Δt) / ndofs(DG)) s"
