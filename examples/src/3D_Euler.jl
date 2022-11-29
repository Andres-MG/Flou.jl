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
tf = 0.25
save_steps = range(0, 2500, 11)
solver = ORK256(williamson_condition=false)

equation = EulerEquation{3}(1.4)

std = StdHex{Float64}(4, GLL(), nvariables(equation))
mesh = UnstructuredMesh{3,Float64}("../test/meshes/3D_bullet_refined.msh")

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

∇ = SplitDivOperator(
    ChandrasekharAverage(),
    MatrixDissipation(ChandrasekharAverage(), 1.0),
)
DG = DGSEM(mesh, std, equation, ∇, ∂Ω)

Q = StateVector{nvariables(equation),Float64}(undef, DG.dofhandler)
for i in eachdof(Q)
    Q.dof[i] = Q0
end

display(DG)
println()

mb, mvals = get_monitor_callback(Float64, DG, equation, :entropy)
sb = get_save_callback("../results/solution"; iter=save_steps)
cb = make_callback_list(mb, sb)

@info "Starting simulation..."

_, exetime = timeintegrate(
    Q, DG, equation, solver, tf;
    save_everystep=false, alias_u0=true, adaptive=false, dt=Δt, callback=cb,
    progress=true, progress_steps=5,
)

@info "Elapsed time: $(exetime) s"
@info "Time per iteration and DOF: $(exetime / (tf/Δt) / ndofs(DG)) s"
