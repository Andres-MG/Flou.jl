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
tf = 0.15
solver = ORK256(williamson_condition=false)

equation = BurgersEquation()

std = StdSegment{Float64}(4, GL(), nvariables(equation))
mesh = CartesianMesh{1,Float64}(0, 1, 20)
apply_periodicBCs!(mesh, "1" => "2")

div = WeakDivOperator(LxFNumericalFlux(StdAverageNumericalFlux(), 1.0))
DG = DGSEM(mesh, std, equation, div, ())

x0 = 0.4
sx = 0.1
h = 1.0
Q = StateVector{Float64}(undef, nvariables(equation), DG.dofhandler)
for i in eachdof(DG)
    x = DG.geometry.elements.coords[i][1]
    Q.data[i, 1] = Flou.gaussian_bump(x, 0.0, 0.0, x0, 0.0, 0.0, sx, 1.0, 1.0, h)
end

display(DG)
println()

@info "Starting simulation..."

sol, exetime = timeintegrate(
    Q.data, DG, equation, solver, tf;
    alias_u0=true, adaptive=false, dt=Δt,
)

@info "Elapsed time: $(exetime) s"
@info "Time per iteration and DOF: $(exetime / (tf/Δt) / ndofs(DG)) s"
