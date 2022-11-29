using Flou
using OrdinaryDiffEq
using LinearAlgebra: BLAS

# Header
print_flou_header()

# BLAS threading pool clashes with Julia's one 😢
if Threads.nthreads() > 1
    BLAS.set_num_threads(1)
end

# Discretization
Δt = 1e-3
tf = 1.0
save_steps = range(0, 1000, 21)
solver = ORK256(williamson_condition=false)

equation = LinearAdvection(2.0, -1.0)

std = StdQuad{Float64}(4, GLL(), nvariables(equation))
mesh = CartesianMesh{2,Float64}((0, 0), (1, 1), (10, 10))
apply_periodicBCs!(mesh, "1" => "2", "3" => "4")

div = StrongDivOperator(LxFNumericalFlux(StdAverageNumericalFlux(), 1.0))
DG = DGSEM(mesh, std, equation, div, ())

x0 = y0 = 0.5
sx = sy = 0.1
h = 1.0
Q = StateVector{nvariables(equation),Float64}(undef, DG.dofhandler)
for i in eachdof(DG)
    x, y = DG.geometry.elements.coords[i]
    Q.dof[i] = Flou.gaussian_bump(x, y, 0.0, x0, y0, 0.0, sx, sy, 1.0, h)
end

display(DG)
println()

sb = get_save_callback("../results/solution"; iter=save_steps)

@info "Starting simulation..."

_, exetime = timeintegrate(
    Q, DG, equation, solver, tf;
    save_everystep=false, alias_u0=true, adaptive=false, dt=Δt, callback=sb,
)

@info "Elapsed time: $(exetime) s"
@info "Time per iteration and DOF: $(exetime / (tf/Δt) / ndofs(DG)) s"
