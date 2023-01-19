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

std = FRStdQuad{Float64}(4, GLL(), :dgsem, nvariables(equation))
mesh = CartesianMesh{2,Float64}((0, 0), (1, 1), (10, 10))
apply_periodicBCs!(mesh, "1" => "2", "3" => "4")

∇ = WeakDivOperator(LxFNumericalFlux(StdAverageNumericalFlux(), 1.0))
dg = FR(mesh, std, equation, ∇, ())

x0 = y0 = 0.5
sx = sy = 0.1
h = 1.0
Q = StateVector{nvariables(equation),Float64}(undef, dg.dofhandler)
for i in eachdof(dg)
    x, y = dg.geometry.elements.coords[i]
    Q.dof[i] = Flou.gaussian_bump(x, y, x0, y0, sx, sy, h)
end

display(dg)
println()

sb = get_save_callback("../results/solution"; iter=save_steps)

@info "Starting simulation..."

_, exetime = timeintegrate(
    Q, dg, equation, solver, tf;
    save_everystep=false, alias_u0=true, adaptive=false, dt=Δt, callback=sb,
)

@info "Elapsed time: $(exetime) s"
@info "Time per iteration and DOF: $(exetime / (tf/Δt) / ndofs(dg)) s"
