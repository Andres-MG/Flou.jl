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
Δt = 1e-4
tf = 0.15
solver = ORK256(williamson_condition=false)

equation = BurgersEquation()

std = FRStdSegment{Float64}(4, GLL(), :dgsem, nvariables(equation))
mesh = CartesianMesh{1,Float64}(0, 1, 20)
apply_periodicBCs!(mesh, "1" => "2")

∇ = WeakDivOperator(LxFNumericalFlux(StdAverageNumericalFlux(), 1.0))
dg = FR(mesh, std, equation, ∇, ())

x0 = 0.4
sx = 0.1
h = 1.0
Q = StateVector{nvariables(equation),Float64}(undef, dg.dofhandler)
for i in eachdof(dg)
    x = dg.geometry.elements.coords[i][1]
    Q.dof[i] = Flou.gaussian_bump(x, x0, sx, h)
end

display(dg)
println()

@info "Starting simulation..."

sol, exetime = timeintegrate(
    Q, dg, equation, solver, tf;
    alias_u0=true, adaptive=false, dt=Δt,
)

@info "Elapsed time: $(exetime) s"
@info "Time per iteration and DOF: $(exetime / (tf/Δt) / ndofs(dg)) s"
