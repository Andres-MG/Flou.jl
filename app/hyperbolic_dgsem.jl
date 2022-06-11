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
Δt = 1e-2
tf = 3.0
solver = ORK256(;williamson_condition=false)

order = (5, 5)
qtype = (GLL(), GLL())
div = SSFVDivOperator(
    ChandrasekharAverage(),
    LxFNumericalFlux(
        StdAverageNumericalFlux(),
        1.0,
    ),
    1e+0,
)
numflux = MatrixDissipation(
    ChandrasekharAverage(),
    1.0,
)

mesh = CartesianMesh{2,Float64}((-1, 0), (1, 1), (11, 3))
apply_periodicBCs!(mesh, 3 => 4)

equation = EulerEquation{2}(div, 1.4)

ρ0, M0, p0 = 1.0, 2.0, 1.0
a0 = Flou.soundvelocity(ρ0, p0, equation)
u0 = M0 * a0
ρ1, u1, p1 = Flou.normal_shockwave(ρ0, u0, p0, equation)
const Q0 = Flou.vars_prim2cons((ρ0, u0, 0.0, p0), equation)
const Q1 = Flou.vars_prim2cons((ρ1, u1, 0.0, p1), equation)

function Q!(Q, xy, n, t, b, time, equation)
    x = xy[1]
    Q .= (x < 0) ? Q0 : Q1
    return nothing
end
∂Ω = [
    1 => DirichletBC(Q!),
    2 => DirichletBC(Q!),
    # 3 => DirichletBC(Q!),
    # 4 => DirichletBC(Q!),
]
DG = DGSEM(mesh, order .+ 1, qtype, equation, ∂Ω, numflux)

Q = StateVector{Float64}(undef, DG.dofhandler, DG.stdvec, nvariables(equation))
for ie in eachelement(mesh)
    for i in eachindex(DG.stdvec[1])
        xy = coords(DG.physelem, ie)[i]
        Q!(view(Q[1], i, :, ie), xy, [], [], [], 0.0, equation)
    end
end

display(DG)

sb = get_save_callback("../results/solution", 0:0.05:tf)

@info "Starting simulation..."

_, exetime = integrate(Q, DG, solver, tf; save_everystep=false, alias_u0=true,
    adaptive=false, dt=Δt, callback=sb, progress=true, progress_steps=50)

@info "Elapsed time: $(exetime) s"
@info "Time per iteration and DOF: $(exetime / (tf/Δt) / ndofs(DG)) s"
