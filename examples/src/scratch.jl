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
# save_steps = 0:Int(round(tf/Δt))
# save_steps = range(0, Int(round(tf/Δt)), 11) .|> Int
save_steps = (0, Int(round(tf/Δt)))
solver = ORK256(williamson_condition=false)
# solver = DGLDDRK84_F()

std = StdSegment{Float64}(4, GL())
# std = StdQuad{Float64}((4, 4), GLL())
div = SSFVDivOperator(
    ChandrasekharAverage(),
    # LxFNumericalFlux(
    #     StdAverageNumericalFlux(),
    #     1.0,
    # ),
    # 1.0,
    ScalarDissipation(
        ChandrasekharAverage(),
        1.0,
    ),
    1.0,
)
# div = SplitDivOperator(
#     ChandrasekharAverage(),
# )
# div = WeakDivOperator()
# numflux = ScalarDissipation(
#     ChandrasekharAverage(),
#     1.0,
# )
# numflux = LxFNumericalFlux(
#     StdAverageNumericalFlux(),
#     1.0,
# )
# numflux = ChandrasekharAverage()
numflux = MatrixDissipation(
    ChandrasekharAverage(),
    1.0,
)

mesh = CartesianMesh{1,Float64}(0, 1, 21)
# mesh = CartesianMesh{2,Float64}((0, 0), (1, 0.5), (11, 5))
# mesh = StepMesh{Float64}((0, 0), (3, 1), 0.6, 0.2, ((16, 8), (16, 24), (32, 24)))
# mesh = StepMesh{Float64}((0, 0), (2, 2), 1.5, 1.0, ((30, 20), (30, 20), (10, 20)))
# mesh = UnstructuredMesh{2,Float64}("../test/meshes/mesh2d_refined.msh")
apply_periodicBCs!(mesh, "1" => "2")

equation = EulerEquation{1}(div, 1.4)

# Q0 = Flou.vars_prim2cons((5.997, -98.5914, 0.0, 11_666.5), equation)
# Q1 = Flou.vars_prim2cons((1.0, 0.0, 0.0, 1.0), equation)
# ∂Ω = Dict(
#     "1" => EulerOutflowBC(),
#     "2" => EulerInflowBC(Q0),
#     "3" => EulerSlipBC(),
#     "4" => EulerSlipBC(),
#     "5" => EulerSlipBC(),
#     "6" => EulerSlipBC(),
# )
∂Ω = Dict()
# Q0 = Flou.vars_prim2cons((1.0, 2.0, 0.0, 1.0), equation)
# ∂Ω = Dict(
#     "Left" => EulerInflowBC(Q0),
#     "Bottom" => EulerSlipBC(),
#     "Right" => EulerOutflowBC(),
#     "Top" => EulerSlipBC(),
#     "Hole" => EulerSlipBC(),
# )
DG = DGSEM(mesh, std, equation, ∂Ω, numflux)

x0 = 0.5
sx = 0.1
h = 1.0
Q = StateVector{Float64}(undef, nvariables(equation), DG.dofhandler)
# for i in eachdof(DG)
#     x = DG.geometry.elements.coords[i][1]
#     ρ = 1.0 + Flou.gaussian_bump(x, 0.0, 0.0, x0, 0.0, 0.0, sx, 1.0, 1.0, h)
#     u = 1.0
#     p = 1.0
#     Q.data[i, :] = Flou.vars_prim2cons((ρ, u, p), equation)
# end
# Q = StateVector{Float64}(undef, nvariables(equation), DG.dofhandler)
# for i in eachdof(DG)
#     x = DG.geometry.elements.coords[i][1]
#     Q.data[i, :] .= x > 1.5 ? Q0 : Q1
# end
Q = StateVector{Float64}(undef, nvariables(equation), DG.dofhandler)
for i in eachdof(DG)
    P = 1.0 .+ rand(Float64, nvariables(equation)) .* 0.8
    Q.data[i, :] .= Flou.vars_prim2cons(P, equation)
end
# Q = StateVector{Float64}(undef, DG.dofhandler, DG.stdvec, nvariables(equation))
# for ie in eachelement(mesh)
#     local std = get_std(DG, ie)
#     for i in eachindex(std)
#         Q[ie][i, :] .= Q0
#     end
# end

display(DG)
println()

mb, mvals = get_monitor_callback(Float64, DG, :entropy)
sb = get_save_callback("../results/solution", save_steps)
cb = make_callback_list(mb, sb)

@info "Starting simulation..."

_, exetime = timeintegrate(Q.data, DG, solver, tf; save_everystep=false, alias_u0=true,
    adaptive=false, dt=Δt, callback=cb, progress=false, progress_steps=50)

@info "Elapsed time: $(exetime) s"
@info "Time per iteration and DOF: $(exetime / (tf/Δt) / ndofs(DG)) s"
