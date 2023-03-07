# Copyright (C) 2023 Andrés Mateo Gabín
#
# This file is part of Flou.jl.
#
# Flou.jl is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# Flou.jl is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Flou.jl. If
# not, see <https://www.gnu.org/licenses/>.

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
tf = 0.1
save_steps = range(0, 1000, 11)
solver = ORK256(williamson_condition=false)

equation = EulerEquation{3}(1.4)

std = FRStdHex{Float64}(4, GLL(), :DGSEM, nvariables(equation))
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
    MatrixDissipation(ChandrasekharAverage(), 1.0)
)
dg = FR(mesh, std, equation, ∇, ∂Ω)

Q = StateVector{nvariables(equation),Float64}(undef, dg.dofhandler)
for i in eachdof(Q)
    Q.dof[i] = Q0
end

display(dg)
println()

mb, mvals = get_monitor_callback(Float64, dg, equation, :entropy)
sb = get_save_callback("../results/solution"; iter=save_steps)
cb = make_callback_list(mb, sb)

@info "Starting simulation..."

_, exetime = timeintegrate(
    Q, dg, equation, solver, tf;
    save_everystep=false, alias_u0=true, adaptive=false, dt=Δt, callback=cb,
    progress=true, progress_steps=5,
)

@info "Elapsed time: $(exetime) s"
@info "Time per iteration and DOF: $(exetime / (tf/Δt) / ndofs(dg)) s"
