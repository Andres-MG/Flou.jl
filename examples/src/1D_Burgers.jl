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

std = FRStdSegment{Float64}(GLL(4), :DGSEM, nvariables(equation))
mesh = CartesianMesh{1,Float64}(0, 1, 20)
apply_periodicBCs!(mesh, "1" => "2")

∇ = StrongDivOperator(LxFNumericalFlux(StdAverageNumericalFlux(), 1.0))
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
