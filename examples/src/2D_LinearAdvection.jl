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
Δt = 1e-3
tf = 1.0
save_steps = range(0, 1000, 21)
solver = ORK256(williamson_condition=false)

equation = LinearAdvection(2.0, -1.0)

basis = LagrangeBasis(:GL, 4)
rec = basis |> DGSEMrec
std = StdQuad(basis, rec, nvariables(equation))
mesh = CartesianMesh{2,Float64}((0, 0), (1, 1), (10, 10))
apply_periodicBCs!(mesh, "1" => "2", "3" => "4")

∇ = StrongDivOperator(
    LxF(
        StdAverage(),
        1.0,
    ),
)
fr = MultielementDisc(mesh, std, equation, ∇, ())

x0 = y0 = 0.5
sx = sy = 0.1
h = 1.0
Q = GlobalStateVector{nvariables(equation)}(undef, fr.dofhandler)
for i in eachdof(fr)
    x, y = fr.geometry.elements.coords[i]
    Q.dofsmut[i][1] = Flou.gaussian_bump(x, y, x0, y0, sx, sy, h)
end

display(fr)
println()

sb = get_save_callback("../results/solution"; iter=save_steps)

@info "Starting simulation..."

_, exetime = timeintegrate(
    Q.data, fr, equation, solver, tf;
    save_everystep=false, alias_u0=true, adaptive=false, dt=Δt, callback=sb,
)

@info "Elapsed time: $(exetime) s"
@info "Time per iteration and DOF: $(exetime / (tf/Δt) / ndofs(dg)) s"
