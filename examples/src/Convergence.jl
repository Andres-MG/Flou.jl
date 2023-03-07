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

using Distributed

@everywhere using Pkg
@everywhere Pkg.activate(".")

@everywhere using Flou
@everywhere using OrdinaryDiffEq

# Common parameters
cfl = 0.125
tf = 0.7
solver = CarpenterKennedy2N54(williamson_condition=false)

nelem_list = [2^i for i in 2:6]
order_list = 1:5

∇ = SplitDivOperator(
    LxFNumericalFlux(ChandrasekharAverage(), 1.0),
)

equation = EulerEquation{1}(1.4)

cfl_callback = get_cfl_callback(cfl, 1.0)

# Per-process simulation
@everywhere function solution(x, t, eq)
    return Flou.vars_prim2cons((2 + sin((x[1] - t)π), 1, 1), eq)
end

@everywhere function l2error(q, t, dg, equation)
    (; geometry) = dg
    s = zero(Flou.datatype(q))
    for (i, xi) in enumerate(geometry.elements.coords)
        qe = solution(xi, t, equation)
        s += sum(abs2, qe - q.dof[i])
    end
    return sqrt(s / ndofs(dg))
end

@everywhere function run(nelem, order, quad, ∇, tf, solver, equation, cfl)
    # Solve the problem
    std = FRStdSegment{Float64}(order + 1, quad, :DGSEM, nvariables(equation))
    mesh = CartesianMesh{1,Float64}(-1, 1, nelem)
    apply_periodicBCs!(mesh, "1" => "2")
    dg = FR(mesh, std, equation, ∇, ())
    Q = StateVector{nvariables(equation),Float64}(undef, dg.dofhandler)
    for i in eachdof(dg)
        x = dg.geometry.elements.coords[i]
        Q.dof[i] = solution(x, 0.0, equation)
    end
    timeintegrate(
        Q, dg, equation, solver, tf;
        alias_u0=true, adaptive=false, callback=cfl, dt=1.0,
    )

    # Compute the L2 error
    return l2error(Q, tf, dg, equation)
end

# Parallel execution
tasks = [(nelem, order, quad)
    for nelem in nelem_list,
    order in order_list,
    quad in (GLL(), GL())
]
error = pmap((t) -> run(t..., ∇, tf, solver, equation, cfl_callback), tasks)

# Post-processing
using Plots
using LaTeXStrings

Δx = 2 ./ nelem_list
Δx = repeat(Δx, 1, length(order_list))

orderlabel = fill("", 1, length(order_list))
orderlabel[1] = "GLL"
plot(
     Δx, error[:, :, 1],
     linestyle=:dash, linecolor=:cyan3, linewidth=2,
     markercolor=:cyan3, markersize=4, markershape=:circle,
     xaxis=:log, yaxis=:log,
     label=orderlabel,
)

orderlabel = fill("", 1, length(order_list))
orderlabel[1] = "GL"
plot!(
     Δx, error[:, :, 2],
     linecolor=:goldenrod3, linewidth=2,
     markercolor=:goldenrod3, markersize=4, markershape=:square,
     label=orderlabel, legend=:bottomright,
)

plot!(
    xaxis=:log, yaxis=:log, yticks=[1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12],
    xlabel=L"Mesh size, $h$", ylabel=L"$L_2$ error",
    fontfamily="Computer Modern",
    framestyle=:box,
    title="Split-form operator",
)

savefig("convergence.pdf")
