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

#==========================================================================================#
#                                        Parameters                                        #

cfl = 0.125
tf = 0.7
solver = CarpenterKennedy2N54(williamson_condition=false)

# Initial condition and solution since this is only advection
@everywhere function solution(x, t, eq)
    return Flou.vars_prim2cons((2 + sinpi(x[1] - t), 1, 1), eq)
end

equation = EulerEquation{1}(1.4)

numflux = LxFNumericalFlux(
    ChandrasekharAverage(),
    1.0,
)

# Test space
k_list = 1:5
nelem_list = [2 * 2^k for k in k_list]   # The mesh has length = 2 and Δx = 1 / 2^k
order_list = 1:5
∇_list = (
    SplitDivOperator(
        numflux,
    ),
    HybridDivOperator(
        numflux,
        1.0,
    ),
)

#==========================================================================================#
#                                      Individual runs                                     #

cfl_callback = get_cfl_callback(cfl, 1.0)

@everywhere function l2error(q1, q2)
    s = mapreduce((x, y) -> (x - y) ^ 2, +, q1, q2)
    return sqrt(s)
end

@everywhere function run(nelem, order, ∇, tf, solver, equation, cfl)
    # Construct the mesh and reference element
    mesh = CartesianMesh{1,Float64}(-1, 1, nelem)
    apply_periodicBCs!(mesh, "1" => "2")

    std = FRStdSegment{Float64}(order + 1, GL(), :DGSEM, nvariables(equation))

    # First divergence operator
    dg = FR(mesh, std, equation, ∇[1], ())

    Q1 = StateVector{nvariables(equation),Float64}(undef, dg.dofhandler)
    for i in eachdof(dg)
        x = dg.geometry.elements.coords[i]
        Q1.dof[i] = solution(x, 0.0, equation)
    end

    timeintegrate(
        Q1, dg, equation, solver, tf;
        alias_u0=true, adaptive=false, callback=cfl, dt=1.0,
    )

    # Second divergence operator
    dg = FR(mesh, std, equation, ∇[2], ())

    Q2 = StateVector{nvariables(equation),Float64}(undef, dg.dofhandler)
    for i in eachdof(dg)
        x = dg.geometry.elements.coords[i]
        Q2.dof[i] = solution(x, 0.0, equation)
    end

    timeintegrate(
        Q2, dg, equation, solver, tf;
        alias_u0=true, adaptive=false, callback=cfl, dt=1.0,
    )

    # Compute the L2 error
    return l2error(Q1, Q2)
end

#==========================================================================================#
#                                    Parallel execution                                    #

tasks = [(nelem, order) for nelem in nelem_list, order in order_list]
error = pmap((t) -> run(t..., ∇_list, tf, solver, equation, cfl_callback), tasks)

# Error[i, j] is the L₂ difference between the two formulations for a case with:
#   - `nelem_list[i]` elements in the mesh
#   - order `order_list[j]` polynomial approximation

#==========================================================================================#
#                                      Post-processing                                     #

using Printf

scaling = 1e-13
scaling_str = "\$10^{13}\$"

tophead = map(x -> "\$k\$ = " * @sprintf("%d", x), k_list)
lefthead = map(x -> "\$N\$ = " * @sprintf("%d", x), order_list)
error_str = map(x -> @sprintf("%7.5f", x), error / scaling)

open("errors.tex", "w") do f
    println(f, "\\documentclass{standalone}")
    println(f, "\\usepackage{booktabs}\n")

    println(f, "\\begin{document}\n")

    println(f, "\\begin{tabular}{@{} l *{$(length(k_list))}{c} @{}}")

    println(f, "    \\toprule")
    println(f, "    & ", join(tophead, " & "), " \\\\")
    println(f, "    \\midrule")
    for (p, hr) in enumerate(lefthead)
        println(f, "    ", hr, " & ", join(error_str[:, p], " & "), " \\\\")
    end
    println(f, "    \\midrule")
    println(f, "    \\multicolumn{$(1 + length(k_list))}{l}{",
        "\\footnotesize \$^*\$values are scaled by $scaling_str} \\\\")
    println(f, "    \\bottomrule")
    println(f, "\\end{tabular}\n")

    println(f, "\\end{document}")
end

Base.run(`pdflatex errors.tex`)
Base.rm.(["errors.aux", "errors.log"]);