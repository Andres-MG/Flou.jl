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
nelem_list = [2^i for i in 2:6]
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

@everywhere function l2error(q, t, dg, equation)
    (; geometry) = dg
    s = zero(eltype(q))
    for i in eachdof(dg)
        qe = solution(geometry.elements.coords[i], t, equation)
        s += sum(abs2, qe - q.dof[i])
    end
    # return sqrt(s / ndofs(dg))
    return sqrt(s)
end

@everywhere function run(nelem, order, ∇, tf, solver, equation, cfl)
    # Construct the 1D mesh
    mesh = CartesianMesh{1,Float64}(-1, 1, nelem)
    apply_periodicBCs!(mesh, "1" => "2")

    # Reference 1D element
    std = FRStdSegment{Float64}(order + 1, GL(), :DGSEM, nvariables(equation))

    # Spatial discretization
    dg = FR(mesh, std, equation, ∇, ())

    # Initial condition
    Q = StateVector{nvariables(equation),Float64}(undef, dg.dofhandler)
    for i in eachdof(dg)
        x = dg.geometry.elements.coords[i]
        Q.dof[i] = solution(x, 0.0, equation)
    end

    # Time integration
    timeintegrate(
        Q, dg, equation, solver, tf;
        alias_u0=true, adaptive=false, callback=cfl, dt=1.0,
    )

    # Compute the L2 error
    return l2error(Q, tf, dg, equation)
end

#==========================================================================================#
#                                    Parallel execution                                    #

tasks = [(nelem, order, div) for nelem in nelem_list, order in order_list, div in ∇_list]
error = pmap((t) -> run(t..., tf, solver, equation, cfl_callback), tasks)

# Error[i, j, k] is the L₂ difference between the two formulations and the exact solution
# for a case with:
#   - `nelem_list[i]` elements in the mesh
#   - order `order_list[j]` polynomial approximation
#   - `∇_list[k]` divergence formulation

#==========================================================================================#
#                                      Post-processing                                     #

using Plots
using LaTeXStrings

Δx = 2 ./ nelem_list
Δx = repeat(Δx, 1, length(order_list))

# Plot for the split form of J. Chan
serieslabel = fill("", 1, length(order_list))
serieslabel[1] = "Original"
plot(
     Δx, error[:, :, 1],
     linecolor=:black, linewidth=1,
     markercolor=:black, markersize=4, markershape=:square,
     label=serieslabel,
)

# Plot for the subcell version
serieslabel = fill("", 1, length(order_list))
serieslabel[1] = "Sub-cell"
plot!(
     Δx, error[:, :, 2],
     linestyle=:dash, linecolor=:grey, linewidth=1,
     markercolor=:grey, markersize=4, markershape=:circle,
     label=serieslabel,
)

# Plot combining the two results
xticks = [-1.5, -1.25, -1, -0.75, -0.5]
xtickslabels = [L"10^{%$i}" for i in xticks]
yticks = [-2, -4, -6, -8, -10]
ytickslabels = [L"10^{%$i}" for i in yticks]
plot!(
    xaxis=:log, yaxis=:log,
    xticks=(exp10.(xticks), xtickslabels),
    yticks=(exp10.(yticks), ytickslabels),
    xlabel=L"Mesh size, $h$", ylabel=L"$L_2$ error",
    fontfamily="Computer Modern",
    framestyle=:box,
    size=(350, 250),
    legend=:bottomright,
)

savefig("convergence.pdf")
