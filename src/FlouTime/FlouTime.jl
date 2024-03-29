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

module FlouTime

using ..FlouCommon
using ..FlouBiz
using Printf: Printf, @sprintf
using OrdinaryDiffEq: OrdinaryDiffEq, ODEProblem, solve, DiscreteCallback, CallbackSet

export MonitorOutput
export timeintegrate, make_callback_list
export get_save_callback, get_cfl_callback, get_monitor_callback, get_limiter_callback

"""
    timeintegrate(rhs::Function, Q0::AbstractArray, disc::AbstractSpatialDiscretization,
                  tfinal::Number; kwargs)

Wrapper around the `ODEProblem` and `solve` functions of the `OrdinaryDiffEq.jl` module.
Accepts the same keyword arguments, `kwargs`, as `OrdinayrDiffEq.solve`.
"""
function timeintegrate(Q0, disc, equation, solver, tfinal; kwargs...)
    problem = ODEProblem(rhs!, Q0, tfinal, EquationConfig(disc, equation))
    exetime = @elapsed begin
        sol = try
            solve(problem, solver; kwargs...)
        catch e
            if isa(e, TaskFailedException)
                if isa(e.task.result, DomainError)
                    @error "Simulation crashed!"
                else
                    throw(e)
                end
            elseif isa(e, DomainError)
                @error "Simulation crashed!"
            else
                throw(e)
            end
        end
    end
    return (sol, exetime)
end

#==========================================================================================#
#                                        Callbacks                                         #

function _is_iter_selected(iterations=true)
    return if iterations == true
        (_, _, _) -> true
    else
        (_, _, integrator) -> integrator.iter in iterations
    end
end

function _save_function(basename)
    return (integrator) -> begin
        (; disc, equation) = integrator.p

        filename = @sprintf("%s_%010d.hdf", basename, integrator.iter)
        file = open_for_write(filename, disc)
        add_fielddata!(file, [integrator.t], "Time")
        add_solution!(file, integrator.u, disc, equation)
        close_file!(file)
        OrdinaryDiffEq.u_modified!(integrator, false)

        @info @sprintf("Saved solution at t=%.7g in `%s`", integrator.t, filename)
    end
end

function get_save_callback(basename; iter=true)
    condition = _is_iter_selected(iter)
    affect = _save_function(basename)
    initialize = (_, _, _, integrator) -> begin
        affect(integrator)
    end
    return DiscreteCallback(condition, affect, initialize=initialize,
                            save_positions=(false, false))
end

function get_cfl_callback(cfl, Δtmax=Inf; iter=true)
    condition = _is_iter_selected(iter)
    affect = (integrator) -> begin
        (; disc, equation) = integrator.p
        Δt = min(get_max_dt(integrator.u, disc, equation, cfl), Δtmax)
        OrdinaryDiffEq.set_proposed_dt!(integrator, Δt)
        OrdinaryDiffEq.u_modified!(integrator, false)
    end
    initialize = (_, _, _, integrator) -> begin
        affect(integrator)
    end
    return DiscreteCallback(condition, affect, initialize=initialize,
                            save_positions=(false, false))
end

struct MonitorOutput{RT<:Real,RV}
    time::Vector{RT}
    iter::Vector{Int}
    value::RV
end

function get_monitor_callback(
    timetype::Type{<:Real},
    valuetype::Type{<:Number},
    disc::AbstractSpatialDiscretization,
    equation::AbstractEquation,
    name::Symbol,
    p=nothing;
    iter=true
)
    _monitorfunc = get_monitor(disc, equation, name, p)
    monitor = MonitorOutput(timetype[], Int[], valuetype[])
    condition = _is_iter_selected(iter)
    affect = (integrator) -> begin
        (; disc, equation) = integrator.p
        value = _monitorfunc(integrator.u, disc, equation)
        push!(monitor.time, integrator.t)
        push!(monitor.iter, integrator.iter)
        push!(monitor.value, value)
        OrdinaryDiffEq.u_modified!(integrator, false)
    end
    return DiscreteCallback(condition, affect, save_positions=(false, false)), monitor
end

function get_limiter_callback(
    disc::AbstractSpatialDiscretization,
    equation::AbstractEquation,
    name::Symbol,
    p=nothing,
)
    _limiterfunc = get_limiter(disc, equation, name, p)
    return (_, integrator, _, _) -> begin
        (; disc, equation) = integrator.p
        _limiterfunc(integrator.u, disc, equation)
        OrdinaryDiffEq.u_modified!(integrator, true)
    end
end

function make_callback_list(callbacks...)
    return if isempty(callbacks)
        nothing
    elseif length(callbacks) == 1
        callbacks[1]
    else
        CallbackSet(callbacks...)
    end
end

end # FlouTime
