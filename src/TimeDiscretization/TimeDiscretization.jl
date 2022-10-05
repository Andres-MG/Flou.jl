"""
    timeintegrate(rhs::Function, Q0::AbstractArray, disc::AbstractSpatialDiscretization,
                  tfinal::Number; kwargs)

Wrapper around the `ODEProblem` and `solve` functions of the `OrdinaryDiffEq.jl` module.
Accepts the same keyword arguments, `kwargs`, as `OrdinayrDiffEq.solve`.
"""
function timeintegrate(Q0, disc, solver, tfinal; kwargs...)
    problem = ODEProblem(rhs!, Q0, tfinal, disc)
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

function _is_iter_selected(iterations)
    return (_, _, integrator) -> integrator.iter in iterations
end

function _save_function(basename)
    return (integrator) -> begin
        disc = integrator.p

        filename = @sprintf("%s_%010d.hdf", basename, integrator.iter)
        file = open_for_write(filename, disc)
        add_fielddata!(file, [integrator.t], "Time")
        add_solution!(file, integrator.u, disc)
        close_file!(file)

        @info @sprintf("Saved solution at t=%.7g in `%s`", integrator.t, filename)
        return nothing
    end
end

function get_save_callback(basename, iterations=1:typemax(Int))
    condition = _is_iter_selected(iterations)
    affect = _save_function(basename)
    initialize = (_, _, _, integrator) -> begin
        affect(integrator)
    end
    return DiscreteCallback(condition, affect, initialize=initialize)
end

struct MonitorOutput{RT<:Real,RV}
    time::Vector{RT}
    iter::Vector{Int}
    value::RV
end

function get_monitor end

function get_monitor_callback(
    timetype::Type{<:Real},
    disc::AbstractSpatialDiscretization,
    name::Symbol,
    iterations=1:typemax(Int),
)
    valuetype, monitorfunc_ = get_monitor(disc, name)
    monitor = MonitorOutput(timetype[], Int[], valuetype[])
    condition = _is_iter_selected(iterations)
    affect = (integrator) -> begin
        value = monitorfunc_(integrator.u, disc)
        push!(monitor.time, integrator.t)
        push!(monitor.iter, integrator.iter)
        push!(monitor.value, value)
        return nothing
    end
    return (DiscreteCallback(condition, affect), monitor)
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
