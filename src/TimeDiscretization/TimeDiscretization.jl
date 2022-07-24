"""
    solve(rhs::Function, Q0::StateVector, disc::AbstractSpatialDiscretization,
          tfinal::Number; kwargs)

Wrapper around the `ODEProblem` and `solve` functions of the `OrdinaryDiffEq.jl` module.
Accepts the same keyword arguments, `kwargs`, as `OrdinayrDiffEq.solve`.
"""
function integrate(Q0, disc, solver, tfinal; kwargs...)
    problem = ODEProblem(rhs!, Q0.raw, tfinal, disc)
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
        (; dofhandler, stdvec, equation) = disc

        filename = @sprintf("%s_%010d.hdf", basename, integrator.iter)
        file = open_for_write(filename, disc)
        add_fielddata!(file, [integrator.t], "Time")

        Q = StateVector(integrator.u, dofhandler, stdvec, nvariables(equation))
        add_solution!(file, Q, disc)

        close_file!(file)
        msg = @sprintf("Saved solution at t=%.7g in `%s`", integrator.t, filename)
        @info msg
        return nothing
    end
end

function get_save_callback(basename, iterations)
    condition = _is_iter_selected(iterations)
    affect = _save_function(basename)
    initialize = (_, _, _, integrator) -> begin
        affect(integrator)
    end
    return DiscreteCallback(condition, affect, initialize=initialize)
end

function get_monitors_callback(rt, monitors...)
    # Entropy monitor
    if monitors[1] == :entropy
        saved_vals = SavedValues(rt, rt)
        save_func = (u, _, integrator) -> begin
            disc = integrator.p
            (; dofhandler, stdvec, equation) = disc
            s = zero(eltype(rt))
            Q = StateVector(u, dofhandler, stdvec, nvariables(equation))
            s = zero(rt)
            @inbounds for ir in eachregion(Q), ie in eachelement(Q, ir)
                std = stdvec[ir]
                for i in eachindex(std)
                    s += math_entropy(view(Q[ir], i, :, ie), equation) * std.ω[i]
                end
            end
            return s
        end
    end
    return (SavingCallback(save_func, saved_vals), saved_vals)
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
