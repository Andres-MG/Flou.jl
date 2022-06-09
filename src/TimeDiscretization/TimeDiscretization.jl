"""
    solve(rhs::Function, Q0::StateVector, disc::AbstractSpatialDiscretization,
          tfinal::Number; kwargs)

Wrapper around the 'ODEProblem' and 'solve' functions of the 'OrdinaryDiffEq.jl' module.
Accepts the same keyword arguments, 'kwargs', as 'OrdinayrDiffEq.solve'.
"""
function integrate(Q0, disc, solver, tfinal; kwargs...)
    problem = ODEProblem(evaluate!, Q0.raw, tfinal, disc)
    exetime = @elapsed begin
        sol = try
            solve(problem, solver; kwargs...)
        catch e
            if isa(e, TaskFailedException)
                if isa(e.task.result, DomainError)
                    @error "Simulation crashed!"
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

function get_save_callback(basename, tstops)
    f = function save_callback(integrator)
        (; dofhandler, stdvec, equation) = integrator.p
        Q = StateVector(integrator.u, dofhandler, stdvec, nvariables(equation))
        filename = @sprintf("%s_%010d.hdf", basename, integrator.iter)
        save(filename, Q, integrator.p)
        @info "Saved solution at t=$(integrator.t) in '$(filename)'"
        return nothing
    end
    return PresetTimeCallback(tstops, f)
end
