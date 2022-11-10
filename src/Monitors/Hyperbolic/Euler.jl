function list_monitors(::DGSEM, ::EulerEquation)
    return (:entropy,)
end

function get_monitor(dg::DGSEM, equation::EulerEquation, name::Symbol, _)
    if name == :entropy
        return entropy_monitor(dg, equation)
    else
        error("Unknown monitor '$(name)'.")
    end
end

# TODO: paralellization does not work, why?
function entropy_monitor(dg::DGSEM, ::EulerEquation)
    monitor = (Q_, dg, equation) -> begin
        Q = StateVector{nvariables(equation)}(Q_, dg.dofhandler)
        svec = dg.std.cache.scalar[1][1]
        s = zero(datatype(Q))
        for ie in eachelement(dg)
            @inbounds for i in eachindex(svec)
                svec[i] = math_entropy(Q[ie][i], equation)
            end
            s += integrate(svec, dg.geometry.elements[ie])
        end
        return s
    end
    return (eltype(dg), monitor)
end
