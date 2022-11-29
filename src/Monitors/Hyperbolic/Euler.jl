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

function entropy_monitor(dg::DGSEM, ::EulerEquation)
    monitor = (Q, dg, equation) -> begin
        svec = dg.std.cache.scalar[1][1]
        s = zero(eltype(Q))
        @flouthreads for ie in eachelement(dg)
            @inbounds for i in eachindex(svec)
                svec[i] = math_entropy(Q.element[ie][i], equation)
            end
            s += integrate(svec, dg.geometry.elements[ie])
        end
        return s
    end
    return (datatype(dg), monitor)
end
