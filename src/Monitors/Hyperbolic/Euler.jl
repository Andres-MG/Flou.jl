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
    monitor = (Q_, dg, equation) -> begin
        rt = eltype(Q_)
        Q = StateVector(Q_, dg.dofhandler)
        s = zero(rt)
        svec = Vector{rt}(undef, get_std(dg, 1) |> ndofs)
        @inbounds for ie in eachelement(dg)
            std = get_std(dg, ie)
            resize!(svec, ndofs(std))
            for i in eachindex(std)
                svec[i] = math_entropy(view(Q[ie], i, :), equation)
            end
            s += integrate(svec, dg.geometry.elements[ie])
        end
        return s
    end
    return (eltype(dg), monitor)
end
