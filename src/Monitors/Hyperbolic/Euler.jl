function list_monitors(::DGSEM{EulerEquation})
    return (:entropy,)
end

function get_monitor(dg::DGSEM{<:EulerEquation}, name::Symbol)
    if name == :entropy
        return entropy_monitor(dg)
    else
        error("Unknown monitor '$(name)'.")
    end
end

function entropy_monitor(dg::DGSEM{<:EulerEquation})
    monitor = (Q_, dg) -> begin
        rt = eltype(Q_)
        Q = StateVector(Q_, dg.dofhandler)
        s = zero(rt)
        svec = Vector{rt}(undef, get_std(dg, 1) |> ndofs)
        @inbounds for ie in eachelement(dg)
            std = get_std(dg, ie)
            resize!(svec, ndofs(std))
            for i in eachindex(std)
                svec[i] = math_entropy(view(Q[ie], i, :), dg.equation)
            end
            s += integrate(svec, dg.geometry.elements[ie])
        end
        return s
    end
    return (eltype(dg), monitor)
end
