abstract type AbstractEquation{NV} end

nvariables(::AbstractEquation{NV}) where {NV} = NV

eachvariable(e::AbstractEquation) = Base.Base.OneTo(nvariables(e))

function variablenames end

function volumeflux end

"""
    rhs!(dQ, Q, p::Tuple{AbstractSpatialDiscretization,AbstractEquation}, time)

Evaluate the right-hand side (spatial term) of the ODE.
"""
function rhs! end
