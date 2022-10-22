abstract type AbstractEquation{ND,NV} end

ndims(::AbstractEquation{ND}) where {ND} = ND
nvariables(::AbstractEquation{ND,NV}) where {ND,NV} = NV

eachdim(e::AbstractEquation) = Base.OneTo(ndims(e))
eachvariable(e::AbstractEquation) = Base.Base.OneTo(nvariables(e))

function variablenames end

function volumeflux end

"""
    rhs!(dQ, Q, p::Tuple{AbstractSpatialDiscretization,AbstractEquation}, time)

Evaluate the right-hand side (spatial term) of the ODE.
"""
function rhs! end
