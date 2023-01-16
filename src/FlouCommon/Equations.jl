abstract type AbstractEquation{ND,NV} end

"""
    spatialdim(equation)

Return the spatial dimension of `equation`.
"""
function spatialdim(::AbstractEquation{ND}) where {ND}
    return ND
end

"""
    eachdim(equation)

Return a range that covers the spatial dimensions of `equation`.
"""
function eachdim(eq::AbstractEquation)
    return Base.OneTo(spatialdim(eq))
end

"""
    nvariables(equation)

Return the number of variables of `equation`.

See also [`eachvariable`](@ref), [`variablenames`](@ref).
"""
function nvariables(::AbstractEquation{ND,NV}) where {ND,NV}
    return NV
end

"""
    eachvariable(equation)

Return a range that covers all the variables of `equation`.

See also [`nvariables`](@ref), [`variablenames`](@ref).
"""
function eachvariable(e::AbstractEquation)
    return Base.OneTo(nvariables(e))
end

"""
    variablenames(equation, unicode=false)

Return the names of the variables in `equation`. If `unicode=true` the names may contain
unicode symbols.

See also [`nvariables`](@ref), [`eachvariable`](@ref).
"""
function variablenames(::AbstractEquation, ::Bool=false) end

#==========================================================================================#
#                                     Implementations                                      #

abstract type HyperbolicEquation{ND,NV} <: AbstractEquation{ND,NV} end
include("LinearAdvection.jl")
include("Burgers.jl")
include("KPP.jl")
include("Euler.jl")

abstract type GradientEquation{ND,NV} <: AbstractEquation{ND,NV} end
include("Gradient.jl")
