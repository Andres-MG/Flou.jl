#==========================================================================================#
#                                   Hyperbolic equations                                   #

abstract type HyperbolicEquation{ND,NV} <: AbstractEquation{ND,NV} end

include("Hyperbolic/Hyperbolic.jl")
include("Hyperbolic/LinearAdvection.jl")
include("Hyperbolic/Burgers.jl")
include("Hyperbolic/Euler.jl")
include("Hyperbolic/KPP.jl")

#==========================================================================================#
#                                     Gradient equation                                    #

struct GradientEquation{ND,NV} <: AbstractEquation{ND,NV} end

include("Gradient/Gradient.jl")
