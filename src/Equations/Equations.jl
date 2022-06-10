#==========================================================================================#
#                                   Hyperbolic equations                                   #

abstract type HyperbolicEquation{NV} <: AbstractEquation{NV} end

include("Hyperbolic/Hyperbolic.jl")
include("Hyperbolic/LinearAdvection.jl")
include("Hyperbolic/Burgers.jl")
include("Hyperbolic/Euler.jl")
include("Hyperbolic/KPP.jl")

#==========================================================================================#
#                                    Elliptic equations                                    #

abstract type EllipticEquation{NV} <: AbstractEquation{NV} end

include("Elliptic/Elliptic.jl")
