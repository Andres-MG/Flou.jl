abstract type AbstractEquation{NV} end

nvariables(::AbstractEquation{NV}) where {NV} = NV
eachvariable(e::AbstractEquation) = Base.Base.OneTo(nvariables(e))

requires_subgrid(::AbstractEquation) = false

function variablenames end

function rotate2face! end
function rotate2phys! end

function evaluate! end


#==========================================================================================#
#                                   Boundary Conditions                                    #

abstract type AbstractBC end

function state_BC! end

function applyBC!(Qext, Qint, coords, n, t, b, time, eq, bc)
    @boundscheck begin
        size(Qext, 1) == size(Qint, 1) && size(Qext, 2) == size(Qint, 2) ||
            throw(ArgumentError("Qext and Qint must have the same dimensions."))
    end
    for (i, Qi) in enumerate(eachrow(Qint))
        copy!(view(Qext, i, :), Qi)
        stateBC!(view(Qext, i, :), coords[i], n[i], t[i], b[i], time, eq, bc)
    end
    return nothing
end

struct DirichletBC{QF} <: AbstractBC
    Q!::QF     # Q!(Q, x, n, t, b, time, eq)  in/out
end

function stateBC!(Q, x, n, t, b, time, eq, bc::DirichletBC)
    bc.Q!(Q, x, n, t, b, time, eq)
    return nothing
end

#==========================================================================================#
#                                     Equation Systems                                     #

include("Hyperbolic.jl")
