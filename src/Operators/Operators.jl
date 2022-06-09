abstract type AbstractOperator end

requires_subgrid(::AbstractOperator) = false

include("Divergence.jl")
