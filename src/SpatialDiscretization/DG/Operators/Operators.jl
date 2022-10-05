abstract type AbstractOperator end

function volume_contribution!  end

function surface_contribution!  end

requires_subgrid(::AbstractOperator) = false

include("Divergence.jl")
