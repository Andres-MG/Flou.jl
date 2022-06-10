abstract type AbstractOperator end

requires_subgrid(::AbstractOperator) = false

function volume_contribution!  end

function surface_contribution!  end

include("Divergence.jl")
