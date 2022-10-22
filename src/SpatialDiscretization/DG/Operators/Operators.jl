"""
    AbstractOperator

Must contain a `numflux` field specifing the numerical flux to use.
"""
abstract type AbstractOperator end

function volume_contribution!  end

function surface_contribution!  end

requires_subgrid(::AbstractOperator, _) = false

include("Divergence.jl")
include("Gradient.jl")
