"""
    list_monitors(disc)

List the monitors available for the given discretization and equation.
"""
function list_monitors end

"""
    get_monitor(disc, name)

Get the monitor with the given name for the given discretization and equation.
"""
function get_monitor end

include("Hyperbolic/Euler.jl")
