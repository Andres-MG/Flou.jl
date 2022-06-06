using Test
using Flou
using OrdinaryDiffEq

include("tests.jl")

@testset "Advection" begin
    @testset "1D" begin
        sol = Advection1D()
        @test sol.u[end] ≈ sol.u[1] rtol=1e-3
    end
    @testset "2D" begin
        sol = Advection2D()
        @test sol.u[end] ≈ sol.u[1] rtol=1e-3
    end
end
