using Test
using Flou
using OrdinaryDiffEq
using StaticArrays

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

@testset "Euler" begin
    @testset "Sod tube" begin
        sol = SodTube1D()
        @test minimum(sol.u[end]) ≈ -0.1662939230897596 rtol=1e-7
        @test maximum(sol.u[end]) ≈ 254.90504152149907  rtol=1e-7
    end
end
