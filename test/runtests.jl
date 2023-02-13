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
        @test maximum(sol.u[end]) ≈ 254.90504152149907 rtol=1e-7
    end
    @testset "Shockwave" begin
        sol = Shockwave2D()
        @test minimum(sol.u[end]) ≈ -4.680030848090326e-13 rtol=1e0
        @test maximum(sol.u[end]) ≈ 12.977466260673845 rtol=1e-7
    end
end
