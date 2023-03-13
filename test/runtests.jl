# Copyright (C) 2023 Andrés Mateo Gabín
#
# This file is part of Flou.jl.
#
# Flou.jl is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# Flou.jl is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Flou.jl. If
# not, see <https://www.gnu.org/licenses/>.

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
