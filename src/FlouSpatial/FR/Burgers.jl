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

function rotate2face(Qf, _, ::BurgersEquation)
    return SVector(Qf[1])
end

function rotate2phys(Qrot, _, ::BurgersEquation)
    return SVector(Qrot[1])
end

function numericalflux(Ql, Qr, n, ::BurgersEquation, ::StdAverageNumericalFlux)
    return SVector((Ql[1]^2 + Qr[1]^2) / 4 * n[1])
end

function numericalflux(Ql, Qr, n, eq::BurgersEquation, nf::LxFNumericalFlux)
    # Average
    Fn = numericalflux(Ql, Qr, n, eq, nf.avg)

    # Dissipation
    λ = max(abs(Ql[1]), abs(Qr[1]))
    return SVector(Fn[1] + λ * (Ql[1] - Qr[1]) / 2 * nf.intensity)
end
