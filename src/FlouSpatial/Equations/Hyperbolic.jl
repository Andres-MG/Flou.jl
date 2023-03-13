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

struct HyperbolicCache{NV,RT} <: AbstractCache
    Qf::FaceStateVector{NV,RT}
    Fn::FaceStateVector{NV,RT}
end

function construct_cache(
    ftype::Type,
    dofhandler::DofHandler,
    equation::HyperbolicEquation,
)
    Qf = FaceStateVector{nvariables(equation)}(undef, dofhandler, ftype)
    Fn = FaceStateVector{nvariables(equation)}(undef, dofhandler, ftype)
    return HyperbolicCache(Qf, Fn)
end

function FlouCommon.rhs!(
    _dQ::Matrix,
    _Q::Matrix,
    p::EquationConfig{<:MultielementDisc,<:HyperbolicEquation},
    time::Real,
)
    # Unpack
    (; disc, equation) = p
    cache = disc.cache

    dQ = GlobalStateVector{nvariables(equation)}(_dQ, disc.dofhandler)
    Q = GlobalStateVector{nvariables(equation)}(_Q, disc.dofhandler)

    # Restart time derivative
    fill!(dQ, zero(datatype(dQ)))

    # Project Q to faces
    project2faces!(cache.Qf, Q, disc)

    # Volume flux
    volume_contribution!(dQ, Q, disc, equation, disc.operators[1])

    # Boundary conditions
    applyBCs!(cache.Qf, disc, equation, time)

    # Interface fluxes
    interface_fluxes!(cache.Fn, cache.Qf, disc, equation, disc.operators[1].numflux)

    # Surface contribution
    surface_contribution!(dQ, Q, disc, equation, disc.operators[1])

    # Apply mass matrix
    apply_massmatrix!(dQ, disc)

    # Add source term
    apply_sourceterm!(dQ, Q, disc, time)

    return nothing
end

function volume_contribution!(dQ, Q, disc, equation::HyperbolicEquation, operator)
    @flouthreads for ie in eachelement(disc)
        @inbounds volume_contribution!(dQ, Q, ie, disc.std, disc, equation, operator)
    end

    return nothing
end

function surface_contribution!(dQ, Q, disc, equation::HyperbolicEquation, operator)
    @flouthreads for ie in eachelement(disc)
        @inbounds surface_contribution!(dQ, Q, ie, disc.std, disc, equation, operator)
    end
    return nothing
end
