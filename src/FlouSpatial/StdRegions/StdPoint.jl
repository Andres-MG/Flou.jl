struct StdPoint{RT} <: AbstractStdRegion{0}
    ω::Vector{RT}
    ωf::Vector{RT}
    ξ::Vector{SVector{1,RT}}
    ξf::Vector{SVector{1,RT}}
    ξe::Vector{SVector{1,RT}}
    ξc::Vector{SVector{1,RT}}
end

function StdPoint(ftype=Float64)
    return StdPoint(
        [one(ftype)],
        [one(ftype)],
        [SVector(zero(ftype))],
        [SVector(zero(ftype))],
        [SVector(zero(ftype))],
        [SVector(zero(ftype))],
    )
end

ndirections(::StdPoint) = 1
nvertices(::StdPoint) = 1

function tpdofs(::StdPoint, _)
    return (Colon(),)
end

function slave2master(i::Integer, _, ::StdPoint)
    return i
end

function master2slave(i::Integer, _, ::StdPoint)
    return i
end
