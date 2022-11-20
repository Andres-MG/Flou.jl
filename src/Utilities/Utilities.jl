function gaussian_bump(x, y, z, x0, y0, z0, sx, sy, sz, h)
    return h * exp(-(x-x0)^2/(2sx^2) - (y-y0)^2/(2sy^2) - (z-z0)^2/(2sz^2))
end

function logarithmic_mean(al, ar)
    ξ = al / ar
    f = (ξ - 1) / (ξ + 1)
    u = f^2
    F = if u < convert(typeof(u), 0.01)
        1 + u/3 + u^2/5 + u^3/7
    else
        log(ξ) / 2f
    end
    return (al + ar) / 2F
end

macro flouthreads(expr)
    # esc(quote
    #     return if Threads.nthreads() == 1
    #         $(expr)
    #     else
    #         Threads.@threads $(expr)
    #     end
    # end)
    return esc(:(@batch $(expr)))
end

# TODO: mul! will not work without this
@inline Base.:+(a::SVector{N,T}, s::Number) where {N,T} = SVector{N,T}(a .+ s)

# TODO: dot will not work without this
@inline LinearAlgebra.dot(a::Number, b::SVector) = a * b

# TODO: OrdinaryDiffEq will not work without this
@inline Base.abs2(a::SVector) = abs2.(a)
@inline Base.abs(a::SVector) = abs.(a)

include("Storage.jl")
