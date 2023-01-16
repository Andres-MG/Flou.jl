function gaussian_bump(x, y, z, x0, y0, z0, sx, sy, sz, h)
    return h * exp(-(x-x0)^2/(2sx^2) - (y-y0)^2/(2sy^2) - (z-z0)^2/(2sz^2))
end

function gaussian_bump(x, y, x0, y0, sx, sy, h)
    rt = typeof(x)
    v0 = zero(rt)
    v1 = one(rt)
    return gaussian_bump(x, y, v0, x0, y0, v0, sx, sy, v1, h)
end

function gaussian_bump(x, x0, sx, h)
    rt = typeof(x)
    v0 = zero(rt)
    v1 = one(rt)
    return gaussian_bump(x, v0, v0, x0, v0, v0, sx, v1, v1, h)
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

include("Storage.jl")
