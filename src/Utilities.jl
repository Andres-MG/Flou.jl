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

# TODO: Workaround to get fast `mul!` to work with a sparse matrix and an array from
# `StaticArrays`:
# https://github.com/JuliaSparse/SparseArrays.jl/blob/a3b2736abbe814899ac4317d7aab7652a650cd90/src/linalg.jl#L30
function LinearAlgebra.mul!(
    C::StridedVecOrMat,
    A::SparseArrays.AbstractSparseMatrixCSC,
    B::StaticArrays.StaticArray,
    α::Number,
    β::Number,
)
    size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    nzv = SparseArrays.nonzeros(A)
    rv = SparseArrays.rowvals(A)
    if β != 1
        β != 0 ? LinearAlgebra.rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    for k in 1:size(C, 2)
        @inbounds for col in 1:size(A, 2)
            αxj = B[col,k] * α
            for j in SparseArrays.nzrange(A, col)
                C[rv[j], k] += nzv[j]*αxj
            end
        end
    end
    return nothing
end

macro flouthreads(expr)
    esc(quote
        return if Threads.nthreads() == 1
            $(expr)
        else
            Threads.@threads $(expr)
        end
    end)
end

macro threadbuff(expr)
    return :(Tuple($(esc(expr)) for _ in 1:Threads.nthreads()))
end
