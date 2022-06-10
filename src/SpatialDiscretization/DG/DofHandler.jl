struct DofHandlerDG <: AbstractDofHandler
    regoffsets::Vector{Int}
    reg2loc::Vector{Vector{Int}}
    loc2reg::Vector{Pair{Int,Int}}
end

function DofHandlerDG(nelems_per_region::AbstractVector{Int})
    nregions = length(nelems_per_region)
    nelements = sum(nelems_per_region)
    regoffsets = Vector{Int}(undef, nregions)
    reg2loc = Vector{Vector{Int}}(undef, nregions)
    loc2reg = Vector{Pair{Int,Int}}(undef, nelements)
    cnt = 0
    for (ireg, nelems) in enumerate(nelems_per_region)
        ielems = (1:nelems) .+ cnt
        regoffsets[ireg] = cnt
        reg2loc[ireg] = ielems |> collect
        loc2reg[ielems] .= Pair.(ireg, 1:nelems)
        cnt += nelems
    end
    return DofHandlerDG(regoffsets, reg2loc, loc2reg)
end

function DofHandlerDG(elemindices::AbstractVector{<:AbstractVector{Int}})
    nregions = length(elemindices)
    nelements = sum(length.(elemindices))
    regoffsets = Vector{Int}(undef, nregions)
    reg2loc = Vector{Vector{Int}}(undef, nregions)
    loc2reg = Vector{Pair{Int,Int}}(undef, nelements)
    cnt = 0
    for (ireg, ielems) in enumerate(elemindices)
        nelems = length(ielems)
        regoffsets[ireg] = cnt
        reg2loc[ireg] = ielems
        loc2reg[ielems] .= Pair.(ireg, 1:nelems)
        cnt += nelems
    end
    return DofHandlerDG(regoffsets, reg2loc, loc2reg)
end

nregions(dh::DofHandlerDG) = length(dh.reg2loc)
nelements(dh::DofHandlerDG) = length(dh.loc2reg)
nelements(dh::DofHandlerDG, i) = length(dh.reg2loc[i])

eachregion(dh::DofHandlerDG) = Base.OneTo(nregions(dh))
eachelement(dh::DofHandlerDG) = Base.OneTo(nelements(dh))
eachelement(dh::DofHandlerDG, i) = Base.OneTo(nelements(dh, i))

region_offset(dh::DofHandlerDG, i) = dh.regoffsets[i]
reg2loc(dh::DofHandlerDG, ireg, ielem) = dh.reg2loc[ireg][ielem]
loc2reg(dh::DofHandlerDG, ielem) = dh.loc2reg[ielem]
