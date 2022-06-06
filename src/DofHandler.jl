struct DofHandler
    regoffsets::Vector{Int}
    reg2loc::Vector{Vector{Int}}
    loc2reg::Vector{Pair{Int,Int}}
end

function DofHandler(nelems_per_region::AbstractVector{Int})
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
    return DofHandler(regoffsets, reg2loc, loc2reg)
end

function DofHandler(elemindices::AbstractVector{<:AbstractVector{Int}})
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
    return DofHandler(regoffsets, reg2loc, loc2reg)
end

nregions(dh::DofHandler) = length(dh.reg2loc)
nelements(dh::DofHandler) = length(dh.loc2reg)
nelements(dh::DofHandler, i) = length(dh.reg2loc[i])

eachregion(dh::DofHandler) = Base.OneTo(nregions(dh))
eachelement(dh::DofHandler) = Base.OneTo(nelements(dh))
eachelement(dh::DofHandler, i) = Base.OneTo(nelements(dh, i))

region_offset(dh::DofHandler, i) = dh.regoffsets[i]
reg2loc(dh::DofHandler, ireg, ielem) = dh.reg2loc[ireg][ielem]
loc2reg(dh::DofHandler, ielem) = dh.loc2reg[ielem]
