function Base.show(io::IO, ::MIME"text/plain", eq::GradientEquation)
    @nospecialize
    nd = ndims(eq)
    nvars = nvariables(eq)
    vstr = (nvars == 1) ? " variable" : " variables"
    print(io, nd, "D Gradient equation with ", nvars, vstr)
    return nothing
end

function variablenames(::GradientEquation{ND,NV}; unicode=false) where {ND,NV}
    names = if unicode
        ["∂u$(i)/∂x$(j)" for i in 1:NV, j in 1:ND]
    else
        ["u_$(i)$(j)" for i in 1:NV, j in 1:ND]
    end
    return names |> vec |> Tuple
end
