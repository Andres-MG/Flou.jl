function save2csv end

function save2vtkhdf end

function save(filename, Q::StateVector, disc)
    ext = split(filename, ".")[end]
    if ext == "csv"
        save2csv(filename, Q, disc)
    elseif ext == "hdf"
        save2vtkhdf(filename, Q, disc)
    end
    return nothing
end
