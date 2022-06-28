abstract type AbstractSensor end

function sense! end

struct ModalSensor <: AbstractSensor end

function sense!(sensorval, Q, ::ModalSensor)
end
