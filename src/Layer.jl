module Layer


include("Function.jl")


abstract type AbstractLayer end

struct Sigmoid <: AbstractLayer end

function forward(
    _::Sigmoid,
    x::VecOrMat{<:Real}
)::VecOrMat{<:Real}
    Function.sigmoid.(x)
end


struct Affine <: AbstractLayer
    W::Matrix{<:Real}
    b::Vector{<:Real}
end

function forward(
    L::Affine,
    x::VecOrMat{<:Real}
)::VecOrMat{<:Real}
    L.W' * x + L.b
end


end # module
