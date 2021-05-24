module Layer


include("Function.jl")


abstract type AbstractLayer end

struct Sigmoid <: AbstractLayer
    params::Array{VecOrMat{<:Real}, 1}
    Sigmoid() = new([])
end

function forward(
    _::Sigmoid,
    x::VecOrMat{<:Real}
)::VecOrMat{<:Real}
    Function.sigmoid.(x)
end


mutable struct Affine <: AbstractLayer
    W::Matrix{<:Real}
    b::Vector{<:Real}
    params::Array{VecOrMat{<:Real}, 1}
    function Affine(W::Matrix{<:Real}, b::Vector{<:Real})::Affine
        layer = new(W, b, [])
        layer.params = [layer.W, layer.b]
        return layer
    end
end

function forward(
    L::Affine,
    x::VecOrMat{<:Real}
)::VecOrMat{<:Real}
    L.W' * x + L.b
end


end # module
