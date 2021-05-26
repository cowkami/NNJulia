module Layer


include("Function.jl")


abstract type AbstractLayer end


mutable struct Sigmoid <: AbstractLayer
    params::Array{VecOrMat{<:Real},1}
    grads::VecOrMat{<:Real}
    out::VecOrMat{<:Real}
    function Sigmoid()
        new(VecOrMat{<:Real}[], Real[], Real[])
    end
end

function forward(
    L::Sigmoid,
    x::VecOrMat{<:Real}
)::VecOrMat{<:Real}
    L.out = Function.sigmoid.(x)
    L.out
end


mutable struct Affine <: AbstractLayer
    W::Matrix{<:Real}
    b::Vector{<:Real}
    params::Array{VecOrMat{<:Real},1}
    grads::VecOrMat{<:Real}
    out::VecOrMat{<:Real}
    function Affine(W::Matrix{<:Real}, b::Vector{<:Real})::Affine
        new(
            W,
            b,
            [W, b],
            zeros(size(b)),
            zeros(size(b))
        )
    end
end

function forward(
    L::Affine,
    x::VecOrMat{<:Real}
)::VecOrMat{<:Real}
    L.out = L.W' * x .+ L.b
    return L.out
end


mutable struct Softmax <: AbstractLayer
    params::Array{VecOrMat{<:Real},1}
    grads::VecOrMat{<:Real}
    out::VecOrMat{<:Real}
    function Softmax()
        new(VecOrMat{<:Real}[], Real[], Real[])
    end
end

function forward(
    L::Softmax,
    x::VecOrMat{<:Real}
)::VecOrMat{<:Real}
    L.out = Function.softmax(x)
    L.out
end


end # module
