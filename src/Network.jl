module Network


include("Layer.jl")

using Random


abstract type BaseNetwork end

function predict(
    N::BaseNetwork,
    x::VecOrMat{<:Real}
)::VecOrMat{<:Real}
    for layer = N.layers
        x = Layer.forward(layer, x)
    end
    return x
end


mutable struct TwoLayerNet <: BaseNetwork
    input_size::Int
    hidden_size::Int
    output_size::Int
    layers::Array{Layer.AbstractLayer,1}
    params::Array{VecOrMat,1}
    
    function TwoLayerNet(I::Int, H::Int, O::Int)
        W1 = randn(I, H)
        b1 = randn(H)
        W2 = randn(H, O)
        b2 = randn(O)
        layers = [
            Layer.Affine(W1, b1),
            Layer.Sigmoid(),
            Layer.Affine(W2, b2)
        ]
        params = [l.params for l = layers]
        return new(I, H, O, layers, params)
    end
end 


end  # module
