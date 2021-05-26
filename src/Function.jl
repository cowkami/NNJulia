module Function

sigmoid(x::Real)::Real = 1 / (1 + exp(-x))

softmax(x::VecOrMat{<:Real})::VecOrMat{<:Real} =
    exp.(x) / sum(exp.(x))

end # module
