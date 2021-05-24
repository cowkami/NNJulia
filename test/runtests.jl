using Test

include("../src/Function.jl")
include("../src/Layer.jl")

@testset "Test sigmoid function" begin
    @testset "check values" begin
        @test Function.sigmoid(0.) == 0.5
        @test round(Function.sigmoid(10) - 1, digits=3) == 0
        @test round(Function.sigmoid(-100), digits=3) == 0
        @test Function.sigmoid.([0., 0.]) == [0.5, 0.5]
    end
    
    @testset "check types" begin
        x32::Float32 = 0.
        x64::Float64 = 0.
        @test Function.sigmoid(x32) == 0.5
        @test Function.sigmoid(x64) == 0.5
    end
end

@testset "Test Sigmoid layer" begin
    sig = Layer.Sigmoid()
    @test Layer.forward(sig, [0.]) == [0.5]
    @test Layer.forward(sig, [0., 0.]) == [0.5, 0.5]
end

@testset "Test Affine layer" begin
    # two dimensional data, three parameters.
    afn = Layer.Affine(
        [
            0. 1. 0.5;
            1. 0. 0.5;
        ],  # W
        [1., 0.5, -1.]   # b
    )
    @test isapprox(Layer.forward(afn, [0.2, -0.9]), [0.1, 0.7, -1.35])
end
