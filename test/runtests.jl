using Test

include("../src/Function.jl")
include("../src/Layer.jl")
include("../src/Network.jl")

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

        
@testset "Test softmax function" begin
    @testset "check values" begin
        @test Function.softmax([0.]) == [1.]
        @test isapprox(Function.softmax([100, -1., 0.5]), [1., 0., 0.])
        @test isapprox(Function.softmax([-1., -1., -100]), [0.5, 0.5, 0.])
        @test Function.softmax([0., 0., 0.]) == [1 / 3, 1 / 3, 1 / 3]
    end
    
    @testset "check types" begin
        x32::Float32 = 0.
        x64::Float64 = 0.
        @test Function.softmax([x32]) == [1.]
        @test Function.softmax([x64]) == [1.]
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
        [0. 1. 0.5;
         1. 0. 0.5],  # Weights
        [1., 0.5, -1.]   # bias
    )
    @test isapprox(Layer.forward(afn, [0.2, -0.9]), [0.1, 0.7, -1.35])
end


@testset "Test Softmax" begin
    sm = Layer.Softmax()
    @test isapprox(
        Layer.forward(sm, [0., -1., 0.5]),
        [0.33150, 0.12195, 0.54655],
        atol=1e-5
    )
end


@testset "Test TwoLayerNet" begin
net = Network.TwoLayerNet(3, 10, 5)
    x = [3., 0., 0.1]
    X = [
        3. 0.9;
    0. 3.;
        0.1 0.1
    ]
    @test size(Network.predict(net, x)) == (5,)
    @test size(Network.predict(net, X)) == (5, 2)
end
