using StatsBase
using Base.MathConstants

# need to add an input variable
mutable struct network
    weights::Vector{Array{Float64,2}}
    biases::Vector{Array{Float64,1}}
    nodeinputs::Vector{Array{Float64,1}}
    nodeactivations::Vector{Array{Float64,1}}

    # these are the gradients calculated when using back propogation
    weight_grs::Vector{Array{Float64,2}}
    bias_grs::Vector{Array{Float64,1}}
    nodeinput_grs::Vector{Array{Float64,1}}
    nodeactivation_grs::Vector{Array{Float64,1}}

    # training inputs and outputs
    training_ins::Vector{Array{Float64,1}}
    training_outs::Vector{Array{Float64,1}}

    # testing inputs and outputs
    test_ins::Vector{Array{Float64,1}}
    test_outs::Vector{Array{Float64,1}}

    costfunc::Function
    activationfunc::Function
    fda_costfunc::Function
    fd_activationfunc::Function


    # helper variables
    n_layers::Int64

    layerSizes::Vector{Int64}

    # layerSizes describes the sizes of the hidden layers and the output layer
    function network(inputSize::Int64,layerSizes::Vector{Int64},
                     cost_func, activation_func,
                     fda_costfunc, fd_activationfunc)

        n_layers::Int64 = length(layerSizes)

        weights = Vector{Array{Float64,2}}(undef, n_layers)
        biases = Vector{Array{Float64,1}}(undef, n_layers)
        nodeinputs = Vector{Array{Float64,1}}(undef, n_layers)
        nodeactivations = Vector{Array{Float64,1}}(undef, n_layers)

        weight_grs = Vector{Array{Float64,2}}(undef, n_layers)
        bias_grs = Vector{Array{Float64,1}}(undef, n_layers)
        nodeinput_grs = Vector{Array{Float64,1}}(undef, n_layers)
        nodeactivation_grs = Vector{Array{Float64,1}}(undef, n_layers)

        # it will initialise the weights and biases randomly
        weights[1] = rand(layerSizes[1], inputSize)
        biases[1] = -Vector(rand(layerSizes[1]));
        nodeinputs[1] = Vector{Float64}(undef,layerSizes[1])
        nodeactivations[1] = Vector{Float64}(undef,layerSizes[1])

        weight_grs[1] = Array{Float64,2}(undef,layerSizes[1], inputSize)
        bias_grs[1] = Vector{Float64}(undef,layerSizes[1])
        nodeinput_grs[1] = Vector{Float64}(undef,layerSizes[1])
        nodeactivation_grs[1] = Vector{Float64}(undef,layerSizes[1])

        for i = 2:length(layerSizes)
            weights[i] = rand(layerSizes[i],layerSizes[i-1])
            biases[i] = -Vector(rand(layerSizes[i]))
            nodeinputs[i] = Vector{Float64}(undef,layerSizes[i])
            nodeactivations[i] = Vector{Float64}(undef,layerSizes[i])

            weight_grs[i] = Array{Float64,2}(undef, layerSizes[i], layerSizes[i-1],)
            bias_grs[i] = Vector{Float64}(undef,layerSizes[i])
            nodeinput_grs[i] = Vector{Float64}(undef,layerSizes[i])
            nodeactivation_grs[i] = Vector{Float64}(undef,layerSizes[i])
        end

        new(weights,biases,nodeinputs,nodeactivations,weight_grs,bias_grs,
        nodeinput_grs,nodeactivation_grs,[],[],[],[],
        cost_func, activation_func, fda_costfunc, fd_activationfunc,length(layerSizes),layerSizes)

     end
 end

"""
    Feeds an input vector (inVector) through the network and returns the output
    vector of the network. The network will store the input and output of all
    the nodes after you have got the result
"""
function predict!(net::network, inVector::Array{Float64,1})::Vector{Float64}
    net.nodeinputs[1] = (net.weights[1] * inVector) + net.biases[1]
    net.nodeactivations[1] = net.activationfunc.(net.nodeinputs[1])

    for i = 2:net.n_layers
        net.nodeinputs[i] = (net.weights[i] * net.nodeactivations[i-1]) + net.biases[i]
        net.nodeactivations[i] = net.activationfunc.(net.nodeinputs[i])
    end

    round.(net.nodeactivations[end])
    # net.nodeactivations[end]
end


"""
    backPropogates through the network and calculates the gradients for the weights and
    the biases then stores them in the network object. \n\n
    net is the network object you want to propogate through \n
    input is that last input vector to the object. \n
    y is the last output that should have been obtained \n\n

    #Examples
    ```julia-repl
    julia> predict!(network, input)\n
    julia> backPropogate!(network, input, expected_output)
    ```
"""
function backPropogate!(net::network, input::Array{Float64,1}, y::Array{Float64,1})::Nothing
    net.nodeinput_grs[end] =
        net.fd_costfunc.(net.nodeactivations[end], y) .* net.fd_activationfunc.(net.nodeinputs[end]);
    net.bias_grs[end] = net.nodeinput_grs[end]
    net.weight_grs[end] = net.nodeinput_grs[end] * net.nodeactivations[end-1]'

    for i=(net.n_layers-1):-1:2
        net.nodeinput_grs[i] =
            (net.weights[i+1]' * net.nodeinput_grs[i+1]) .* net.fd_activationfunc.(net.nodeinputs[i]);
        net.bias_grs[i] = net.nodeinput_grs[i]
        net.weight_grs[i] = net.nodeinput_grs[i] * net.nodeactivations[i-1]'
    end

    net.nodeinput_grs[1]=
        (net.weights[2]' * net.nodeinput_grs[2]) .* net.fd_activationfunc.(net.nodeinputs[1]);
    net.bias_grs[1] = net.nodeinput_grs[1]
    net.weight_grs[1] = net.nodeinput_grs[1] * input';
    nothing
end


function learn(net::network, learningRate::Float64, batchSize::Int64)
        # first we randomly select our data from the data in the class
        indexList = sample(1:(length(net.training_ins)), batchSize, replace=false)

        sample_ins = net.training_ins[indexList]
        sample_outs = net.training_outs[indexList]

        weight_sums::Array{Array{Float64,2},1} = Array{Array{Float64,2},1}(undef, net.n_layers)
        bias_sums::Array{Array{Float64,1},1} = Array{Array{Float64,1},1}(undef, net.n_layers)

        for i=1:net.n_layers
            s = size(net.weight_grs[i])
            weight_sums[i] = zeros(Float64,(s[1],s[2]))
            bias_sums[i] = zeros(Float64,(s[1]))
        end
        for i=1:batchSize
            predict!(net,sample_ins[i])
            backPropogate!(net,sample_ins[i],sample_outs[i])
            for j=1:length(net.weight_grs)
                weight_sums[j] += net.weight_grs[j]
                bias_sums[j] += net.bias_grs[j]
            end
        end

        for i=1:net.n_layers
            net.weights[i] -=  (learningRate / batchSize) * weight_sums[i]
            net.biases[i] -= (learningRate / batchSize) * bias_sums[i]
        end
end



sigmoid = x -> 1 / (1 + e^-x)
function fd_sigmoid(x)
    sig = sigmoid(x)
    sig * (1 - sig)
end


costfunc = (a, y) -> 0.5 * sum((y - a)^2)
fda_costfunc = (a, y) -> a - y

net = network(2, [4,2,1], costfunc, sigmoid, fda_costfunc,fd_sigmoid)

# setting up weights to make an XOR neural net
# net.weights = [[[5 5];[-5 -5]], [-1 -1]]
# net.biases[1] = [ -8; 2]
# net.biases[2] = [0.75]

# creating training sets
net.training_ins = [[0.0;0.0],[0.0;1.0],[1.0;0.0],[1.0;1.0]]
net.training_outs = [[0.0],[1.0],[1.0],[0.0]]
