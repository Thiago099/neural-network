class layer {
    constructor(length,inputLength) {
        this.length = length
        this.inputLength = inputLength
        this.weights = []
        this.biases = []
        this.costGradientWeight = []
        this.costGradientBias = []
        for(var outputId = 0; outputId < this.length; outputId++) {
            this.biases[outputId] = 0
            this.costGradientBias[outputId] = 0
            this.costGradientWeight[outputId] = []
            this.weights[outputId] = []
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                this.costGradientWeight[outputId][inputId] = 0
                this.weights[outputId][inputId] = 0
            }
        }
        this.weightedInputs = []
    }
    clearGradient() {
        this.weightedInputs = []
        for(var outputId = 0; outputId < this.length; outputId++) {
            this.costGradientBias[outputId] = 0
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                this.costGradientWeight[outputId][inputId] = 0
            }
        }
    }
    reLu(input) {
        return input > 0 ? input : 0
    }
    reLuDerivative(input) {
        return input > 0 ? 1 : 0
    }
    sigmoid(input) {
        return 1 / (1 + Math.exp(-input))
    }
    sigmoidDerivative(input) {
        return input * (1 - input)
    }
    Predict(input) {
        var output = []
        for(var outputId = 0; outputId < this.length; outputId++) {
            output[outputId] = this.biases[outputId]
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                output[outputId] += this.reLu(this.weights[outputId][inputId] * input[inputId] )
            }
            // output[outputId] = output[outputId] / this.inputLength
        }
        return output
    }   
    // nodeCostDerivative(activation, expected)
    // {
    //     return activation * (1 - activation) * (expected - activation)
    // }
    
    
    applyCostGradient(learnRate) {
        for(var outputId = 0; outputId < this.length; outputId++) {
            this.biases[outputId] -= learnRate * this.costGradientBias[outputId]
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                this.weights[outputId][inputId] -= learnRate * this.costGradientWeight[outputId][inputId]
            }
        }
    }

    updateAllGradients(input,output) {
    
    }
    DerivativeCost(output,expected) 
    {
        return 2 * (output - expected)   
    }


}
export class network {
    constructor(...layerLength) {
        this.layers = []
        for(var layerId = 0; layerId < layerLength.length; layerId++) {
            this.layers[layerId] = new layer(layerLength[layerId],layerLength[layerId-1]||0)
        }
    }

    CategoryCrossEntropy(output,expected) {
        var cost = 0
        for(var outputId = 0; outputId < output.length; outputId++) {
            cost += -expected[outputId] * Math.log(output[outputId]) - (1 - expected[outputId]) * Math.log(1 - output[outputId])
        }
        return cost / output.length
    }
    Predict(input) {
        var output = input
        for(var i = 1; i < this.layers.length; i++) {
            output = this.layers[i].Predict(output)
        }
        return output
    }
    DullCost(output,expected) {
        var cost = 0
        for(var outputId = 0; outputId < output.length; outputId++) {
            cost += Math.pow(output[outputId] - expected[outputId],2)
        }
        return cost / output.length
    }
    Cost(input, expected){
        var output = this.Predict(input)
        var cost = 0
        for(var outputId = 0; outputId < output.length; outputId++) {
            cost += this.DullCost(output, expected)
        }
        return cost / output.length
    }
   
    Learn(inputArray,outputArray)
    {
        const h = 0.0001
        for(var i = 0; i < inputArray.length; i++) {
            var input = inputArray[i]
            var output = outputArray[i]
            const originalCost = this.Cost(input,output)
            for(const layer of this.layers) {
                layer.clearGradient()
                for(var outputId = 0; outputId < layer.length; outputId++) {
                    for(var inputId = 0; inputId < layer.inputLength; inputId++) {
                        const def = layer.weights[outputId][inputId]
                        layer.weights[outputId][inputId] += h
                        var deltaCost = this.Cost(input,output) - originalCost
                        layer.weights[outputId][inputId] = def
                        layer.costGradientWeight[outputId][inputId] += deltaCost / h

                    }
                    const def = layer.biases[outputId]
                    layer.biases[outputId] += h
                    var deltaCost = this.Cost(input,output) - originalCost
                    layer.biases[outputId] = def
                    layer.costGradientBias[outputId] += deltaCost / h
                }  
            }
            for(const layer of this.layers)
            {
                layer.applyCostGradient(h)
            }
        }
        
    }
}