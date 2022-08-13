class layer {
    constructor(length,inputLength) {
        this.length = length
        this.inputLength = inputLength
        this.weights = []
        this.biases = []
        this.costGradientWeight = []
        this.costGradientBias = []
        for(var outputId = 0; outputId < this.length; outputId++) {
            this.biases[outputId] = Math.random()
            this.weights[outputId] = []
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                this.weights[outputId][inputId] = Math.random()
            }
        }   
    }
    reLu(input) {
        return input > 0 ? input : 0
    }
    sigmoid(input) {
        return 1 / (1 + Math.exp(-input))
    }
    cost(output,expected) {
        var cost = 0
        for(var outputId = 0; outputId < output.length; outputId++) {
        cost += Math.pow(output[outputId] - expected[outputId],2)
        }
        return cost / output.length
    }
    sparseCategoryCrossEntropy(output,expected) {
        var cost = 0
        for(var outputId = 0; outputId < output.length; outputId++) {
            cost += -expected[outputId] * Math.log(output[outputId])
        }
        return cost / output.length
    }
    CategoryCrossEntropy(output,expected) {
        var cost = 0
        for(var outputId = 0; outputId < output.length; outputId++) {
            cost += -expected[outputId] * Math.log(output[outputId]) - (1 - expected[outputId]) * Math.log(1 - output[outputId])
        }
        return cost / output.length
    }

    calculateOutput(input) {
            var output = []
            for(var outputId = 0; outputId < this.length; outputId++) {
            output[outputId] = this.biases[outputId]
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                output[outputId] += reLu(input[inputId] * this.weights[outputId][inputId])
            }
        }
    }   

    applyCostGradient(learnRate) {
        for(var outputId = 0; outputId < this.length; outputId++) {
            this.biases[outputId] -= learnRate * this.costGradientBias[outputId]
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                this.weights[outputId][inputId] -= learnRate * this.costGradientWeight[outputId][inputId]
            }
        }
    }

}
export class network {
    constructor(...layerLength) {
        this.layers = []
        for(var layerId = 0; layerId < layerLength.length; layerId++) {
        this.layers[layerId] = new layer(layerLength[layerId],layerLength[layerId-1] || 0)
        }
    }
    calculateOutput(input) {
        var output = input
        for(const layer of this.layers) {
        output = layer.calculateOutput(output)
        }
        return output
    }
    Cost(input, expected){
        var output = this.calculateOutput(input)
        var cost = 0
        for(var outputId = 0; outputId < output.length; outputId++) {
            cost += output.sparseCategoryCrossEntropy(output, expected)
        }
        return cost / output.length
    }

    // random learn lmao
    Learn(input,output)
    {
        for(const layer of this.layers) {
            for(var outputId = 0; outputId < layer.length; outputId++) {
                layer.biases[outputId] = Math.random()
                layer.weights[outputId] = []
                for(var inputId = 0; inputId < layer.inputLength; inputId++) {
                    layer.weights[outputId][inputId] = Math.random()
                }
            }  
        }
    }
}