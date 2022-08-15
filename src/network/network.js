class layer {
    constructor(length,inputLength) {
        this.length = length
        this.inputLength = inputLength
        this.weights = []
        this.costGradientWeight = []
        this.costGradientBias = []
        for(var outputId = 0; outputId < this.length; outputId++) {
            this.costGradientBias[outputId] = 0
            this.costGradientWeight[outputId] = []
            this.weights[outputId] = []
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                this.costGradientWeight[outputId][inputId] = 0
                this.weights[outputId][inputId] = 1
            }
        }
        this.weightedInputs = []
        this.activations = []
        this.inputs = []
    }
    clearGradient() {
        this.weightedInputs = []
        this.activations = []
        this.inputs = []
        for(var outputId = 0; outputId < this.length; outputId++) {
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                this.costGradientWeight[outputId][inputId] = 0
            }
        }
    }
    reLu(input) {
        return input > 0 ? input : 0
    }
    Predict(input) {
        var output = []
        this.inputs = input
        for(var outputId = 0; outputId < this.length; outputId++) {
            output[outputId] = 0
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                output[outputId] += this.weights[outputId][inputId] * input[inputId]
            }
            this.weightedInputs[outputId] = output[outputId]    
            this.activations[outputId] = this.reLu(output[outputId])
            output[outputId] = this.activations[outputId]
        }
        return output
    }   
    applyCostGradient(learnRate) {
        for(var outputId = 0; outputId < this.length; outputId++) {
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                this.weights[outputId][inputId] -= learnRate * this.costGradientWeight[outputId][inputId]
            }
        }
    }
    nodeCostDerivative(output, expectedOutput)
    {
        return 2 * (output - expectedOutput)
    }
    activationDerivative(input) {
        return input > 0 ? 1 : 0
    }
    CalculateNodeValues(expectedOutput)
    {
        var nodeValues = []
        for(var outputId = 0; outputId < this.length; outputId++) {
            const costDerivative = this.nodeCostDerivative(this.activations[outputId], expectedOutput[outputId])
            const activationDerivative = this.activationDerivative(this.weightedInputs[outputId])
            nodeValues[outputId] = costDerivative * activationDerivative
        }
        return nodeValues
    }
    updateGradients(nodeValues)
    {
        for(var outputId = 0; outputId < this.length; outputId++) {
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                this.costGradientWeight[outputId][inputId] += nodeValues[outputId] * this.inputs[inputId]
            }
        }
    }


}
export class network {
    constructor(...layerLength) {
        this.layers = []
        for(var layerId = 0; layerId < layerLength.length; layerId++) {
            this.layers[layerId] = new layer(layerLength[layerId],layerLength[layerId-1]||0)
        }
    }
    updateAllGradients(input,output) {
        this.Predict(input)
        var outputLayer = this.layers[this.layers.length-1]
        var nodeValues = outputLayer.CalculateNodeValues(output)
        outputLayer.updateGradients(nodeValues)
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
    Learn(inputArray,outputArray,epochs)
    {
        for(var j = 0; j < epochs; j++) {
            for(var i = 0; i < inputArray.length; i++) {
                this.updateAllGradients(inputArray[i],outputArray[i])
            }
            for(var i = 1; i < this.layers.length; i++) {
                this.layers[i].applyCostGradient(0.1)
            }
            for(const layer of this.layers) {
                layer.clearGradient()
            }
        }
    }
}