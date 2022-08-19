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
                this.weights[outputId][inputId] = inputId % this.length  == outputId % this.inputLength  ? 1 : 0
            }
        }
    }
    clearGradient() {
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
        for(var outputId = 0; outputId < this.length; outputId++) {
            output[outputId] = 0
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                output[outputId] += this.weights[outputId][inputId] * input[inputId]
            }
            output[outputId] = this.reLu(output[outputId])
        }
        return output
    }   
    CollectData(input) {
        var output = []
        input = input
        for(var outputId = 0; outputId < this.length; outputId++) {
            var cur = 0
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                cur += this.weights[outputId][inputId] * input[inputId]
            }
            const nonActivationOutput = cur    
            const activations = this.reLu(cur)
            output[outputId] = {
                activations:activations,
                nonActivations:nonActivationOutput
            }
        }
        return {
            output,
            input,
        }
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
    CalculateNodeValues(prediction,expectedOutput)
    {
        var nodeValues = []
        for(var outputId = 0; outputId < this.length; outputId++) {
            const costDerivative = this.nodeCostDerivative(prediction.output[outputId].activations, expectedOutput[outputId])
            const activationDerivative = this.activationDerivative(prediction.output[outputId].nonActivations)
            nodeValues[outputId] = costDerivative * activationDerivative
        }
        return nodeValues
    }
    updateGradients(prediction,nodeValues)
    {
        for(var outputId = 0; outputId < this.length; outputId++) {
                for(var inputId = 0; inputId < this.inputLength; inputId++) {
                this.costGradientWeight[outputId][inputId] += nodeValues[outputId] * prediction.input[inputId]
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
        this.learnRate = 0.1
    }
    updateAllGradients(input,output) {
        var outputLayer = this.layers[this.layers.length-1]
        const prediction = outputLayer.CollectData(input)
        var nodeValues = outputLayer.CalculateNodeValues(prediction,output)
        outputLayer.updateGradients(prediction,nodeValues)

        // var hiddenLayer = this.layers[this.layers.length-2]
        // nodeValues = hiddenLayer.CalculateHiddenNodeValues(outputLayer,nodeValues)
        // hiddenLayer.updateGradients(nodeValues)
    }

    Predict(input) {
        var output = input
        for(var i = 1; i < this.layers.length; i++) {
            output = this.layers[i].Predict(output)
        }
        return output
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