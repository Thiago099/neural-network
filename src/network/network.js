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
        this.weightedInputs = []
        this.activations = []
        this.inputs = []
    }
    addNode(nextLayer)
    {
        this.length++
        this.weights.push([])
        this.costGradientWeight.push([])
        for(var i = 0; i<this.inputLength; i++) {
            this.weights[this.weights.length-1][i] = 0
            this.costGradientWeight[this.costGradientWeight.length-1][i] = 0    
        }
        nextLayer.inputLength++
        for(var i = 0; i<nextLayer.length; i++) {
            nextLayer.weights[i].push(0)
            nextLayer.costGradientWeight[i].push(0)
        }

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
        this.learnRate = 0.1
    }
    updateAllGradients(input,output) {
        // this.Predict(input)
        // var outputLayer = this.layers[this.layers.length-1]
        // var nodeValues = outputLayer.CalculateNodeValues(output)
        // outputLayer.updateGradients(nodeValues)

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
            if(j % 100 == 0) 
            this.layers[1].addNode(this.layers[2])
        }
    }
}