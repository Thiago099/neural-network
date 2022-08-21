class layer {
    constructor(length,inputLength) {
        this.length = length
        this.inputLength = inputLength
        this.weights = []
        this.costGradientWeight = []
        this.costGradientBias = []
        this.bias = []
        for(var outputId = 0; outputId < this.length; outputId++) {
            this.costGradientBias[outputId] = 0
            this.costGradientWeight[outputId] = []
            this.weights[outputId] = []
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                this.costGradientWeight[outputId][inputId] = 0
                this.weights[outputId][inputId] = inputId % this.length  == outputId % this.inputLength  ? 1 : 0 // 1  
            }
            this.bias[outputId] = 0
        }
    }
    clearGradient() {
        for(var outputId = 0; outputId < this.length; outputId++) {
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                this.costGradientWeight[outputId][inputId] = 0
            }
            this.costGradientBias[outputId] = 0

        }
    }
    reLu(input) {
        return input > 0 ? input : 0
    }
    Predict(input) {
        var output = []
        for(var outputId = 0; outputId < this.length; outputId++) {
            output[outputId] = this.bias[outputId]
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
            var cur = this.bias[outputId]
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
                this.weights[outputId][inputId] -= this.costGradientWeight[outputId][inputId] / (this.inputLength*this.length)
            }
            // this.bias[outputId] -= this.costGradientBias[outputId]
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
    CalculateHiddenLayerNodeValues(prediction,oldLayer,oldNodeValues)
    {
        const newNodeValues = []
        for(var newNodeIndex = 0; newNodeIndex < this.length; newNodeIndex++) {
            var newNodeValue = 0
            for(var oldNodeIndex = 0; oldNodeIndex < oldNodeValues.length; oldNodeIndex++) {
                const weightDerivative = oldLayer.weights[oldNodeIndex][newNodeIndex]
                newNodeValue += oldNodeValues[oldNodeIndex] * weightDerivative
            }
            newNodeValues[newNodeIndex] = newNodeValue * this.activationDerivative(prediction.output[newNodeIndex].nonActivations)
        }
        return newNodeValues
    }
    updateGradients(prediction,nodeValues)
    {
        for(var outputId = 0; outputId < this.length; outputId++) {
            var MaxWeight = {value:0,bad:0}
            var MaxBias = {value:0,bad:0}
            for(var inputId = 0; inputId < this.inputLength; inputId++) {
                var sumWeight = 0
                var sumBias = 0
                for(var sampleId = 0; sampleId < nodeValues.length; sampleId++) {
                    sumWeight += nodeValues[sampleId][outputId] * prediction[sampleId].input[inputId]
                    sumBias += nodeValues[sampleId][outputId]
                }
                var bad = Math.abs(sumWeight) 
                if(bad > MaxWeight.bad) {
                    MaxWeight = {
                        value:sumWeight / nodeValues.length,
                        bad,
                        inputId
                    }
                }
                var badBias = Math.abs(sumBias)
                if(badBias > MaxBias.bad) {
                    MaxBias = {
                        value:sumBias / nodeValues.length,
                        badBias,
                        inputId
                    }
                }
            }
            if(MaxWeight.value != 0) {
                this.costGradientWeight[outputId][MaxWeight.inputId] = MaxWeight.value
            }
            if(MaxBias.value != 0) {
                this.costGradientBias[outputId] = MaxBias.value
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
        var nodeValues = []
        const prediction = []
        const currentPredictions = []
        for(var sampleId = 0; sampleId < input.length; sampleId++)
        {
            currentPredictions[sampleId] = this.Fill(input[sampleId])
            nodeValues.push(outputLayer.CalculateNodeValues(currentPredictions[sampleId][this.layers.length-2], output[sampleId]))
            prediction.push(currentPredictions  [sampleId][this.layers.length -2])
        }
        outputLayer.updateGradients(prediction,nodeValues)
        for(var layerId = this.layers.length - 3; layerId >= 0; layerId--) {
            const nextLayer = this.layers[layerId+1]
            const newNodeValues = []
            const newPrediction = []
            for(var sampleId = 0; sampleId < input.length; sampleId++)
            {
                newNodeValues.push(nextLayer.CalculateHiddenLayerNodeValues(currentPredictions[sampleId][layerId],outputLayer, nodeValues[sampleId]))
                newPrediction.push(currentPredictions[sampleId][layerId])
            }
            nextLayer.updateGradients(newPrediction ,newNodeValues)
            nodeValues = newNodeValues
            outputLayer = nextLayer
        }
    }

    Predict(input) {
        var output = input
        for(var i = 1; i < this.layers.length; i++) {
            output = this.layers[i].Predict(output)
        }
        return output
    }
    Fill(input)
    {
        var result = []
        var output = this.layers[1].CollectData(input)
        for(var i = 2; i < this.layers.length; i++) {
            result.push(output)
            output = this.layers[i].CollectData(output.output.map(x=>x.activations))
        }
        result.push(output)
        return result
    }
    Learn(inputArray,outputArray,epochs)
    {
        for(var j = 0; j < epochs; j++) {
            this.updateAllGradients(inputArray,outputArray)
            for(var i = 1; i < this.layers.length; i++) {
                this.layers[i].applyCostGradient(0.1)
            }
            for(const layer of this.layers) {
                layer.clearGradient()
            }
        }
    }
}


