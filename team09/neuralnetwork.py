from layer import Layer 
class NeuralNetwork:

    def __init__(self, layerSize):
        self.layerSize = layerSize
        self.layers = []

    def makeNetwork(self, layerSizes):
        for l in range(0,len(self.layerSize)):
            self.layers[l] = Layer(layerSizes[l], layerSizes(l+1))
    
    def outputNetwork(self, inputV):
        for layers in self.layers:
            inputV = layers.CalcOutputs(inputV)
        return inputV
    
    def classify(self, inputV):
        outputs = self.outputNetwork(inputV)
        return max(outputs)
    
    def cost(self, dataPoint):
        outputs = self.outputNetwork(dataPoint.inputV)
        outputLayer = self.layers[len(self.layers)-1]
        c = 0
        for nodeOut in range(0,len(outputs)):
            c += outputLayer.nodeCost(outputs[nodeOut], dataPoint.ex)