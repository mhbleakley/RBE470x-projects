import math
class Layer:

    def __init__(self, nIn, nOut):
        self.nIn = nIn
        self.nOut = nOut
        self.weights = [nIn, nOut]
        self.biases = [nOut]
    
    def makeLayer(self, nIn, nOut):
        temp = Layer(nIn, nOut)
        return temp

    def CalcOutputs(self, inputV):
        activations = []
        for nodeOut in range(0,self.nOut):
            weightedInput = inputV.biases[nodeOut]
            for nodeIn in range(0,self.nIn):
                weightedInput += inputV[nodeIn] * inputV.weights[nodeIn, nodeOut]
            activations[nodeOut] = self.actFunc(weightedInput)
        return activations
    
    def actFunc(self, weightedInput):
        output = 1 / (1 + math.exp(-weightedInput))
        return output
    
    def nodeCost(self, outputAct, expOutput):
        error = outputAct - expOutput
        return error*error