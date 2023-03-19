# import numpy as np
import random
import math
import dataProcess
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as nps
# The sigmoid function f(x) = 1/(1+e^-x)
def sigmoid(x):
    return 1/(1+math.exp(-x))
# The derivative of the sigmoid function f'(x) = f(x)(1-f(x))
def sigmoidDerivative(x):
    return x*(1-x)
def destandardiseOutput(output):
    return (output - 0.1) * (maxOut - minOut) / 0.8 + minOut
# Initialising the weights and biases
def initWB(inputs, hidden, output):
    weights = []
    outWeights = []
    biases = []
    for x in range(hidden):
        # weights.append([])
        
        for x in range(output):# add weights from hidden layer to output layer
            outWeights.append(random.uniform(0,1))
        nestWeights = []
        for y in range(inputs): # add weights from input layer to hidden layer
            nestWeights.append(random.uniform(0,1))
            # weights[x][y] = random.uniform(0,1)
        weights.append(nestWeights)
    for x in range(hidden): # add biases to hidden layer
        biases.append(random.uniform(0,1))
    outBias = random.uniform(0,1) # add bias to output layer
    
    return weights, outWeights, biases, outBias
def trainModel(useData, weights, outWeights, biases, outBias, learningRate):
    #retrieving data as well as min and max output values to be used in destandardisation
    # Set the learning rate
    # learningRate = 0.1
    # weights, outWeights, biases, outBias = initWB(5, 8, 1)
    # print ("w", weights,"\n o", outWeights, "\n B", biases, "\n b", outBias)
    weightedSums = []
    activations = []
    for x in range(len(weights)):
        weightedSums.append(0)
        activations.append(0)
    # Forward propagation
    # Iterate through the training data
    # for x in range(len(useData)):# for each training example
    for x in range(len(useData)):
        outSum = 0 # initialise weighted sum for output layer as 0
        for y in range(len(weights)):# for each neuron in the hidden layer
            weightedSum = 0 # initialise weighted sum as 0
            for z in range(len(useData[x])-2): # for each input
                weightedSum += weights[y][z] * useData[x][z] # calculate weighted sum, add to weighted sum
            weightedSum += biases[y] # add bias to weighted sum
            weightedSums[y] = weightedSum # store weighted sum in list
            activations[y] = sigmoid(weightedSum) # calculate activation, store in list
            outSum += outWeights[y] * activations[y] # calculate weighted sum, add to weighted sum
        outSum += outBias 
        outActivation = sigmoid(outSum) # calculate activation for output layer 
        # print(outActivation, useData[0][4])
        
    # Backpropagation
    # Calculate the delta values for the output layer
        hiddenDelta = []

        pastWeightChange = [] # initialise list to store past weight changes to add momentum
        for i in range(len(weights)):
            pastWeightChange.append(0)
        deltaOut = (useData[x][len(useData[x])-2] - outActivation) * sigmoidDerivative(outActivation) # calculate delta for output layer
    # Calculate the delta values for the hidden layer
        for y in range(len(weights)): # iterate through the hidden layer again
            hiddenDelta.append(deltaOut * outWeights[y] * sigmoidDerivative(activations[y])) # calculate delta for hidden layer    
    # Update the weights and biases
            outWeights[y] = outWeights[y] + learningRate * deltaOut * activations[y] # update the weights from hidden layer to output layer
            biases[y] = biases[y] + learningRate * hiddenDelta[y] # update the biases for the hidden layer
            weightChange = []
            for z in range(len(weights[y])): # iterate through the weights again
                weightChange.append(learningRate * hiddenDelta[y] * useData[x][z])
                weights[y][z] = (weights[y][z] + weightChange[z])# + s0.9 * pastWeightChange[z]) # update the weights from input layer to hidden layer, pastWeightChange is added to add momentum
            pastWeightChange = weightChange# update pastWeightChange to be used in next iteration
        outBias = outBias + learningRate * deltaOut # update the bias for the output layer
    return weights, outWeights, biases, outBias

def useModel(weights, outWeights, biases, outBias, useData, minOut, maxOut):  
    activations = []
    weightedSums = []
    
    for x in range(len(weights)):
        weightedSums.append(0)
        activations.append(0)
    
    errorSum = 0
        
    for x in range(len(useData)):# test with one training example
        for y in range(len(weights)):# for each neuron in the hidden layer
            weightedSum = 0 # initialise weighted sum as 0
            
            for z in range(len(useData[x])-2): # for each input
                weightedSum += weights[y][z] * useData[x][z] # calculate weighted sum, add to weighted sum
            weightedSum += biases[y] # add bias to weighted sum
            weightedSums[y] = weightedSum # store weighted sum in list
            activations[y] = sigmoid(weightedSum) # calculate activation, store in list
        outSum = 0 # initialise weighted sum for output layer as 0

        for y in range(len(outWeights)): # for each weight from hidden layer to output layer
            outSum += outWeights[y] * activations[y] # calculate weighted sum, add to weighted sum
        outSum += outBias
        outActivation = sigmoid(outSum) # calculate activation for output layer
        errorSum += abs(outActivation - useData[x][len(useData[x])-2])
        # print(errorSum)

    averageError = errorSum / len(useData)
    # print("Average error with current weights: ",averageError)
    return averageError

if __name__ == "__main__":
    bestError = 1
    trainingData, testData, minOut, maxOut = dataProcess.getAllData()
    epochs = 1
    # weights, outWeights, biases, outBias = initWB(5, 10, 1) # first parameter is number of inputs, second is number of hidden neurons, third is number of outputs
    for x in range(5):
        bestError = 1
        # for y in range(5):
        weights, outWeights, biases, outBias = initWB(5, 10, 1) # first parameter is number of inputs, second is number of hidden neurons, third is number of outputs
        for z in range(epochs): # set the number of epochs to train for
            weights, outWeights, biases, outBias = trainModel(trainingData, weights, outWeights, biases, outBias, 0.1)
            # weights, outWeights, biases, outBias = trainModel(validationData, weights, outWeights, biases, outBias, 0.1)
        averageError = useModel(weights, outWeights, biases, outBias, testData, minOut, maxOut)
        if averageError < bestError:
            bestError, bestWeights, bestOutWeights, bestBiases, bestOutBias = averageError, weights, outWeights, biases, outBias
        # print(x, y)
        print("Lowest error with", epochs, "epochs is", bestError)#, "Achieved with: " "Best weights: ", bestWeights, " Best output weights: ", bestOutWeights, " Best biases: ", bestBiases, " Best output bias: ", bestOutBias )
        epochs = epochs * 10