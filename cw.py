# import numpy as np
import random
import math
import dataProcess

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
    for x in range(len(useData)):# test with one training example
        for y in range(len(weights)):# for each neuron in the hidden layer
            weightedSum = 0 # initialise weighted sum as 0
            for z in range(len(useData[x])-1): # for each input
                weightedSum += weights[y][z] * useData[x][z] # calculate weighted sum, add to weighted sum
            weightedSum += biases[y] # add bias to weighted sum
            weightedSums[y] = weightedSum # store weighted sum in list
            activations[y] = sigmoid(weightedSum) # calculate activation, store in list
        outSum = 0 # initialise weighted sum for output layer as 0
        for y in range(len(outWeights)): # for each weight from hidden layer to output layer
            outSum += outWeights[y] * activations[y] # calculate weighted sum, add to weighted sum
        outActivation = sigmoid(outSum) # calculate activation for output layer
        
        #testing
        # print(outActivation, useData[0][4])
        
    # Backpropagation
    # Calculate the delta values for the output layer
        hiddenDelta = []
        deltaOut = (useData[x][4] - outActivation) * sigmoidDerivative(outActivation) # calculate delta for output layer
    # Calculate the delta values for the hidden layer
        for y in range(len(weights)): # iterate through the hidden layer again
            hiddenDelta.append(deltaOut * outWeights[y] * sigmoidDerivative(activations[y])) # calculate delta for hidden layer    
    # Update the weights and biases
        for y in range(len(weights)): # iterate through the weights again
            outWeights[y] = outWeights[y] + learningRate * deltaOut * activations[y] # update the weights from hidden layer to output layer
            outBias = outBias + learningRate * deltaOut # update the bias for the output layer
            biases[y] = biases[y] + learningRate * hiddenDelta[y] # update the biases for the hidden layer
            for z in range(len(weights[y])): # iterate through the weights again
                weights[y][z] = (weights[y][z] + learningRate * hiddenDelta[y] * useData[x][z]) # update the weights from input layer to hidden layer
    return weights, outWeights, biases, outBias
    
def useModel(weights, outWeights, biases, outBias, useData, minOut, maxOut):
    activations = []
    weightedSums = []
    
    for x in range(len(weights)):
        weightedSums.append(0)
        activations.append(0)
        
    for x in range(len(useData)):# test with one training example
        for y in range(len(weights)):# for each neuron in the hidden layer
            weightedSum = 0 # initialise weighted sum as 0
            
            for z in range(len(useData[x])-1): # for each input
                weightedSum += weights[y][z] * useData[x][z] # calculate weighted sum, add to weighted sum
            weightedSum += biases[y] # add bias to weighted sum
            weightedSums[y] = weightedSum # store weighted sum in list
            activations[y] = sigmoid(weightedSum) # calculate activation, store in list
        outSum = 0 # initialise weighted sum for output layer as 0
        
        for y in range(len(outWeights)): # for each weight from hidden layer to output layer
            outSum += outWeights[y] * activations[y] # calculate weighted sum, add to weighted sum
        outActivation = sigmoid(outSum) # calculate activation for output layer
        errorSum = outActivation - useData[x][len(useData[x])-1]
        
    averageError = errorSum / len(useData)
    print(averageError)
                
if __name__ == "__main__":
    trainingData, validationData, testData, minOut, maxOut = dataProcess.getAllData()
    weights, outWeights, biases, outBias = initWB(5, 8, 1)
    for x in range(2000): # set the number of epochs to train for
        weights, outWeights, biases, outBias = trainModel(trainingData, weights, outWeights, biases, outBias, 0.1)
    useModel(weights, outWeights, biases, outBias, testData, minOut, maxOut)