# import numpy as np
import random
import math
import dataProcess

#retrieving data as well as min and max output values to be used in destandardisation
trainingData, validationData, testData, minOut, maxOut = dataProcess.getAllData()
print (len(trainingData), len(validationData), len(testData), minOut, maxOut)

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
    outweights = []
    biases = []
    for x in range(hidden):
        # weights.append([])
        
        for x in range(output):# add weights from hidden layer to output layer
            outweights.append(random.uniform(0,1))
        nestWeights = []
        for y in range(inputs): # add weights from input layer to hidden layer
            nestWeights.append(random.uniform(0,1))
            # weights[x][y] = random.uniform(0,1)
        weights.append(nestWeights)
    for x in range(hidden): # add biases to hidden layer
        biases.append(random.uniform(0,1))
    
    return weights, outweights, biases

# Set the learning rate
learningRate = 0.1

weights, outweights, biases = initWB(5, 8, 1)
print ("w", weights,"\n o", outweights, "\n B", biases)

weightedSums = []
activations = []
for x in range(len(weights)):
    weightedSums.append(0)
    activations.append(0)

# Forward propagation
# Iterate through the training data
# for x in range(len(trainingData)):# for each training example
for x in range(1):# test with one training example
    for y in range(len(weights)):# for each neuron in the hidden layer
        weightedSum = 0 # initialise weighted sum as 0
        for z in range(len(trainingData[x])-1): # for each input
            weightedSum += weights[y][z] * trainingData[x][z] # calculate weighted sum, add to weighted sum
        weightedSum += biases[y] # add bias to weighted sum
        weightedSums[y] = weightedSum # store weighted sum in list
        activations[y] = sigmoid(weightedSum) # calculate activation, store in list
    outSum = 0 # initialise weighted sum for output layer as 0
    for y in range(len(outweights)): # for each weight from hidden layer to output layer
        outSum += outweights[y] * activations[y] # calculate weighted sum, add to weighted sum
    outActivation = sigmoid(outSum) # calculate activation for output layer
    
    #testing
    print(outActivation, trainingData[0][4])
    
# Backpropagation
# Calculate the delta values for the output layer
    deltaOut = (trainingData[x][4] - outActivation) * sigmoidDerivative(outActivation) # calculate delta for output layer
    for y in range(len(weights)):
        

# Calculate the delta values for the hidden layer
    
    
# Update the weights and biases
