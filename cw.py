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
def trainModel(useData, weights, outWeights, biases, outBias, initialLearningRate, epochs, maxEpochs):
    
    # Initialising data structures to be used
    weightedSums = []
    activations = []
    for x in range(len(weights)):
        weightedSums.append(0)
        activations.append(0)
        
    # Implementing simulated annealing, adjusts learning rate every epoch
    learningRate = 0.01 + (initialLearningRate - 0.01) * (1-(1/(1+math.exp(10-((20*epochs+1)/maxEpochs)))))
    print("Learning rate: " + str(learningRate))
    
    # Adding momentum to the algorithm
    # Initialising data structures to be used
    pastWeightChange = [] 
    pastOutWeightChange = []
    for x in range(len(weights)):
        pastOutWeightChange.append(0)
        pastWeightChange.append([])
        for i in range(len(weights)):
            pastWeightChange[x].append(0)
    
    # Forward propagation
    # Iterate through the training data
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
        
    # Backpropagation
    # Calculate the delta values for the output layer
        hiddenDelta = []
        
        
        deltaOut = (useData[x][len(useData[x])-2] - outActivation) * sigmoidDerivative(outActivation) # calculate delta for output layer
    # Calculate the delta values for the hidden layer
        for y in range(len(weights)): # iterate through the hidden layer again
            hiddenDelta.append(deltaOut * outWeights[y] * sigmoidDerivative(activations[y])) # calculate delta for hidden layer    
    # Update the weights and biases
            outWeightChange = learningRate * deltaOut * activations[y]
            outWeights[y] = outWeights[y] + outWeightChange + (0.9 * pastOutWeightChange[y]) # update the weights from hidden layer to output layer
            pastOutWeightChange[y] = outWeightChange # update pastOutWeightChange to be used in next iteration
            
            biases[y] = biases[y] + learningRate * hiddenDelta[y] # update the biases for the hidden layer
            weightChange = []
            for z in range(len(weights[y])): # iterate through the weights again
                weightChange.append(learningRate * hiddenDelta[y] * useData[x][z])
                weights[y][z] = (weights[y][z] + weightChange[z] + 0.9 * pastWeightChange[y][z]) # update the weights from input layer to hidden layer, pastWeightChange is added to add momentum
            pastWeightChange[y] = weightChange# update pastWeightChange to be used in next iteration
        outBias = outBias + learningRate * deltaOut # update the bias for the output layer

    return weights, outWeights, biases, outBias

def useModel(weights, outWeights, biases, outBias, useData, minOut, maxOut):  
    
    # Initialising data structures to be used
    activations = []
    weightedSums = []
    for x in range(len(weights)):
        weightedSums.append(0)
        activations.append(0)
    
    errorSum = 0
    
    # Loop through the data
    for x in range(len(useData)):
        
        # For each neuron in the hidden layer
        for y in range(len(weights)):
            # Initialise weighted sum as 0
            weightedSum = 0 
            
            # Loop through each input
            for z in range(len(useData[x])-2): 
                weightedSum += weights[y][z] * useData[x][z] # calculate weighted sum, add to weighted sum
            weightedSum += biases[y] # add bias to weighted sum
            weightedSums[y] = weightedSum # store weighted sum in list
            activations[y] = sigmoid(weightedSum) # calculate activation, store in list
            
        # Initialise weighted sum for output layer as 0
        outSum = 0 

        for y in range(len(outWeights)): # for each weight from hidden layer to output layer
            outSum += outWeights[y] * activations[y] # calculate weighted sum, add to weighted sum
        outSum += outBias
        outActivation = sigmoid(outSum) # calculate activation for output layer
        
        # Calculate how far off the prediction was, add to the sum of the errors
        errorSum += abs(outActivation - useData[x][len(useData[x])-2])

    averageError = errorSum / len(useData)
    return averageError

if __name__ == "__main__":
    bestError = 1
    # Get data from expernal file
    trainingData, testData, minOut, maxOut = dataProcess.getAllData()

    # Initialise weights and biases
    weights, outWeights, biases, outBias = initWB(5, 10, 1) # first parameter is number of inputs, second is number of hidden neurons, third is number of outputs
    
    # Initialise number of epochs and learning rate values
    epochs = 2000
    initialLearningRate = 0.2
    
    # Used to plot graphs
    plt.figure(figsize=(8,3), layout="constrained")
    zList = []
    errorList = []
    
    #Train with 5 different sets of weights and biases
    for x in range(1):
        bestError = 1
        
         # For each epoch, train the model
        for z in range(epochs): 
            weights, outWeights, biases, outBias = trainModel(trainingData, weights, outWeights, biases, outBias, initialLearningRate, z, epochs)
            
            # Collecting data to plot graphs
            if z % 10 == 0:
                averageError = useModel(weights, outWeights, biases, outBias, trainingData, minOut, maxOut)
                errorList.append(averageError)
                zList.append(z)
        
        # Evaluate this set of weights and biases
        averageError = useModel(weights, outWeights, biases, outBias, testData, minOut, maxOut)
        
        # If this set of weights and biases is the best found so far, store them
        if averageError < bestError:
            bestError, bestWeights, bestOutWeights, bestBiases, bestOutBias = averageError, weights, outWeights, biases, outBias

    # Output the best values found in this cycle
    print("Lowest error with", epochs, "epochs is", bestError, "Achieved with: " "Best weights: ", bestWeights, " Best output weights: ", bestOutWeights, " Best biases: ", bestBiases, " Best output bias: ", bestOutBias )
    
    # Used to plot graphs
    plt.plot (zList, errorList)
    plt.xlabel("Epochs")
    plt.ylabel("Average Error")
    plt.show()
                