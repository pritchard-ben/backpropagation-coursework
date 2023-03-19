import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as nps

def getData(): 
    file = open("differentHandling.txt", "r")
    dataSet = [] # initialise empty list
    for x in file: # for each line in the file
        x = x[0:len(x)-1]
        lineList = x.split("\t")
        dataSet.append(lineList)
    return dataSet

def convertFloat(dataSet):
    errorLocation = [] # record indexes of missing data
    errorCount = 0 # count number of missing data
    for i in range(0, len(dataSet)): # for each row
        for j in range(0, len(dataSet[i])):
            try: # try to convert to float
                dataSet[i][j] = float(dataSet[i][j])
                
            except ValueError: # if not possible, record index and update count
                errorCount += 1
                dataSet[i][j] = dataSet[i-4][j]
                errorLocation.append(i)
    # display number of missing data
    # print("Error count: " + str(errorCount))
    # reverse sort to avoid index errors
    errorLocation.sort(reverse=True)
    # remove rows with missing data
    for i in errorLocation:
        dataSet.pop(i)
    #return cleansed data
    # print(dataSet[7])
    return dataSet

def standardiseData(dataSet):
    minList = [] # initialise empty list
    maxList = [] # initialise empty list
    # traverse list, record min and max, use that to 
    # standardise data by traversing again to standardise
    for x in range(len(dataSet[0])-1):
        minList.append(dataSet[0][x])
        maxList.append(dataSet[0][x])
            
    # print(dataSet[0])
    for i in range(len(dataSet)):
        for j in range(len(dataSet[0])-1):
            if dataSet[i][j] < minList[j]:
                minList[j] = dataSet[i][j]
            if dataSet[i][j] > maxList[j]:
                maxList[j] = dataSet[i][j]
    # print(minList, maxList)
    for i in range(0, len(dataSet)):
        for j in range(0, len(dataSet[0])-1):
            dataSet[i][j] = 0.8*((dataSet[i][j] - minList[j]) / (maxList[j] - minList[j])) + 0.1
    # plt.figure(figsize=(8,5), layout="constrained")
    # date, t, w, sr, dsp, drh, panE = [], [], [], [], [], [], []
    # for x in range(len(dataSet)):
    #     date.append(dataSet[x][6] / 100)
    #     t.append(dataSet[x][0])
    #     w.append(dataSet[x][1])
    #     sr.append(dataSet[x][2])
    #     dsp.append(dataSet[x][3])
    #     drh.append(dataSet[x][4])
    #     panE.append(dataSet[x][5])
    # plt.plot(date, t, label="Temperature")
    # plt.plot(date, w, label="Wind")
    # plt.plot(date, sr, label="Solar Radiation")
    # plt.plot(date, dsp, label="Air Pressure")
    # plt.plot(date, drh, label="Humidity")
    # plt.plot(date, panE, label="Pan Evaporation")
    # plt.xlabel("Date")
    # plt.ylabel("Values")
    # plt.legend()
    # plt.show()
    return dataSet, minList[len(minList)-2], maxList[len(maxList)-2]

def splitData(dataSet):
    trainingSet = [[]] # initialise empty list
    validationSet = [[]] # initialise empty list
    testSet = [[]] # initialise empty list
    for i in range(0, len(dataSet)):
        if i % 5 == 0:
            testSet.append(dataSet[i])
        else:
            trainingSet.append(dataSet[i])
    trainingSet.pop(0)
    validationSet.pop(0)
    testSet.pop(0)
    return trainingSet, testSet

def getAllData():
    data = getData()
    data = convertFloat(data)
    data, minOut, maxOut = standardiseData(data)
    trainingSet, testSet = splitData(data)
    return trainingSet, testSet, minOut, maxOut

if __name__ == "__main__":
    trainingSet, testSet , minOut, maxOut = getAllData()
    print (len(trainingSet), len(testSet))
    
    
    