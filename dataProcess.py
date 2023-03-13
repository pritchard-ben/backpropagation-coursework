def getData(): 
    file = open("dataset.txt", "r")
    dataSet = [] # initialise empty list
    # print(file.readline())
    for x in file: # for each line in the file
        x = x[0:len(x)-1]
        lineList = x.split("\t")
        #The following lines will convert the strings to floats, however while data not clean this is impossible
        # for y in range(0, len(lineList)-1):
        #     lineList[y] = float(lineList[y])
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
                # print(i, j, dataSet[i])
                errorLocation.append(i)
    # display number of missing data
    # print("Error count: " + str(errorCount))
    # reverse sort to avoid index errors
    errorLocation.sort(reverse=True)
    # remove rows with missing data
    for i in errorLocation:
        dataSet.pop(i)
    #return cleansed data
    return dataSet

def standardiseData(dataSet):
    minList = [] # initialise empty list
    maxList = [] # initialise empty list
    # traverse list, record min and max, use that to 
    # standardise data by traversing again to standardise
    for i in range(0, len(dataSet)):
        for j in range(0, len(dataSet[0])):
            if i == 0:
                minList.append(dataSet[i][j])
                maxList.append(dataSet[i][j])
            else:
                if dataSet[i][j] < minList[j]:
                    minList[j] = dataSet[i][j]
                if dataSet[i][j] > maxList[j]:
                    maxList[j] = dataSet[i][j]
    for i in range(0, len(dataSet)):
        for j in range(0, len(dataSet[0])):
            dataSet[i][j] = 0.8*((dataSet[i][j] - minList[j]) / (maxList[j] - minList[j])) + 0.1
    return dataSet, minList[len(minList)-1], maxList[len(maxList)-1]

def splitData(dataSet):
    trainingSet = [[]] # initialise empty list
    validationSet = [[]] # initialise empty list
    testSet = [[]] # initialise empty list
    for i in range(0, len(dataSet)):
        if i % 5 == 0:
            testSet.append(dataSet[i])
        elif i % 5 == 1:
            validationSet.append(dataSet[i])
        else:
            trainingSet.append(dataSet[i])
    trainingSet.pop(0)
    validationSet.pop(0)
    testSet.pop(0)
    return trainingSet, validationSet, testSet

def getAllData():
    data = getData()
    data = convertFloat(data)
    data, minOut, maxOut = standardiseData(data)
    trainingSet, validationSet, testSet = splitData(data)
    return trainingSet, validationSet, testSet, minOut, maxOut

if __name__ == "__main__":
    trainingSet, validationSet, testSet , minOut, maxOut = getAllData()
    print (len(trainingSet), len(validationSet), len(testSet))