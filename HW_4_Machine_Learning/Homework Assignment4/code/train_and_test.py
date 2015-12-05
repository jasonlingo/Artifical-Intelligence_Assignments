import numpy as np
from classifier import Classifier
import load_data

"""
This is the main python method that will be run.
You should determine what sort of command line arguments
you want to use. But in this module you will need to 
1) initialize your classifier and its params 
2) load training/test data 
3) train the algorithm
4) test it and output the desired statistics.
"""
def runDecisionTree(train, test, igMode, pruneThreshold):
    print "Decision Tree ==============================="
    print " - using",
    if igMode == "ig":
        print "information gain"
    elif igMode == "igr":
        print "information gain ratio"
    else:
        print igMode

    print " - pruning threshold", pruneThreshold

    dt = Classifier("decision_tree", igMode=igMode, pruneThreshold=pruneThreshold)
    dt.train(train.copy())
    dt.test(test.copy(), "test")

def runNaiveBayes(train, test):
    print "Naive Bayes ================================="

    nb = Classifier("naive_bayes")
    nb.train(train.copy())
    nb.test(test.copy(), "test")

def runNeuralNetwork(train, test, hLayer=None, mode=None, momentumFactor=0.0):
    print "Neural Network =============================="
    print " - number of hidden layer nodes:",
    if hLayer is not None:
        print hLayer
    else:
        print " default (one hidden layer with node number = 2 * feature number)"

    print " - weight initialization mode:",
    if mode is not None:
        print mode
    else:
        print "default"

    print " - momentum factor", momentumFactor

    nn = Classifier("neural_network", hidden_layer=hLayer, weightInitMode=mode, momentumFactor=momentumFactor)
    nn.train(train.copy())
    nn.test(test.copy(), "test")


'''
=====================================================================
 Decision Tree
=====================================================================
'''
print "########## Decision Tree #####################################"
''' 1.Congressional Voting Records dataset '''
print "1. Congressional Voting Records dataset without pruning"
(trainingData, testData) = load_data.load_congress_data(0.6)
pruningThreshold = 0.0
# ======================
runDecisionTree(trainingData, testData, "ig", pruningThreshold)
# ======================
runDecisionTree(trainingData, testData, "igr", pruningThreshold)

''' 2. MONKS Problem dataset '''
print ""
for i in [1, 2, 3]:
    print "2. MONKS Problem with Information Gain without pruning- dataset ", i
    (trainingData, testData) = load_data.load_monks(i)
    # ======================
    runDecisionTree(trainingData, testData, "ig", pruningThreshold)
    # ======================
    runDecisionTree(trainingData, testData, "igr", pruningThreshold)

''' 3. Iris dataset '''
print ""
print "3. Iris without pruning"
(trainingData, testData) = load_data.load_iris(0.6)
trainingData = np.floor(trainingData)
testData = np.floor(testData)
# ======================
runDecisionTree(trainingData, testData, "ig", pruningThreshold)
# ======================
runDecisionTree(trainingData, testData, "igr", pruningThreshold)


'''Pruning ================================================'''
print "1. Congressional Voting Records dataset with pruning"
(trainingData, testData) = load_data.load_congress_data(0.6)
pruningThreshold = 0.05
# ======================
runDecisionTree(trainingData, testData, "ig", pruningThreshold)
# ======================
runDecisionTree(trainingData, testData, "igr", pruningThreshold)

''' 2. MONKS Problem dataset '''
print ""
for i in [1, 2, 3]:
    print "2. MONKS Problem with Information Gain with pruning - dataset", i
    (trainingData, testData) = load_data.load_monks(i)
    # ======================
    runDecisionTree(trainingData, testData, "ig", pruningThreshold)
    # ======================
    runDecisionTree(trainingData, testData, "igr", pruningThreshold)

''' 3. Iris dataset '''
print ""
print "3. Iris with pruning"
(trainingData, testData) = load_data.load_iris(0.6)
trainingData = np.floor(trainingData)
testData = np.floor(testData)
# ======================
runDecisionTree(trainingData, testData, "ig", pruningThreshold)
# ======================
runDecisionTree(trainingData, testData, "igr", pruningThreshold)


'''
=====================================================================
 Naive Bayes
=====================================================================
'''
print ""
print "########## Naive Bayes #######################################"
''' 1.Congressional Voting Records dataset '''
print "1. Congressional Voting Records dataset"
(trainingData, testData) = load_data.load_congress_data(0.6)
# ======================
runNaiveBayes(trainingData, testData)

''' 2. MONKS Problem dataset '''
for i in [1, 2, 3]:
    print "2. MONKS Problem - dataset ", i
    (trainingData, testData) = load_data.load_monks(i)
    # ======================
    runNaiveBayes(trainingData, testData)

''' 3. Iris dataset '''
print "3. Iris"
(trainingData, testData) = load_data.load_iris(0.6)
trainingData = np.floor(trainingData)
testData = np.floor(testData)
# ======================
runNaiveBayes(trainingData, testData)


'''
=====================================================================
 Neural Network
=====================================================================
'''
print ""
print "########## Neural Network ####################################"
print "1. Congressional Voting Records dataset without momentum"

(trainingData, testData) = load_data.load_congress_data(0.6)
momentumFactor = 0.0
# ======================
weightInitMode = None
hidden_layer   = [32]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)
# ======================
weightInitMode = "shallow"
hidden_layer   = [32]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)
# ======================
weightInitMode = "deep"
hidden_layer   = [32, 32, 32]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)


''' 2. MONKS Problem dataset '''
print ""
for i in [1, 2, 3]:
    print "2. MONKS Problem without momentum - dataset ", i
    (trainingData, testData) = load_data.load_monks(i)
    # ======================
    weightInitMode = None
    hidden_layer   = [14]
    runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)
    # ======================
    weightInitMode = "shallow"
    hidden_layer   = [14]
    runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)
    # ======================
    weightInitMode = "deep"
    hidden_layer   = [14, 14, 14]
    runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)


''' 3. Iris dataset '''
print ""
print "3. Iris without momentum"
(trainingData, testData) = load_data.load_iris(0.6)
# ======================
weightInitMode = None
hidden_layer   = [16]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)
# ======================
weightInitMode = "shallow"
hidden_layer   = [16]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)
# ======================
weightInitMode = "deep"
hidden_layer   = [16, 16, 16]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)


'''========================================================='''
print "1. Congressional Voting Records dataset with momentum"

(trainingData, testData) = load_data.load_congress_data(0.6)
momentumFactor = 0.8
# ======================
weightInitMode = None
hidden_layer   = [32]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)
# ======================
weightInitMode = "shallow"
hidden_layer   = [32]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)
# ======================
weightInitMode = "deep"
hidden_layer   = [32, 32, 32]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)


''' 2. MONKS Problem dataset '''
print ""
for i in [1, 2, 3]:
    print "2. MONKS Problem with momentum - dataset ", i
    (trainingData, testData) = load_data.load_monks(i)
    # ======================
    weightInitMode = None
    hidden_layer   = [14]
    runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)
    # ======================
    weightInitMode = "shallow"
    hidden_layer   = [14]
    runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)
    # ======================
    weightInitMode = "deep"
    hidden_layer   = [14, 14, 14]
    runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)


''' 3. Iris dataset '''
print ""
print "3. Iris with momentum"
(trainingData, testData) = load_data.load_iris(0.6)
# ======================
weightInitMode = None
hidden_layer   = [16]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)
# ======================
weightInitMode = "shallow"
hidden_layer   = [16]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)
# ======================
weightInitMode = "deep"
hidden_layer   = [16, 16, 16]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)
