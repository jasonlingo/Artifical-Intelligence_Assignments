import numpy as np
import sys
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
    print "Decision Tree =============="
    print " - using ", igMode
    dt = Classifier("decision_tree", igMode=igMode, pruneThreshold=pruneThreshold)
    dt.train(train.copy())
    dt.test(test.copy(), "test")

def runNaiveBayes(train, test):
    print "Naive Bayes ================"
    nb = Classifier("naive_bayes")
    nb.train(train.copy())
    nb.test(test.copy(), "test")

def runNeuralNetwork(train, test, hLayer=None, mode=None, momentumFactor=0.0):
    print "Neural Network ============= "
    print " - hidden layer: ",
    if hLayer is not None:
        print hLayer
    else:
        print " default (one hidden layer with node number = 2 * feature number)"

    print " - weight initialization mode: ",
    if mode is not None:
        print mode
    else:
        print "default"

    nn = Classifier("neural_network", hidden_layer=hLayer, weightInitMode=mode, momentumFactor=momentumFactor)
    nn.train(train.copy())
    nn.test(test.copy(), "test")


'''
=====================================================================
 Decision Tree
=====================================================================
'''
print "====== Decision Tree ========================================="
''' 1.Congressional Voting Records dataset '''
print "1. Congressional Voting Records dataset without pruning"
(trainingData, testData) = load_data.load_congress_data(0.6)
# ======================
runDecisionTree(trainingData, testData, "ig", 0.0)
# ======================
runDecisionTree(trainingData, testData, "igr", 0.0)

''' 2. MONKS Problem dataset '''
print ""
for i in [1, 2, 3]:
    print "2. MONKS Problem with Information Gain without pruning- dataset ", i
    (trainingData, testData) = load_data.load_monks(i)
    # ======================
    runDecisionTree(trainingData, testData, "ig", 0.0)
    # ======================
    runDecisionTree(trainingData, testData, "igr", 0.0)

''' 3. Iris dataset '''
print ""
print "3. Iris without pruning"
(trainingData, testData) = load_data.load_iris(0.6)
trainingData = np.floor(trainingData)
testData = np.floor(testData)
# ======================
runDecisionTree(trainingData, testData, "ig", 0.0)
# ======================
runDecisionTree(trainingData, testData, "igr", 0.0)

'''==================='''
print "1. Congressional Voting Records dataset with pruning"
(trainingData, testData) = load_data.load_congress_data(0.6)
# ======================
runDecisionTree(trainingData, testData, "ig", 0.05)
# ======================
runDecisionTree(trainingData, testData, "igr", 0.05)

''' 2. MONKS Problem dataset '''
print ""
for i in [1, 2, 3]:
    print "2. MONKS Problem with Information Gain with pruning- dataset ", i
    (trainingData, testData) = load_data.load_monks(i)
    # ======================
    runDecisionTree(trainingData, testData, "ig", 0.05)
    # ======================
    runDecisionTree(trainingData, testData, "igr", 0.05)

''' 3. Iris dataset '''
print ""
print "3. Iris with pruning"
(trainingData, testData) = load_data.load_iris(0.6)
trainingData = np.floor(trainingData)
testData = np.floor(testData)
# ======================
runDecisionTree(trainingData, testData, "ig", 0.05)
# ======================
runDecisionTree(trainingData, testData, "igr", 0.05)

'''
=====================================================================
 Naive Bayes
=====================================================================
'''
print ""
print "====== Naive Bayes ==========================================="
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
print "====== Neural Network ========================================"
print "1. Congressional Voting Records dataset without momentum"

(trainingData, testData) = load_data.load_congress_data(0.6)
# ======================
weightInitMode = None
hidden_layer   = [32]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.0)
# ======================
weightInitMode = "shallow"
hidden_layer   = [32]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.0)
# ======================
weightInitMode = "deep"
hidden_layer   = [32, 32, 32]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.0)


''' 2. MONKS Problem dataset '''
print ""
for i in [1, 2, 3]:
    print "2. MONKS Problem without momentum - dataset ", i
    (trainingData, testData) = load_data.load_monks(i)
    # ======================
    weightInitMode = None
    hidden_layer   = [14]
    runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.0)
    # ======================
    weightInitMode = "shallow"
    hidden_layer   = [14]
    runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.0)
    # ======================
    weightInitMode = "deep"
    hidden_layer   = [14, 14, 14]
    runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.0)


''' 3. Iris dataset '''
print ""
print "3. Iris with momentum"
(trainingData, testData) = load_data.load_iris(0.6)

# ======================
weightInitMode = None
hidden_layer   = [16]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.5)
# ======================
weightInitMode = "shallow"
hidden_layer   = [16]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.5)
# ======================
weightInitMode = "deep"
hidden_layer   = [16, 16, 16]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.5)

'''========'''
print "1. Congressional Voting Records dataset with momentum"

(trainingData, testData) = load_data.load_congress_data(0.6)
# ======================
weightInitMode = None
hidden_layer   = [32]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.5)
# ======================
weightInitMode = "shallow"
hidden_layer   = [32]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.5)
# ======================
weightInitMode = "deep"
hidden_layer   = [32, 32, 32]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.5)


''' 2. MONKS Problem dataset '''
print ""
for i in [1, 2, 3]:
    print "2. MONKS Problem with momentum - dataset ", i
    (trainingData, testData) = load_data.load_monks(i)
    # ======================
    weightInitMode = None
    hidden_layer   = [14]
    runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.5)
    # ======================
    weightInitMode = "shallow"
    hidden_layer   = [14]
    runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.5)
    # ======================
    weightInitMode = "deep"
    hidden_layer   = [14, 14, 14]
    runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.5)


''' 3. Iris dataset '''
print ""
print "3. Iris without momentum"
(trainingData, testData) = load_data.load_iris(0.6)

# ======================
weightInitMode = None
hidden_layer   = [16]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.5)
# ======================
weightInitMode = "shallow"
hidden_layer   = [16]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.5)
# ======================
weightInitMode = "deep"
hidden_layer   = [16, 16, 16]
runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, 0.5)
