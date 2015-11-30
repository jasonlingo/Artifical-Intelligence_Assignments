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


'''
=====================================================================
 Decision Tree
=====================================================================
'''
# print "====== Decision Tree ========================================="
# ''' 1.Congressional Voting Records dataset '''
# print "Congressional Voting Records with Information Gain"
# (trainingData, testData) = load_data.load_congress_data(0.6)
# dt = Classifier("decision_tree", igMode="ig")
# dt.train(trainingData)
# dt.test(testData)
#
# print "Congressional Voting Records with Information Gain Ratio"
# dt = Classifier("decision_tree", igMode="igr")
# dt.train(trainingData)
# dt.test(testData)
#
# ''' 2. MONKS Problem dataset '''
# print ""
# print "MONKS Problem with Information Gain"
# (trainingData, testData) = load_data.load_monks(2)
# dt = Classifier("decision_tree", igMode="ig")
# dt.train(trainingData)
# dt.test(testData)
#
# print "Congressional Voting Records with Information Gain Ratio"
# dt = Classifier("decision_tree", igMode="igr")
# dt.train(trainingData)
# dt.test(testData)
#
# ''' 3. Iris dataset '''
# print ""
# print "3. Iris with Information Gain"
# (trainingData, testData) = load_data.load_iris(0.6)
#
# for i in range(len(trainingData[0])):
#     print i
#     print sorted(set([d[i] for d in trainingData]))
#     print sorted(set([d[i] for d in testData]))
#
# dt = Classifier("decision_tree", igMode="ig")
# dt.train(trainingData)
# dt.test(testData)
#
# print "Congressional Voting Records with Information Gain Ratio"
# dt = Classifier("decision_tree", igMode="igr")
# dt.train(trainingData)
# dt.test(testData)


'''
=====================================================================
 Naive Bayes
=====================================================================
'''
# print "====== Naive Bayes ==========================================="
''' 1.Congressional Voting Records dataset '''
# print "1. Congressional Voting Records with Information Gain"
# (trainingData, testData) = load_data.load_congress_data(0.6)
# dt = Classifier("decision_tree", igMode="ig")
# dt.train(trainingData)
# dt.test(testData)
#
# print "1. Congressional Voting Records with Information Gain Ratio"
# dt = Classifier("decision_tree", igMode="igr")
# dt.train(trainingData)
# dt.test(testData)

''' 2. MONKS Problem dataset '''



''' 3. Iris dataset '''



'''
=====================================================================
 Neural Network
=====================================================================
'''
print "====== Neural Network ========================================"
''' 1.Congressional Voting Records dataset '''
print "1. Congressional Voting Records with Information Gain"
(trainingData, testData) = load_data.load_congress_data(0.6)
dt = Classifier("neural_network")
dt.train(trainingData)
dt.test(testData)


''' 2. MONKS Problem dataset '''



''' 3. Iris dataset '''