"""
Class for a classification algorithm.
"""

import numpy as np


class Classifier:

    def __init__(self, classifier_type, **kwargs):
        """
        Initializer. Classifier_type should be a string which refers
        to the specific algorithm the current classifier is using.
        Use keyword arguments to store parameters
        specific to the algorithm being used. E.g. if you were
        making a neural net with 30 input nodes, hidden layer with
        10 units, and 3 output nodes your initalization might look
        something like this:

        neural_net = Classifier(weights = [], num_input=30, num_hidden=10, num_output=3)

        Here I have the weight matrices being stored in a list called weights (initially empty).
        """
        self.classifier_type = classifier_type
        self.params = kwargs
        """
        The kwargs you inputted just becomes a dictionary, so we can save
        that dictionary to be used in other methods.
        """
        self.clf = None

    def train(self, training_data):
        """
        Data should be nx(m+1) numpy matrix where n is the
        number of examples and m is the number of features
        (recall that the first element of the vector is the label).

        I recommend implementing the specific algorithms in a
        seperate module and then determining which method to call
        based on classifier_type. E.g. if you had a module called
        neural_nets:

        if self.classifier_type == 'neural_net':
            import neural_nets
            neural_nets.train_neural_net(self.params, training_data)

        Note that your training algorithms should be modifying the parameters
        so make sure that your methods are actually modifying self.params

        You should print the accuracy, precision, and recall on the training data.
        """
        if self.classifier_type == "decision_tree":
            from DecisionTree import DecisionTree
            attributes = [i for i in range(1, len(training_data[0]))]  # attributes' indices
            attrValue = self.getAttrValue(training_data)
            print attributes
            print attrValue
            self.clf = DecisionTree(self.params["igMode"], training_data, attributes, None, attrValue)
            self.clf.train()

        elif self.classifier_type == "neural_network":
            from NeuralNetwork import NeuralNetwork
            featureNum = training_data.shape[1] - 1                    # minus the one for label
            # nodeNum = [featureNum, featureNum * 2, featureNum * 2, 1]  # FIXME: make it become a parameter
            nodeNum = [featureNum, featureNum * 2, 1]
            self.clf = NeuralNetwork(training_data, nodeNum)
            self.clf.train()

        elif self.classifier_type == "naive_bayes":
            pass

    def predict(self, data):
        """
        Predict class of a single data vector
        Data should be 1x(m+1) numpy matrix where m is the number of features
        (recall that the first element of the vector is the label).

        I recommend implementing the specific algorithms in a
        seperate module and then determining which method to call
        based on classifier_type.

        This method should return the predicted label.
        """
        return self.clf.predict(data)

    def test(self, test_data):
        """
        Data should be nx(m+1) numpy matrix where n is the
        number of examples and m is the number of features
        (recall that the first element of the vector is the label).

        You should print the accuracy, precision, and recall on the test data.
        """
        classLabel = self.getAttrValue(test_data)[0]

        count = {}
        for label in classLabel:
            count[label] = {}
            count[label]["correct"] = 0
            count[label]["incorrect"] = 0

        for data in test_data:
            if data[0] == self.predict(data):
                count[self.predict(data)]["correct"] += 1
            else:
                count[self.predict(data)]["incorrect"] += 1

        print count

        print "total accuracy              = ", sum([count[label]["correct"] for label in count]) / float(len(test_data))
        # 0: false; 1: true
        # for label in classLabel:  # FIXME
        #     print "precision for class '", label, "' = ", count[label]["correct"] / float(sum([count[label]["correct"]] + [count[label2]["incorrect"] for label2 in classLabel if label2 != label]))
        #     print "recall for class '", label, "'    = ", count[label]["correct"] / float(sum(count[label].values()))

    def getAttrValue(self, ex):
        attrValue = {}
        for i in range(len(ex[0])):
            # attrValue[i] = list(set((ex[:,i]))
            attrValue[i] = list(set([v for v in ex[:,i]]))
        return attrValue