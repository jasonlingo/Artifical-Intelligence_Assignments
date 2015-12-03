import numpy as np
from scipy import stats
import math
from classifier import Classifier
from collections import Counter

class NaiveBayes(Classifier):
    """
    A NaiveBayes classifier.
    """
    def __init__(self, examples, attributes):
        self.prob_y = {}
        self.prob_x_y = {}
        self.examples = examples
        self.attribute = attributes
        self.attribute_list = {}
        self.lable_list = []
        self.countTotal = {}

    def train(self):
        """
        Train the NaiveBayes for prior probability p(y) and posterior probability
        p(x|y) for all possible value for x and y. The probability was calculate:
                      number of times Y = y_k appears
        p(Y = y_k) = ---------------------------------
                        number of all examples
                                number of times X = x_j and Y = y_k appear together
        p(X = x_j | Y = y_k) = -----------------------------------------------------
                                    number of times Y = y_k
        The values for p(y) are stored in dictionary self.prob_y where keys are different
        lables and values are their probability. 
        The values for p(x|y) are stored in dictionary self.prob_x_y where key is represented
        as tuple (y,a,v), p(X_a = v | Y = y) is stored as its value.
        """                     
        self.countTotal = Counter(list(self.examples[:, 0]))
        
        for key in self.countTotal.keys():
            if key not in self.lable_list:
                self.lable_list.append(key)

        total_num = len(self.examples)

        for key in self.countTotal.keys():
            self.prob_y[key] = float(self.countTotal[key]) / total_num

        for a in self.attribute:
            self.attribute_list[a] = self.valueOfA(a, self.examples)

        for y in self.lable_list:

            for a in self.attribute:

                for v in self.valueOfA(a, self.examples):
                    num_y = self.countTotal[y]
                    num_x = 0
                    prob = 0
                    for e in self.examples:
                        if e[0] == y and e[a] == v:
                            num_x += 1

                    prob = float(num_x) / num_y
                    self.prob_x_y[(y,a,v)] = prob
        return

    def predict(self, data):
        """
        Predict the label of the given data. Compute the posterior of particular attribute
        value from probability dictionary derived in training step.
        Args:
            data: the given data
        Returns:
            a label = argmax_{y\in {label values}}{\prod_{j = 1}^{M}{p(x_j|y)}}p(y)
        """
        pred_prob = {}
        for y in self.lable_list:
            prob = self.prob_y[y]
            for a in self.attribute:
                prob *= self.prob_x_y[(y, a, data[a])]
            pred_prob[y] = prob
        pred_value = -1
        pred_lable = None
        for key in pred_prob.keys():
            if pred_prob[key] > pred_value:
                pred_value = pred_prob[key]
                pred_lable = key
        return pred_lable

    def valueOfA(self, a, ex):
        """
        Find the distinct values of attribute a among only the given examples
        (not all example from the beginning).

        Args:
            a: the attribute index
            ex: the given example
        Returns:
            a list of distinct attribute values
        """
        value = set()
        for e in ex:
            value.add(e[a])
        return list(value)