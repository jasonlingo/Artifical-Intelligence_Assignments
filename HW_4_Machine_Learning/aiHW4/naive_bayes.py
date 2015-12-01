import numpy as np
from scipy import stats
import math
from collections import Counter

class Naive_bayes:

    def __init__(self, examples, attributes):
        self.prob_y = {}
        self.prob_x_y = {}
        self.examples = examples
        self.attribute = attributes
        self.attribute_list = {}
        self.lable_list = []
        self.countTotal = {}

    def train(self):

        self.countTotal = Counter(list(self.examples[:, 0]))
        # print countTotal
        for key in self.countTotal.keys():
            if key not in self.lable_list:
                self.lable_list.append(key)

        # print 'label list is:', self.lable_list
        total_num = len(self.examples)

        for key in self.countTotal.keys():
            self.prob_y[key] = float(self.countTotal[key]) / total_num

        # print self.prob_y

        # print self.attribute

        for a in self.attribute:
            self.attribute_list[a] = self.valueOfA(a, self.examples)

        # print self.attribute_list

        for y in self.lable_list:
            # print y
            for a in self.attribute:
                #print y, a
                for v in self.valueOfA(a, self.examples):
                    # print y, a, v
                    num_y = self.countTotal[y]
                    # print num_y
                    num_x = 0
                    prob = 0
                    for e in self.examples:
                        if e[0] == y and e[a] == v:
                            num_x += 1

                    prob = float(num_x) / num_y
                    self.prob_x_y[(y,a,v)] = prob
                    # print prob
        # print self.prob_x_y

        return

    def predict(self, data):
        pred_prob = {}
        for y in self.lable_list:
            prob = self.prob_y[y]
            for a in self.attribute:
                prob *= self.prob_x_y[(y, a, data[a])]
            pred_prob[y] = prob
            # print 'probability is:', prob
        pred_value = -1
        pred_lable = None
        for key in pred_prob.keys():
            if pred_prob[key] > pred_value:
                pred_value = pred_prob[key]
                pred_lable = key
        return pred_lable

    def valueOfA(self, a, ex):
        value = set()
        for e in ex:
            value.add(e[a])
        return list(value)