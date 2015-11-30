from classifier import Classifier
import numpy as np
from scipy.special import expit  # sigmoid function

class NeuralNetwork(Classifier):
    """
    A neural network classifier.
    """

    def __init__(self, training_data, nodeNum):
        # number of total training iteration
        self.ITERATION = 1000

        # for randomly initializing theta
        self.EPISLON = 0.1

        # learning rate
        self.ALPHA = 0.5

        # number of classes for the classification
        self.classNum = nodeNum[-1]

        # number of nodes in each layers
        self.nodeNum = {}
        i = 1
        for nn in nodeNum:
            self.nodeNum[i] = nn
            i += 1

        # number of total layers including the input and output layers
        self.layerNum = len(nodeNum)

        self.theta = {}
        # self.layers = {}
        self.delta = {}
        self.z = {}
        self.a = {}

        # get labels and nodes for the first layer (input layer)
        self.labels, self.a[1] = self.prepareForTraining(training_data)

    def prepareForTraining(self, training_data):
        """
        Split the labels and features from the input training_data.
        Args:
            training_data:
        Returns:
            labels: a (exampleNum x classNum) matrix that contains label for each training example
            examples: a (exampleNum x featureNum ) matrix for the input training example
        """
        labels = training_data.copy()[:, 0]          # get the first column (label) from the matrix
        labels = labels.reshape(labels.shape[0], 1)
        examples = training_data.copy()[:, 1:]       # get the rest columns (features of examples) from the matrix
        onesCol = np.ones((examples.shape[0], 1))
        examples = np.hstack((examples, onesCol))    # add one column (all ones) for bias
        return labels, examples

    def train(self):
        """
        Train this neural network with feed-forward and back-propagation.
        When training is finished, the theta can be used for prediction.
        Args:
            training_data: a matrix (numpy format).
        """
        print "Start training neural network..."

        self.initTheta()

        for iter in range(self.ITERATION):
            self.feedForward()
            self.backpropagation()
            self.updateTheta()

    def initTheta(self):
        """
        initialize each theta by
        theta = random matrix * 2 epislon - epislon
        so each element will be in the range [-epislon, epislon]
        Returns:

        """
        # print "Initialize theta"
        for i in range(1, self.layerNum):  # layer number starts from 1,  add one bias
            th = np.random.rand(self.nodeNum[i+1], self.nodeNum[i] + 1) * 2 * self.EPISLON - self.EPISLON
            self.theta[i] = th

    def feedForward(self):
        """
        Compute all the x in each layer except the first input layer.
        """
        # print "feedforward"

        for i in range(1, self.layerNum):
            self.z[i + 1] = self.a[i].dot(self.theta[i].T)     # z_{i+1} = a_i * theta_i
            self.a[i + 1] = expit(self.z[i + 1])                        # a_{i+1} = sigmoid(z_{i+1})

            # add bias column to each layer except the output layer
            if i + 1 < self.layerNum:
                onesCol = np.ones((self.a[i + 1].shape[0], 1))
                self.a[i + 1] = np.hstack((self.a[i + 1], onesCol))

    def backpropagation(self):
        """
        Compute all the delta for each layer except the first input layer.
        """
        # print "backpropagation"

        # compute the delta for the output layer delta = (y - a) .* a .* (1 - a)
        self.delta[self.layerNum] = (self.labels - self.a[self.layerNum]) * \
                                     self.a[self.layerNum] * (1 - self.a[self.layerNum])
        # self.delta[self.layerNum] = (self.labels - self.a[self.layerNum])


        # compute the delta for other layers (from layerNum - 1 to 1)
        # no delta for input layer
        for l in range(self.layerNum - 1, 1, -1):
            # print l, "-th layer -------------------------"
            # print self.a[l].shape
            # print self.theta[l].shape
            # if l + 1 != self.layerNum:
            #     print self.delta[l+1][:,:-1].shape
            # else:
            #     print self.delta[l+1].shape
            # print "-------------------------------------"

            if l + 1 != self.layerNum:
                preDelta = self.delta[l+1][:, :-1]  # ignore the bias
            else:
                preDelta = self.delta[l+1]

            # self.delta[l] = self.a[l] * (1 - self.a[l]) * (preDelta.dot(self.theta[l]))
            self.delta[l] = preDelta.dot(self.theta[l])
            # print self.delta[l].shape

    def updateTheta(self):
        """
        Update every theta in network using deltas.
        for each layer of theta:
            theta_ij = theta_ij + self.ALPHA * a_i * delta_j
        """
        # print "udpate theta"
        for l in range(self.layerNum - 1, 0, -1):
            # print self.theta[l].shape
            if l + 1 < self.layerNum:
                preDelta = self.delta[l+1][:, :-1].T
            else:
                preDelta = self.delta[l+1].T
            # print preDelta.shape
            # print self.a[l].shape
            self.theta[l] += self.ALPHA * (preDelta.dot(self.a[l]))

    def predict(self, data):
        data = data.reshape(1, data.shape[0])
        # print "the dimension of input data is ", data.shape
        for l in range(1, self.layerNum):
            # print l, "-th layer"
            # print data.shape
            # print transpose(self.theta[l]).shape
            data = data.dot(self.theta[l].T)
            data = expit(data)
            if l + 1 < self.layerNum:
                onesCol = np.ones((data.shape[0], 1))
                data = np.hstack((data, onesCol))

        if self.classNum > 2:
            maxV = -1
            maxIdx = None
            i = 0
            for d in data:
                if d > maxV:
                    maxV = d
                    maxIdx = i
                i += 1
            return maxIdx
        else:
            if data.max() >= 0.5:
                return 1
            else:
                return 0
