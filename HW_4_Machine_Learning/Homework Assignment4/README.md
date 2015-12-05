Artificial Intelligence 600.435 Homework 4 Machine Learning
Name: 
    Chenhao Han / chan17@jh.edu
    Li-Yi Lin / llin34@jhu.edu

Breakdown of Work:
    Decision Tree: Li-Yi Lin and Chenhao Han
    Naive Bayes: Chenhao Han
    Neural Network: Li-Yi Lin

Class Description:
    DecisionTree:
        Each object of this class represents one node in a decision tree. The decision tree will be built using this class recursively.

    NaiveBayes:
        This class will compute then store the prior and posterior probabilities in the training process. Then this class will use these values for predicting label for the given example.
    
    NeuralNetwork:
        This class will perform feed-forward and back-propagation on the training data to learn the weight for each layer.
        When predicting a label for a given example, it will compute the label using feed-forward with the learned weight.

Please run the "train_and_test.py" to get the results. 
Feel free to change the parameters in the "train_and_test.py" file.
The paremeters (default): 
    training ratio = 0.6
    pruningThreshold = 0.05
    momentumFactor = 0.8

Please refer to Homework_Assignment_4.pdf for detailed output results.

Thank you!