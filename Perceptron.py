# Artificial Intelligence Perceptron Implementation
"""
from ImageClassifier import ImageClassifier
from Perceptron import *
from ModelTrainer import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
classifier = ImageClassifier()
X = np.array(classifier.data['digit']['test']['features'][0:200], dtype=float)
Y = np.array(classifier.data['digit']['test']['classification'][0:200], dtype=float)
X = X/100
Y = Y/10
percep_net = Perceptron(X.shape[1],Y.shape[1])
trainer = ModelTrainer(percep_net)
trainer.train(X,Y)
"""

from random import choice
import numpy as np
from scipy import optimize

class Perceptron(object):

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.randn(self.input_size, self.output_size)

    def forward(self, X):
        yHat = np.dot(X,self.W)
        return yHat

    def cost_function(self, X, y):
        yHat = self.forward(X)
        return 0.5*sum((y-yHat)**2)

    def cost_function_prime(self, X, y):
        yHat = self.forward(X)
        J_in = y-yHat
        dJ_dW = np.dot(J_in.T, X)
        return (-1.0)*dJ_dW

    def get_params(self):
        params = self.W.ravel()
        return params

    def set_params(self, params):
        self.W = np.reshape(params, (self.input_size, self.output_size))

    def compute_gradients(self, X, y):
        dJ_dW = self.cost_function_prime(X,y)
        return dJ_dW.ravel()
