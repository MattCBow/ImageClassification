#Artificial Intelligence Nueral Network Implementation
"""
from ImageClassifier import ImageClassifier
from NueralNetwork import *
from ModelTrainer import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
classifier = ImageClassifier()
X = np.array(classifier.data['digit']['test']['features'][0:200], dtype=float)
Y = np.array(classifier.data['digit']['test']['classification'][0:200], dtype=float)
X = X/100
Y = Y/10
nueral_net = NueralNetwork(input_size=X.shape[1],output_size=Y.shape[1])
trainer = ModelTrainer(nueral_net)
trainer.train(X,Y)
"""

import numpy as np
from scipy import optimize

class NueralNetwork(object):

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = 3
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, X):
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoid_prime(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    def cost_function(self, X, y):
        yHat = self.forward(X)
        return 0.5*sum((y-yHat)**2)

    def cost_function_prime(self, X, y):
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y-self.yHat), self.sigmoid_prime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoid_prime(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        return dJdW1, dJdW2

    def get_params(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def set_params(self, params):
        W1_start = 0
        W1_end = self.hidden_size *  self.input_size
        self.W1 = np.reshape(params[W1_start:W1_end], (self.input_size, self.hidden_size))
        W2_end = W1_end + self.hidden_size * self.output_size
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hidden_size, self.output_size))

    def compute_gradients(self, X, y):
        dJdW1, dJdW2 = self.cost_function_prime(X,y)
        return np.concatenate(( dJdW1.ravel(), dJdW2.ravel()))

    def compute_numerical_gradients(N, X, y):
        params_initial = N.get_params()
        numgrad = np.zeros(params_initial.shape)
        perturb = np.zeros(params_initial.shape)
        e = 1e-4
        for p in range(len(params_initial)):
            perturb[p] = e
            N.set_params(params_initial+perturb)
            loss2 = N.cost_function(X,y)
            N.set_params(params_initial - perturb)
            loss1 = N.cost_function(X,y)
            numgrad[p] = (loss2-loss1)/(2*e)
            perturb[p] = 0
        N.set_params(params_initial)
        return numgrad
