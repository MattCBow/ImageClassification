# Artificial Intelligence Perceptron Implementation
"""
from ImageClassifier import ImageClassifier
from Perceptron import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
classifier = ImageClassifier()
X = np.array(classifier.data['digit']['test']['features'][0:200], dtype=float)
Y = np.array(classifier.data['digit']['test']['classification'][0:200], dtype=float)
X = X/100
Y = Y/10
percep_net = Perceptron(X.shape[1],Y.shape[1])
trainer = NodeTrainer(percep_net)
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

    def costFunction(self, X, y):
        yHat = self.forward(X)
        return 0.5*sum((y-yHat)**2)

    def costFunctionPrime(self, X, y):
        yHat = self.forward(X)
        J_in = y-yHat
        dJ_dW = np.dot(J_in.T, X)
        return (-1.0)*dJ_dW

    def getParams(self):
        params = self.W.ravel()
        return params

    def setParams(self, params):
        self.W = np.reshape(params, (self.input_size, self.output_size))

    def computeGradients(self, X, y):
        dJ_dW = self.costFunctionPrime(X,y)
        return dJ_dW.ravel()

class NodeTrainer(object):
    #-------------INIT WITH NUERAL NET-----------
    def __init__(self, model):
        self.model = model
        self.J = []
        self.count=0

    #----------------HELPER FUNCTIONS-----------
    def costFunctionWrapper(self, params, X, y):
        self.model.setParams(params)
        cost = self.model.costFunction(X, y)
        grad = self.model.computeGradients(X,y)
        return cost, grad

    def callbackF(self, params):
        self.count+=1
        print self.count, ' - finish'
        self.model.setParams(params)
        self.J.append(self.model.costFunction(self.X, self.y))

    #-----PROPOAGATE EVIDENCE AND USE BFGS OPTIMIZATION-------
    def train(self, X, y):
        self.X = X
        self.y = y
        self.J = []
        params0 = self.model.getParams()
        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(   self.costFunctionWrapper,
                                    params0,
                                    jac = True,
                                    method = 'BFGS',
                                    args = (X,y),
                                    options = options,
                                    callback = self.callbackF)
        self.model.setParams(_res.x)
        self.optimizationResults = _res
