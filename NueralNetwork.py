#Artificial Intelligence Nueral Network Implementation
"""
from ImageClassifier import ImageClassifier
from NueralNetwork import NueralNetwork, NodeTrainer
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
classifier = ImageClassifier()
X = np.array(classifier.data['digit']['test']['features'][0:200], dtype=float)
Y = np.array(classifier.data['digit']['test']['classification'][0:200], dtype=float)
X = X/100
Y = Y/10
nueral_net = NueralNetwork(input_size=X.shape[1],output_size=Y.shape[1])
trainer = NodeTrainer(nueral_net)
trainer.train(X,Y)

X = X/np.amax(X, axis=0)
Y = y/100
nueral_net = NueralNetwork(input_size=X.shape[1],output_size=Y.shape[1])
trainer = NodeTrainer(nueral_net)
trainer.train(X,Y)
"""

import numpy as np
from scipy import optimize

class NueralNetwork(object):

    #-----------RANDOM STARTER WEIGHTS---------------
    def __init__(self, input_size, output_size):
        self.inputLayerSize = input_size
        self.outputLayerSize = output_size
        self.hiddenLayerSize = 3
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    #---------------NUERAL NET FUNCTION-------------
    def forward(self, X):
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    #---------------SIGMOID FUNCTION-------------
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    #---------------SIGMOID DERIVATIVE-------------
    def sigmoidPrime(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    #---------------COST FUNCTION-------------
    def costFunction(self, X, y):
        yHat = self.forward(X)
        return 0.5*sum((y-yHat)**2)

    #---------------COST DERIVATIVE-------------
    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        return dJdW1, dJdW2

    #----------------HELPER FUNCTIONS-----------
    def getParams(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        W1_start = 0
        W1_end = self.hiddenLayerSize *  self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X,y)
        return np.concatenate(( dJdW1.ravel(), dJdW2.ravel()))

    #-------------NUMERICAL GRADIENT DESCENT------------
    def computeNumericalGradients(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4
        for p in range(len(paramsInitial)):
            perturb[p] = e
            N.setParams(paramsInitial+perturb)
            loss2 = N.costFunction(X,y)
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X,y)
            numgrad[p] = (loss2-loss1)/(2*e)
            perturb[p] = 0
        N.setParams(paramsInitial)
        return numgrad

class NodeTrainer(object):
    #-------------INIT WITH NUERAL NET-----------
    def __init__(self, nueral_net):
        self.nueral_net = nueral_net
        self.J = []
        self.count=0

    #----------------HELPER FUNCTIONS-----------
    def costFunctionWrapper(self, params, X, y):
        self.nueral_net.setParams(params)
        cost = self.nueral_net.costFunction(X, y)
        grad = self.nueral_net.computeGradients(X,y)
        return cost, grad

    def callbackF(self, params):
        self.count+=1
        print self.count, ' - finish'
        self.nueral_net.setParams(params)
        self.J.append(self.nueral_net.costFunction(self.X, self.y))

    #-----PROPOAGATE EVIDENCE AND USE BFGS OPTIMIZATION-------
    def train(self, X, y):
        self.X = X
        self.y = y
        self.J = []
        params0 = self.nueral_net.getParams()
        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(   self.costFunctionWrapper,
                                    params0,
                                    jac = True,
                                    method = 'BFGS',
                                    args = (X,y),
                                    options = options,
                                    callback = self.callbackF)
        self.nueral_net.setParams(_res.x)
        self.optimizationResults = _res



"""
import numpy as np
from scipy import optimize

class NueralNetwork(object):

    #-----------RANDOM STARTER WEIGHTS---------------
    def __init__(self, input_size, output_size):
        self.inputlayer_size = input_size
        self.outputlayer_size = output_size
        self.hiddenlayer_size = 3
        self.W1 = np.random.randn(self.inputlayer_size, self.hiddenlayer_size)
        self.W2 = np.random.randn(self.hiddenlayer_size, self.outputlayer_size)

    #---------------NUERAL NET FUNCTION-------------
    def forward(self, X):
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    #---------------SIGMOID FUNCTION-------------
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    #---------------SIGMOID DERIVATIVE-------------
    def sigmoid_prime(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    #---------------COST FUNCTION-------------
    def cost_function(self, X, y):
        yHat = self.forward(X)
        return 0.5*sum((y-yHat)**2)

    #---------------COST DERIVATIVE-------------
    def cost_function_prime(self, X, y):
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y-self.yHat), self.sigmoid_prime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoid_prime(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        return dJdW1, dJdW2

    #----------------HELPER FUNCTIONS-----------
    def get_params(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def set_params(self, params):
        W1_start = 0
        W1_end = self.hiddenlayer_size *  self.inputlayer_size
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputlayer_size, self.hiddenlayer_size))
        W2_end = W1_end + self.hiddenlayer_size * self.outputlayer_size
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenlayer_size, self.outputlayer_size))

    def compute_gradients(self, X, y):
        dJdW1, dJdW2 = self.cost_function_prime(X,y)
        return np.concatenate(( dJdW1.ravel(), dJdW2.ravel()))

    #-------------NUMERICAL GRADIENT DESCENT------------
    def compute_numerical_gradients(N, X, y):
        paramsInitial = N.get_params()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4
        for p in range(len(paramsInitial)):
            perturb[p] = e
            N.set_params(paramsInitial+perturb)
            loss2 = N.cost_function(X,y)
            N.set_params(paramsInitial - perturb)
            loss1 = N.cost_function(X,y)
            numgrad[p] = (loss2-loss1)/(2*e)
            perturb[p] = 0
        N.set_params(paramsInitial)
        return numgrad

    #----------------HELPER FUNCTIONS-----------
    def cost_function_wrapper(self, params, X, y):
        self.set_params(params)
        cost = self.cost_function(X, y)
        grad = self.compute_gradients(X,y)
        return cost, grad

    def callbackF(self, params):
        self.count+=1
        print self.count, ' - finish'
        self.set_params(params)
        self.J.append(self.cost_function(self.X, self.y))

    #-----PROPOAGATE EVIDENCE AND USE BFGS OPTIMIZATION-------
    def train(self, X, y):
        self.X = X
        self.y = y
        self.J = []
        self.count = 0
        params0 = self.get_params()
        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(   self.cost_function_wrapper,
                                    params0,
                                    jac = True,
                                    method = 'BFGS',
                                    args = (X,y),
                                    options = options,
                                    callback = self.callbackF)
        self.set_params(_res.x)
        self.optimization_results = _res

"""
"""
from ImageClassifier import ImageClassifier
from NueralNetwork import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
classifier = ImageClassifier()

X = np.array(classifier.data['digit']['test']['features'][0:20], dtype=float)
Y = np.array(classifier.data['digit']['test']['classification'][0:20], dtype=float)

nueral_net = NueralNetwork(X.shape[1],Y.shape[1])
nueral_net.train(X,Y)



X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)
X = X/np.amax(X, axis=0)
Y = y/100

nueral_net = NueralNetwork(input_size=X.shape[1],output_size=Y.shape[1])
trainer = NodeTrainer(nueral_net)

"""
