//--- Nueral Net ----//

class Nueral_Network(object):
    def __init__(self):
        #Define HyperParameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        #Weights (Parameters)
        self.W1 = np.random.randn(\
        self.inputLayerSize, self.hiddenLayerSize)

        self.W2 = np.random.randn(\
        self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        #Propogate inputs through network
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(z):
        #Apply sigmoid activation function
        return 1/(1+np.exp(-z))

    def sigmoidPrime(z):
        #Derivative of Sigmoid Function
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, X, y):
        #Compute cost of error from predictions
        return 0.5*sum((y-yHat)**2)

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W1 and W2
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat) \
                    ,self.sigmoidPrime(self.z3))

        dJdW2 =     np.dot(self.a2.T, delta3)

        delta2 =    np.dot(delta3, self.W3.T) * \
                    self.sigmoidPrime(self.z2)

        dJdW1 =     np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def getParams(self):
        #Get W1 and W2 Rolled into vector
        params = np.concatenate((self.W1.ravel(),\
                                self.W2.ravel()))
        return params

    def setParams(self, params):
        #Set W1 and W2 using single parameter vector
        W1_start = 0
        W1_end =    self.hiddenLayerSize * \
                    self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end],\
        (self.inputLayerSize, self.hiddenLayerSize))

        W2_end = W1_end +   self.hiddenLayerSize * \
                            self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], \
                            (self.hiddenLayerSize,\
                            self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X,y)
        return np.concatenate(( dJdW1.ravel(),\
                                dJdW2.ravel()))

    def computeNumericalGradients(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set pertubation vector
            perturb[p] = e
            N.setParams(paramsInitial+perturb)
            loss2 = N.costFunction(X,y)

            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X,y)

            #Compute Numerical Gradient:
            numgrad[p] = (loss2-loss1)/(2*e)

            #Return the value we changed back to zero
            perturb[p] = 0

        #Return Params to original value
        N.setParams(paramsInitial)

        return numgrad

class trainer(object):
    def __init__(self, N):
        #Make local reference to Nueral network
        self.N = N

    def costFuncitonWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(\
           self.N.costFunction(self.X, self.y))

    def train(self, X, y):
        #Make internal variable for callback function
        self.X = X
        self.y = y

        #Make empty list to store costs
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(\
            self.costFunctionWrapper,\
            params0,\
            jac = True,\
            method = 'BFGS',\
            args = {X,y},\
            options = options,\
            callback = self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res
