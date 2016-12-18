# Artificial Intelligence for

from scipy import optimize

class ModelTrainer(object):
    def __init__(self, model):
        self.model = model
        self.J = []
        self.count=0

    def cost_function_wrapper(self, params, X, y):
        self.model.set_params(params)
        cost = self.model.cost_function(X, y)
        grad = self.model.compute_gradients(X,y)
        return cost, grad

    def callbackF(self, params):
        self.count+=1
        # print self.count, ' - finish'
        self.model.set_params(params)
        self.J.append(self.model.cost_function(self.X, self.y))

    def train(self, X, y):
        self.X = X
        self.y = y
        self.J = []
        params0 = self.model.get_params()
        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(   self.cost_function_wrapper,
                                    params0,
                                    jac = True,
                                    method = 'BFGS',
                                    args = (X,y),
                                    options = options,
                                    callback = self.callbackF)
        self.model.set_params(_res.x)
        self.optimization_results = _res
