# from videoSupport import *
from scipy import optimize


class trainer(object):
    def __init__(self, N):
        # Make local reference to Neural Networks
        self.N = N

    def cost_function_wrapper(self, params, X, y):
        self.N.set_params(params)
        cost = self.N.cost_function(X, y)
        grad = self.N.compute_gradients(X, y)
        return cost, grad

    def callback_f(self, params):
        self.N.set_params(params)
        self.J.append(self.N.cost_function(self.X, self.y))

    def train(self, X, y):
        # Make internal variables for callback function
        self.X = X
        self.y = y

        # Make empty list to store costs
        self.J = []

        params0 = self.N.get_params()
        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.cost_function_wrapper, params0, jac=True, method="BFGS", args=(X, y),
                                 options=options, callback=self.callback_f)

        self.N.set_params(_res.x)
        self.optimization_results = _res
