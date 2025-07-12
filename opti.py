import numpy as np
from portfolio import Portfolio
from scipy.optimize import minimize


class Opti:

    constraints = [{'type': 'eq', 'fun': lambda w: Opti.abs_sum(w) - 1, 'tol': 1e-3}]
    solver_method = 'SLSQP'

    def __init__(self, portfolio):

        self.objective = None
        self.optimum = None

        self.portfolio = portfolio
        self.n = len(self.portfolio.etf_list)
        self.w0 = np.full(self.n, 1/self.n)
        self.bounds = ([(-1, 1)] if self.portfolio.allow_short else [(0, 1)]) * self.n
        self.cov = self.portfolio.data.excess_returns.cov().values
        self.get_objective()
        self.optimize()


    @staticmethod
    def abs_sum(lst):

        return sum([abs(x) for x in lst])


    def get_objective(self):

        def f(w):
            excess_series = self.portfolio.data.excess_returns @ w
            mean = excess_series.mean()
            return self.portfolio.weight_cov * w @ self.cov @ w - mean

        self.objective = f


    def optimize(self):
        opt = minimize(lambda w: self.objective(w), self.w0, method=Opti.solver_method, bounds=self.bounds, constraints=Opti.constraints, options={'ftol': 1e-6, 'maxiter': 1000})

        if not opt.success:
            print(f"Optimization failed: {opt.message}")
            return None

        w_opt = opt.x
        w_opt = [0 if abs(w) < 0.01 else w for w in w_opt]
        w_opt /= Opti.abs_sum(w_opt)

        self.optimum = {tick: w for tick, w in zip(self.portfolio.etf_list, w_opt)}


Opti(Portfolio())