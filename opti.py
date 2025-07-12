import numpy as np
from matplotlib import pyplot as plt
from portfolio import Portfolio
from scipy.optimize import minimize


class Opti:

    constraints = [{'type': 'eq', 'fun': lambda w: Opti.abs_sum(w) - 1, 'tol': 1e-3}]
    solver_method = 'SLSQP'

    def __init__(self, portfolio):

        self.objective = None
        self.optimum = None
        self.optimum_all = None
        self.w_opt = None
        self.goal = None
        self.difference = None

        self.portfolio = portfolio
        self.n = len(self.portfolio.etf_list)
        self.w0 = np.full(self.n, 1/self.n)
        self.bounds = ([(-1, 1)] if self.portfolio.allow_short else [(0, 1)]) * self.n
        self.cov = self.portfolio.data.excess_returns.cov().values
        self.get_objective()
        self.optimize()
        self.rebalance()


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

        self.w_opt = opt.x
        self.w_opt = [0 if abs(w) < 0.01 else w for w in self.w_opt]
        self.w_opt /= Opti.abs_sum(self.w_opt)

        self.optimum_all = {tick: w for tick, w in zip(self.portfolio.etf_list, self.w_opt)}
        self.optimum = {ticker: self.optimum_all[ticker] for ticker in self.optimum_all if self.optimum_all[ticker] != 0}
        self.goal = {ticker: self.optimum_all[ticker] * self.portfolio.liquidity for ticker in self.optimum_all}


    def plot_optimum(self):

        plt.bar(self.optimum.keys(), self.optimum.values(), color=self.portfolio.color_plot)
        plt.show()


    def plot_in_sample(self):

        returns = self.portfolio.data.returns[self.optimum.keys()]
        weights = list(self.optimum.values())
        cumulative = ((1 + returns @ weights).cumprod()-1)*100

        plt.plot(cumulative, label=self.portfolio.name)

        spy = (self.portfolio.data.spy / self.portfolio.data.spy.iloc[0] - 1) * 100
        plt.plot(spy, label='SPY', ls='--')

        rf_rate = ((self.portfolio.data.rf_rate + 1).cumprod() - 1) * 100
        plt.plot(rf_rate, label='rate', ls='--')

        plt.axhline(0, color='black')
        plt.ylabel('%')
        plt.legend()
        plt.grid()
        plt.show()


    def rebalance(self):
        self.difference = self.goal.copy()

        for ticker in self.portfolio.holdings:
            self.difference[ticker] -= self.portfolio.holdings[ticker]

        for ticker in self.difference:
            if self.difference[ticker] != 0:
                self.difference[ticker] -= np.sign(self.difference[ticker])

        self.difference = {ticker: int(self.difference[ticker]) for ticker in self.difference if self.difference[ticker]}



Opti(Portfolio())