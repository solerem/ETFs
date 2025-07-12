import numpy as np
from portfolio import Portfolio


class Opti:

    eq_cons = {'type': 'eq', 'fun': lambda w: Opti.abs_sum(w) - 1, 'tol': 1e-3}

    def __init__(self, portfolio):

        self.portfolio = portfolio
        self.n = len(self.portfolio.etf_list)
        self.w0 = np.full(self.n, 1/self.n)
        self.bounds = ([(-1, 1)] if self.portfolio.allow_short else [(0, 1)]) * self.n
        self.cov = self.portfolio.data.excess_returns.cov().values


    @staticmethod
    def abs_sum(lst):

        return sum([abs(x) for x in lst])

Opti(Portfolio())