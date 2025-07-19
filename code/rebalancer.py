from opti import Opti


class Rebalancer:

    def __init__(self, opti):

        self.opti = opti
        self.goal, difference = None, None


    def rebalance(self):

        self.goal = {ticker: self.opti.optimum_all[ticker] * self.opti.portfolio.liquidity for ticker in self.opti.optimum_all}
        self.difference = self.goal.copy()

        for ticker in self.portfolio.holdings:
            self.difference[ticker] -= self.portfolio.holdings[ticker]

        for ticker in self.difference:
            self.difference[ticker] = round(self.difference[ticker])

        self.difference = {ticker: int(self.difference[ticker]) for ticker in self.difference if
                           self.difference[ticker]}