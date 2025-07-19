from opti import Opti
import pandas as pd

class Rebalancer:

    def __init__(self, opti):

        self.opti = opti
        self.goal, self.difference, self.rebalance_df, self.full_names, self.original, self.original_weighted = None, None, None, None, None, None

        self.get_original()
        self.get_difference()
        self.get_full_names()
        self.get_df()


    def get_original(self):

        self.original_weighted = {}
        self.original = self.opti.portfolio.holdings.copy()
        total = sum(self.original.values())

        for ticker in self.original:
            self.original[ticker] /= total
            self.original_weighted[ticker] = 2*self.original[ticker]/3
            self.original[ticker] = str(round(100*self.original[ticker])) + '%'

        for ticker in self.opti.optimum_all:
            if ticker not in self.original_weighted:
                self.original_weighted[ticker] = 0

        if not self.original:
            self.original_weighted = None


    def get_difference(self):

        if not self.original_weighted:
            self.goal = {ticker: self.opti.optimum_all[ticker] * self.opti.portfolio.liquidity for ticker in self.opti.optimum_all}
        else:
            self.goal = {ticker: (self.opti.optimum_all[ticker]/3 + self.original_weighted[ticker]) * self.opti.portfolio.liquidity for ticker in self.opti.optimum_all}

        self.difference = self.goal.copy()

        for ticker in self.opti.portfolio.holdings:
            self.difference[ticker] -= self.opti.portfolio.holdings[ticker]

        for ticker in self.difference:
            self.difference[ticker] = round(self.difference[ticker])

        self.difference = {ticker: self.difference[ticker] for ticker in self.difference if self.difference[ticker]}


    def get_full_names(self):

        self.full_names = {ticker: self.opti.portfolio.data.etf_full_names.loc[ticker] for ticker in self.difference}


    def get_df(self):

        goal = self.goal.copy()
        for ticker in goal:
            goal[ticker] = str(round(100*goal[ticker]/self.opti.portfolio.liquidity)) + '%'
        goal = {ticker: goal[ticker] for ticker in goal if goal[ticker] != '0%'}

        self.rebalance_df = pd.DataFrame({
            'ETF': self.full_names,
            'Buy/Sell': self.difference,
            'Before': self.original,
            'After': goal
        }).reset_index().sort_values(by='Buy/Sell', ascending=False)

        self.rebalance_df.rename(columns={'index': 'Ticker'}, inplace=True)


