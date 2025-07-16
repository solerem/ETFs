from portfolio import Portfolio
from opti import Opti
from tqdm import tqdm


class Backtest:

    def __init__(self, portfolio, static=False):

        self.portfolio = portfolio
        n = len(self.portfolio.data.nav)
        cutoff = round(2*n/3)
        index = list(self.portfolio.data.nav.index)
        for i in tqdm(range(cutoff, n)):
            x = Portfolio(risk=self.portfolio.risk, currency=self.portfolio.currency, allow_short=self.portfolio.allow_short, static=True, backtest=index[i])
            Opti(x)

