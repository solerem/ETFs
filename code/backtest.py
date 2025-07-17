import numpy as np
import pandas as pd
from portfolio import Portfolio
from opti import Opti
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import io
import base64
from dash import html
from data import Data

class Backtest:

    ratio_train_test = .9
    capital_deposit_mensuel = 100

    def __init__(self, portfolio):

        self.portfolio = portfolio
        self.w_opt, self.returns, self.n, self.cutoff, self.index = None, None, None, None, None
        self.parse_data()
        self.get_returns()


    def parse_data(self):

        self.n = len(self.portfolio.data.nav)
        self.cutoff = round(Backtest.ratio_train_test * self.n)
        self.index = list(self.portfolio.data.nav.index)

        self.w_opt = []
        for i in tqdm(range(self.cutoff, self.n)):
            portfolio = Portfolio(risk=self.portfolio.risk, currency=self.portfolio.currency,
                          allow_short=self.portfolio.allow_short, static=True, backtest=self.index[i])
            self.w_opt.append(Opti(portfolio).w_opt)


    def get_returns(self):

        self.returns = []
        for i, w in zip(range(self.cutoff, self.n), self.w_opt):
            self.returns.append(Data.get_test_data_backtest(self.portfolio.data.returns, self.index[i]) @ w)
        self.returns = pd.Series(self.returns)
        self.returns.index = self.index[self.cutoff:]


    def plot_backtest(self):

        cumulative = (1 + self.returns).cumprod()

        fig, ax = plt.subplots()
        ax.plot((cumulative - 1) * 100, label=self.portfolio.name + f' ({self.portfolio.currency})')

        spy = self.portfolio.data.spy.copy()
        spy = spy.loc[self.index[self.cutoff]:]
        spy = (spy / spy.iloc[0] - 1) * 100
        ax.plot(spy, label=f'Total stock market ({self.portfolio.currency})', linestyle='--')

        rf_rate = ((self.portfolio.data.rf_rate.loc[self.index[self.cutoff]:] + 1).cumprod() - 1) * 100
        ax.plot(rf_rate, label='Rate', linestyle='--')

        ax.axhline(0, color='black')

        nb_years = int(Data.period[:-1])*(1-Backtest.ratio_train_test)
        pa_perf = round(((cumulative[-1]) ** (1/nb_years) - 1)*100)

        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = round(drawdown.min()*100)
        plt.setp(ax.get_xticklabels(), rotation=45)

        ax.set_title(f'Backtest ({pa_perf}% p.a., {max_drawdown}% max drawdown)')

        ax.set_ylabel('%')
        ax.legend()
        ax.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        encoded = base64.b64encode(buf.read()).decode('utf-8')
        img_src = f"data:image/png;base64,{encoded}"

        return html.Img(src=img_src, style={"maxWidth": "100%", "height": "auto"})