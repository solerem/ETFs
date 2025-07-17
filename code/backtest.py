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

    ratio_train_test = .95

    def __init__(self, opti):

        self.opti = opti
        self.portfolio = self.opti.portfolio
        self.w_opt, self.returns, self.n, self.cutoff, self.index = None, None, None, None, None
        self.parse_data()
        self.smoothen_weights()
        self.get_returns()


    def parse_data(self):

        self.n = len(self.portfolio.data.nav)
        self.cutoff = round(Backtest.ratio_train_test * self.n)
        self.index = list(self.portfolio.data.nav.index)

        self.w_opt = pd.DataFrame({ticker: [] for ticker in self.portfolio.etf_list})
        for i in tqdm(range(self.cutoff, self.n)):
            portfolio = Portfolio(risk=self.portfolio.risk, currency=self.portfolio.currency,
                          allow_short=self.portfolio.allow_short, static=True, backtest=self.index[i])
            optimum = Opti(portfolio).optimum_all
            self.w_opt.loc[self.index[i]] = optimum


    def smoothen_weights(self):

        smoothed_df = pd.DataFrame(index=self.w_opt.index, columns=self.w_opt.columns, dtype=float)
        smoothed_df.iloc[0] = self.w_opt.iloc[0]

        for t in range(1, len(self.w_opt)):
            smoothed_df.iloc[t] = (self.w_opt.iloc[t] + 2*smoothed_df.iloc[t - 1]) / 3

        self.w_opt = smoothed_df


    def get_returns(self):

        self.returns = Data.get_test_data_backtest(self.portfolio.data.returns, self.index[self.cutoff])
        self.returns *= self.w_opt
        self.returns = self.returns.sum(axis=1)


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

        output_path = Opti.graph_dir_path + f"{self.portfolio.currency}/{self.portfolio.name}/Backtest/backtest.png"
        plt.savefig(output_path, format="png", bbox_inches='tight')

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        encoded = base64.b64encode(buf.read()).decode('utf-8')
        img_src = f"data:image/png;base64,{encoded}"

        return html.Img(src=img_src, style={"maxWidth": "100%", "height": "auto"})


    def plot_weights(self):

        fig, ax = plt.subplots()
        tickers = self.opti.optimum.keys()
        colors = [self.portfolio.color_map[ticker] for ticker in tickers]
        ax.stackplot(self.w_opt.index, 100*self.w_opt[tickers].T, labels=tickers, colors=colors)

        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.set_title(f'Weights history')

        ax.set_ylabel('%')
        ax.legend()
        ax.grid()

        output_path = Opti.graph_dir_path + f"{self.portfolio.currency}/{self.portfolio.name}/Backtest/weights.png"
        plt.savefig(output_path, format="png", bbox_inches='tight')

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        encoded = base64.b64encode(buf.read()).decode('utf-8')
        img_src = f"data:image/png;base64,{encoded}"

        return html.Img(src=img_src, style={"maxWidth": "100%", "height": "auto"})
