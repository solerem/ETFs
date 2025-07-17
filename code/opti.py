import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from portfolio import Portfolio
from data import Data
from scipy.optimize import minimize
import io
import base64
from dash import html
import pandas as pd


class Opti:

    solver_method = 'SLSQP'

    def __init__(self, portfolio):

        self.optimum, self.optimum_all, self.w_opt, self.goal, self.difference, self.constraints, self.bounds, self.color_map = None, None, None, None, None, None, None, None
        self.portfolio = portfolio
        self.get_bounds()
        self.get_constraints()
        self.w0 = np.full(self.portfolio.n, 1/self.portfolio.n)
        self.optimize()
        self.get_color_map()
        self.rebalance()


    def get_bounds(self):
        self.bounds = ([(-1, 1)] if self.portfolio.allow_short else [(0, 1)]) * self.portfolio.n


    @staticmethod
    def abs_sum(lst):

        return sum([abs(x) for x in lst])


    def get_constraints(self):
        func = Opti.abs_sum if self.portfolio.allow_short else sum
        self.constraints = [{'type': 'eq', 'fun': lambda w: func(w) - 1, 'tol': 1e-3}]


    def optimize(self):
        opt = minimize(lambda w: self.portfolio.objective(w=w), self.w0, method=Opti.solver_method, bounds=self.bounds, constraints=self.constraints, options={'ftol': 1e-6, 'maxiter': 1000})

        if not opt.success:
            print(f"Optimization failed: {opt.message}")
            return None

        self.w_opt = opt.x
        self.w_opt = np.array([0. if abs(w) < .01 else float(w) for w in self.w_opt])
        self.w_opt /= Opti.abs_sum(self.w_opt)

        self.optimum_all = {tick: w for tick, w in zip(self.portfolio.etf_list, self.w_opt)}
        self.optimum = {ticker: self.optimum_all[ticker] for ticker in self.optimum_all if self.optimum_all[ticker] != 0}
        self.goal = {ticker: self.optimum_all[ticker] * self.portfolio.liquidity for ticker in self.optimum_all}


    def get_color_map(self):
        assets = list(self.optimum.keys())
        cmap = cm.get_cmap('tab20', len(assets))  # or any colormap you prefer
        self.color_map = {asset: mcolors.to_hex(cmap(i)) for i, asset in enumerate(assets)}


    def plot_optimum(self):
        sorted_optimum = dict(sorted(self.optimum.items(), key=lambda item: item[1], reverse=True))

        fig, ax = plt.subplots()
        colors = [self.color_map[k] for k in sorted_optimum.keys()]
        ax.pie(
            sorted_optimum.values(),
            labels=sorted_optimum.keys(),
            colors=colors,
            autopct=lambda pct: f'{int(round(pct))}%'
        )
        ax.set_title('Optimal Allocation')

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        encoded = base64.b64encode(buf.read()).decode('utf-8')
        img_src = f"data:image/png;base64,{encoded}"

        return html.Img(src=img_src, style={"maxWidth": "100%", "height": "auto"})


    def plot_in_sample(self):
        returns = self.portfolio.data.returns[self.optimum.keys()]
        weights = list(self.optimum.values())
        cumulative = (1 + returns @ weights).cumprod()

        fig, ax = plt.subplots()
        ax.plot((cumulative-1)*100, label=self.portfolio.name + f' ({self.portfolio.currency})')

        spy = (self.portfolio.data.spy / self.portfolio.data.spy.iloc[0] - 1) * 100
        ax.plot(spy, label=f'Total stock market ({self.portfolio.currency})', linestyle='--')

        rf_rate = ((self.portfolio.data.rf_rate + 1).cumprod() - 1) * 100
        ax.plot(rf_rate, label='Rate', linestyle='--')

        ax.axhline(0, color='black')

        nb_years = int(Data.period[:-1])
        pa_perf = round(((cumulative[-1]) ** (1/nb_years) - 1)*100)

        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = round(drawdown.min()*100)

        ax.set_title(f'In-Sample ({pa_perf}% p.a., {max_drawdown}% max drawdown)')
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


    def plot_weighted_perf(self):
        returns = self.portfolio.data.returns[self.optimum.keys()]
        weights = pd.Series(self.optimum)

        cumulative_returns = (1 + returns).cumprod()
        weighted_cumulative = cumulative_returns.multiply(weights, axis=1)
        contribution = weighted_cumulative.subtract(1 * weights, axis=1) * 100

        fig, ax = plt.subplots()
        color_map = self.color_map
        for col in contribution.columns:
            ax.plot(contribution.index, contribution[col], label=col, color=color_map[col])

        ax.legend()
        ax.set_title('In-Sample Performance Attribution')
        ax.axhline(0, color='black')
        ax.set_ylabel('%')
        ax.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        encoded = base64.b64encode(buf.read()).decode('utf-8')
        img_src = f"data:image/png;base64,{encoded}"

        return html.Img(src=img_src, style={"maxWidth": "100%", "height": "auto"})


    def rebalance(self):
        self.difference = self.goal.copy()

        for ticker in self.portfolio.holdings:
            self.difference[ticker] -= self.portfolio.holdings[ticker]

        for ticker in self.difference:
            self.difference[ticker] = round(self.difference[ticker])

        self.difference = {ticker: int(self.difference[ticker]) for ticker in self.difference if self.difference[ticker]}







