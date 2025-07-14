import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
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

        self.optimum, self.optimum_all, self.w_opt, self.goal, self.difference, self.constraints = None, None, None, None, None, None

        self.portfolio = portfolio
        self.get_constraints()
        self.w0 = np.full(self.portfolio.n, 1/self.portfolio.n)
        self.bounds = ([(-1, 1)] if self.portfolio.allow_short else [(0, 1)]) * self.portfolio.n
        self.optimize()
        self.rebalance()


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


    def plot_optimum(self):
        # Create the plot
        fig, ax = plt.subplots()
        ax.bar(self.optimum.keys(), self.optimum.values(), color=self.portfolio.color_plot)
        ax.set_title('Optimal Allocation')

        # Save it to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        # Encode to base64
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        img_src = f"data:image/png;base64,{encoded}"

        return html.Img(src=img_src, style={"maxWidth": "100%", "height": "auto"})


    def plot_in_sample(self):
        returns = self.portfolio.data.returns[self.optimum.keys()]
        weights = list(self.optimum.values())
        cumulative = ((1 + returns @ weights).cumprod() - 1) * 100

        fig, ax = plt.subplots()
        ax.plot(cumulative, label=self.portfolio.name)

        spy = (self.portfolio.data.spy / self.portfolio.data.spy.iloc[0] - 1) * 100
        ax.plot(spy, label='S&P 500', linestyle='--')

        rf_rate = ((self.portfolio.data.rf_rate + 1).cumprod() - 1) * 100
        ax.plot(rf_rate, label='Rate', linestyle='--')

        ax.axhline(0, color='black')

        nb_years = int(Data.period[:-1])
        pa_perf = round(((1+cumulative[-1]/100) ** (1/nb_years) - 1)*100, 1)
        ax.set_title(f'In-Sample Performance ({pa_perf}% p.a.)')
        ax.set_ylabel('%')
        ax.legend()
        ax.grid()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        # Encode to base64
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        img_src = f"data:image/png;base64,{encoded}"

        return html.Img(src=img_src, style={"maxWidth": "100%", "height": "auto"})

    def plot_weighted_perf(self):
        returns = self.portfolio.data.returns[self.optimum.keys()]
        weights = pd.Series(self.optimum)  # Ensure index matches ETF names

        # Compute cumulative returns for each asset
        cumulative_returns = (1 + returns).cumprod()

        # Multiply by initial weights to get dollar-equivalent growth per component
        weighted_cumulative = cumulative_returns.multiply(weights, axis=1)

        # Total portfolio cumulative return (same as in plot_in_sample)
        portfolio_cumulative = weighted_cumulative.sum(axis=1)

        # Contribution of each ETF to total portfolio return
        contribution = weighted_cumulative.subtract(1 * weights, axis=1) * 100  # Subtract starting value

        # Plot
        fig, ax = plt.subplots()
        for col in contribution.columns:
            ax.plot(contribution.index, contribution[col], label=col)

        ax.legend()
        ax.set_title('In-Sample Performance Attribution')
        ax.axhline(0, color='black')
        ax.set_ylabel('%')
        ax.grid()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        # Encode to base64
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







