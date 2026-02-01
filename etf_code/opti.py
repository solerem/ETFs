import math
import time

import numpy as np
import matplotlib
from dash import dash_table, dcc
from scipy.optimize import curve_fit
from portfolio import Portfolio
import matplotlib.cm as cm
import matplotlib.colors as mcolors

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from data import Data
from scipy.optimize import minimize
import io
import base64
from dash import html
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import cvxpy as cp
import plotly.graph_objects as go

from charts import dash_graph, figure_drawdown, figure_performance_vs_benchmarks
from config import (
    NB_PERIODS_PER_YEAR,
    MIN_WEIGHT_DISPLAY,
    BORROW_RATE_ANNUAL_FACTOR,
    CONSTRAINT_TOL,
    DEFAULT_MAX_ASSETS,
    VAR_CONFIDENCE,
)


class Opti:
    solver_method = 'SLSQP'

    def __init__(self, portfolio, long_only=False, max_assets=DEFAULT_MAX_ASSETS, solver_n_threads=None):
        self.optimum, self.optimum_all, self.w_opt, self.constraints, self.bounds, self.cumulative, self.returns, self.color_map = None, None, None, None, None, None, None, None
        self.portfolio = portfolio
        self.long_only = long_only
        self.max_assets = max_assets
        self.solver_n_threads = solver_n_threads
        self.get_bounds()
        self.get_constraints()
        self.w0 = np.full(self.portfolio.n, 1 / self.portfolio.n)
        w_override = getattr(portfolio, "w_opt_override", None)
        # When long_only is True, never use pre-computed weights (they may contain shorts).
        # When max_assets < n, cardinality is constrained so we must run the optimizer.
        use_override = (
            w_override is not None
            and len(w_override) == portfolio.n
            and not self.long_only
            and self.max_assets >= portfolio.n
        )
        if use_override:
            self.w_opt = np.asarray(w_override, dtype=float)
            self.optimum_all = {
                tick: w for tick, w in zip(self.portfolio.etf_list, self.w_opt)
            }
            self.optimum = {
                ticker: weight
                for ticker, weight in self.optimum_all.items()
                if abs(weight) >= MIN_WEIGHT_DISPLAY
            }
        else:
            self.optimize(max_assets=self.max_assets)
        self.get_cumulative()
        self.get_color_map()

    def get_color_map(self):
        cmap = cm.get_cmap('tab20', len(self.optimum))
        self.color_map = {asset: mcolors.to_hex(cmap(i)) for i, asset in enumerate(self.optimum.keys())}

    def get_bounds(self):
        if self.long_only:
            self.bounds = [(0, 1)] * self.portfolio.n
        else:
            self.bounds = [(-1, 1)] * self.portfolio.n

    @staticmethod
    def abs_sum(lst):
        return sum([abs(x) for x in lst])


    def get_constraints(self):
        func = sum
        func = Opti.abs_sum

        self.constraints = [{'type': 'eq', 'fun': lambda w: func(w) - 1, 'tol': CONSTRAINT_TOL}]

    def optimize(self, max_assets=DEFAULT_MAX_ASSETS, borrow_years=1 / Data.NB_PERIOD):
        borrow_rate_annual = {tick: 0.0 for tick in self.portfolio.etf_list}
        # https://www.interactivebrokers.com/en/pricing/reference-benchmark-rates-int.php
        df_repo = pd.read_csv(Data.static_dir_path / 'repo.csv', sep=';')
        #df_repo['REPO'] = df_repo['REPO'].str.replace(',', '.').astype(float)
        for _, row in df_repo.iterrows():
            borrow_rate_annual[row['TICKER']] = row['REPO'] / 100

        # Build data for a mean-variance objective from historical returns
        rets = self.portfolio.data.returns[self.portfolio.etf_list]
        rets[self.portfolio.currency] += (BORROW_RATE_ANNUAL_FACTOR ** (1 / NB_PERIODS_PER_YEAR)) - 1
        mu = rets.mean().values  # expected returns (vector)
        Sigma = rets.cov().values  # covariance matrix (n x n)
        n = self.portfolio.n

        # Borrow cost vector (same order as etf_list)
        borrow_vec = np.array(
            [borrow_rate_annual[t] for t in self.portfolio.etf_list],
            dtype=float,
        )

        # Ensure Sigma is PSD for cvxpy quad_form (numerical stability)
        Sigma = Sigma + 1e-8 * np.eye(n)

        # CVXPY variables: long/short decomposition and cardinality
        M = 1.0
        w_plus = cp.Variable(n, nonneg=True)
        w_minus = cp.Variable(n, nonneg=True)
        z = cp.Variable(n, boolean=True)

        # Constraints
        constraints = [
            w_plus <= M * z,
            w_minus <= M * z,
            cp.sum(w_plus + w_minus) == 1.0,
            cp.sum(z) <= max_assets,
        ]
        if self.long_only:
            constraints.append(w_minus == 0)

        w = w_plus - w_minus
        var_term = cp.quad_form(w, Sigma)
        ret_term = mu @ w
        borrow_cost = borrow_vec @ w_minus * borrow_years

        objective = cp.Minimize(
            self.portfolio.weight_cov * var_term - ret_term + borrow_cost
        )
        prob = cp.Problem(objective, constraints)
        solver_opts = {}
        if self.solver_n_threads is not None:
            solver_opts["parallel/maxnthreads"] = self.solver_n_threads
        prob.solve(solver=cp.SCIP, verbose=False, **solver_opts)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(
                f"CVXPY/SCIP optimization failed with status {prob.status}"
            )

        w_opt = (w_plus.value - w_minus.value).flatten()
        w_opt = np.asarray(w_opt, dtype=float)
        w_opt /= Opti.abs_sum(w_opt)

        self.w_opt = w_opt
        self.optimum_all = {
            tick: w for tick, w in zip(self.portfolio.etf_list, self.w_opt)
        }
        self.optimum = {
            ticker: weight
            for ticker, weight in self.optimum_all.items()
            if abs(weight) >= MIN_WEIGHT_DISPLAY
        }

    def get_cumulative(self):
        self.returns = self.portfolio.data.returns[self.optimum.keys()]
        weights = list(self.optimum.values())
        self.cumulative = (1 + self.returns @ weights).cumprod()

    def plot_optimum(self):
        sorted_optimum = dict(sorted(self.optimum.items(), key=lambda item: item[1], reverse=True))
        values = [abs(sorted_optimum[x]) for x in sorted_optimum]
        display_ticker = lambda t: Data.ticker_display_name(t)
        labels = [display_ticker(x) if sorted_optimum[x] >= 0 else 'short ' + display_ticker(x) for x in sorted_optimum]
        colors = [self.color_map[k] for k in sorted_optimum.keys()]
        full_name_list = [self.portfolio.data.etf_full_names.loc[ticker] for ticker in sorted_optimum]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            customdata=full_name_list
        )])
        fig.update_traces(
            textinfo='label+percent',
            texttemplate='%{label}: %{percent:.1%}',
            hovertemplate='%{label}: %{percent:.1%}<br>%{customdata}<extra></extra>'
        )
        fig.update_layout(title='Optimal Allocation')
        return dash_graph(fig)

    def plot_in_sample(self):
        cumulative_pct = (self.cumulative - 1) * 100
        spy = (self.portfolio.data.benchmarks['SPY'] / self.portfolio.data.benchmarks['SPY'].iloc[0] - 1) * 100
        bonds = (self.portfolio.data.benchmarks['AGG'] / self.portfolio.data.benchmarks['AGG'].iloc[0] - 1) * 100
        gold = (self.portfolio.data.benchmarks['GLD'] / self.portfolio.data.benchmarks['GLD'].iloc[0] - 1) * 100
        fig = figure_performance_vs_benchmarks(
            cumulative_pct, spy, bonds, gold,
            title='In-sample Performance vs Benchmark',
        )
        return dash_graph(fig)

    def plot_weighted_perf(self):
        returns = self.portfolio.data.returns[self.optimum.keys()]
        weights = pd.Series(self.optimum)

        cumulative_returns = (1 + returns).cumprod() - 1
        contribution = cumulative_returns.multiply(weights, axis=1) * 100
        fig = go.Figure()
        for col in contribution.columns:
            ticker_label = Data.ticker_display_name(col)
            display_name = ticker_label if self.optimum[col] >= 0 else 'short ' + ticker_label
            fig.add_trace(go.Scatter(
                x=contribution.index,
                y=contribution[col].values,
                mode='lines',
                name=display_name,
                line=dict(color=self.color_map[col]),
                hovertemplate='%{y:.1f}%<extra>%{fullData.name}</extra>'
            ))

        fig.add_hline(y=0, line_color='black')
        fig.update_layout(
            title='In-sample Performance Breakdown',
            yaxis_title='%',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5)
        )
        fig.update_yaxes(tickformat='.1f')
        return dash_graph(fig)

    def plot_drawdown(self):
        fig = figure_drawdown(self.cumulative, title='Drawdown')
        return dash_graph(fig)

    def plot_info(self):
        info = {}
        explain = {}
        weights = list(self.optimum.values())
        # self.returns = self.returns[self.optimum.keys()]
        returns = self.returns @ weights

        nb_years = int(self.portfolio.data.period[:-1])
        pa_perf = (round(((self.cumulative.iloc[-1]) ** (1 / nb_years) - 1) * 100, 1))
        info['CAGR'] = str(round(pa_perf, 1)) + ' %'
        explain['CAGR'] = 'Average annual growth rate'

        sharpe = returns.mean() / returns.std()
        info['Sharpe ratio'] = round(sharpe * math.sqrt(Data.NB_PERIOD), 2)
        explain['Sharpe ratio'] = 'Risk-adjusted return'

        running_max = self.cumulative.cummax()
        drawdown = (self.cumulative - running_max) / running_max
        info['Max drawdown'] = str(round(drawdown.min() * 100, 1)) + ' %'
        info['Avg drawdown'] = str(round(drawdown.mean() * 100, 1)) + ' %'
        explain['Max drawdown'] = 'Largest peak-to-trough loss'
        explain['Avg drawdown'] = 'Typical loss during downturns'

        spy = self.portfolio.data.benchmarks['SPY'].pct_change().dropna()
        beta = returns[1:].cov(spy) / spy.var()
        info['Beta (Stocks)'] = round(beta, 2)
        explain['Beta (Stocks)'] = 'Sensitivity to stock market movements'

        vol = returns.std() * math.sqrt(Data.NB_PERIOD)
        info['Volatility'] = round(vol, 2)
        explain['Volatility'] = 'Return fluctuations (risk)'

        var95 = np.percentile(returns, (1 - VAR_CONFIDENCE) * 100)
        info['VaR 95%'] = str(round(var95 * 100, 1)) + ' %'
        explain['VaR 95%'] = 'Max expected loss at 95% confidence'

        try:
            X = sm.add_constant(spy)
            model = sm.OLS(returns[1:], X).fit()
            r2 = model.rsquared
            info['R2 (Stocks)'] = str(round(100 * r2)) + ' %'
        except:
            info['R2 (Stocks)'] = '0'
        explain['R2 (Stocks)'] = '% of returns explained by stock benchmark'

        # Convert dict into list of dicts for DataTable
        data = [{"Metric": k, "Value": info[k], 'Detail': explain[k]} for k in info]

        return dash_table.DataTable(
            data=data,
            columns=[
                {"name": "Metric", "id": "Metric"},
                {"name": "Value", "id": "Value"},
                {"name": "Detail", "id": "Detail"}
            ],
            page_size=10,
            sort_action='native',
            style_table={'overflowX': 'auto'},
            style_as_list_view=True,
            style_header={
                'fontWeight': '600',
                'border': 'none',
                'textAlign': 'center'
            },
            style_cell={
                'padding': '14px',
                'border': 'none',
                'textAlign': 'center',
                'fontSize': '16px'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgba(0,0,0,0.02)'
                }
            ]
        )
