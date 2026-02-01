import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from portfolio import Portfolio, Info
from opti import Opti
from data import Data
from tqdm import tqdm
from dash import dash_table, dcc
import statsmodels.api as sm
import numpy as np
import plotly.graph_objects as go

from charts import dash_graph, figure_drawdown, figure_performance_vs_benchmarks
from config import TRAIN_TEST_RATIO, WEIGHT_PLOT_COVERAGE, VAR_CONFIDENCE


def _backtest_worker(args):
    """Run one backtest date in a separate process. Returns (date, optimum_all)."""
    (i, full_data, index, risk, currency, rates, long_only, max_assets) = args
    sliced_data = Data(
        currency,
        Info.etf_list,
        static=True,
        backtest=index[i],
        rates=rates,
        _full_data=full_data,
    )
    portfolio = Portfolio(
        risk=risk,
        currency=currency,
        static=True,
        backtest=index[i],
        rates=rates,
        data=sliced_data,
    )
    opti = Opti(
        portfolio,
        long_only=long_only,
        max_assets=max_assets,
        solver_n_threads=1,
    )
    return (index[i], opti.optimum_all)


class Backtest:
    def __init__(self, opti, progress_callback=None):
        self.opti = opti
        self.portfolio = self.opti.portfolio
        self.progress_callback = progress_callback
        # Adaptive train/test split (matches implementation)
        self.ratio_train_test = TRAIN_TEST_RATIO
        self.to_consider = self.opti.optimum.keys()
        self.w_opt, self.returns, self.n, self.cutoff, self.index, self.returns_decomp = None, None, None, None, None, None
        self.cumulative = None
        self.parse_data()
        self.get_returns()

    def parse_data(self):
        # Load all CSV/parquet once (no backtest slice)
        full_data = Data(
            self.portfolio.currency,
            Info.etf_list,
            static=True,
            backtest=None,
            rates=self.portfolio.rates,
        )
        # Precompute expanding mean/var of excess returns for each date (used in objective(single_ticker=...))
        full_data.expanding_mean_er = full_data.excess_returns.expanding().mean()
        full_data.expanding_var_er = full_data.excess_returns.expanding().var().fillna(0.0)
        self.n = len(full_data.nav)
        self.cutoff = int(self.ratio_train_test * self.n)
        self.index = list(full_data.nav.index)

        self.w_opt = pd.DataFrame({ticker: [] for ticker in self.opti.portfolio.etf_list})
        total = self.n - self.cutoff
        indices = list(range(self.cutoff, self.n))

        # Parallelize backtest: each date is independent
        n_workers = min(max(1, (os.cpu_count() or 4) - 1), 8, total)
        task_args = [
            (
                i,
                full_data,
                self.index,
                self.portfolio.risk,
                self.portfolio.currency,
                self.portfolio.rates,
                self.opti.long_only,
                self.opti.max_assets,
            )
            for i in indices
        ]

        executor = ProcessPoolExecutor(max_workers=n_workers)
        futures = {executor.submit(_backtest_worker, a): a for a in task_args}
        completed = 0
        if self.progress_callback is None:
            iterator = tqdm(as_completed(futures), total=total, desc="Backtest")
        else:
            iterator = as_completed(futures)

        for future in iterator:
            date, optimum_all = future.result()
            self.w_opt.loc[date] = optimum_all
            if self.progress_callback is not None:
                completed += 1
                self.progress_callback(completed, total)

        executor.shutdown(wait=True)

    def get_returns(self):
        self.returns_decomp = Data.get_test_data_backtest(self.portfolio.data.returns, self.index[self.cutoff])
        self.returns_decomp *= self.w_opt
        self.returns = self.returns_decomp.sum(axis=1)

    def plot_backtest(self):
        cumulative = (1 + self.returns).cumprod()
        cumulative_pct = (cumulative - 1) * 100
        spy = self.portfolio.data.benchmarks['SPY'].copy().loc[self.index[self.cutoff]:]
        spy = (spy / spy.iloc[0] - 1) * 100
        bonds = self.portfolio.data.benchmarks['AGG'].copy().loc[self.index[self.cutoff]:]
        bonds = (bonds / bonds.iloc[0] - 1) * 100
        gold = self.portfolio.data.benchmarks['GLD'].copy().loc[self.index[self.cutoff]:]
        gold = (gold / gold.iloc[0] - 1) * 100
        fig = figure_performance_vs_benchmarks(
            cumulative_pct, spy, bonds, gold,
            title='Backtest Performance vs Benchmark',
        )
        return dash_graph(fig)

    def plot_weights(self):
        import math
        from pathlib import Path

        # --- Prep & safety ---
        w = self.w_opt.copy()
        if w.empty:
            raise ValueError("w_opt is empty; nothing to plot.")

        # Ensure numeric & fill NaNs so means/cum sums behave
        w = w.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Only keep columns that actually exist
        to_consider = [t for t in getattr(self, "to_consider", []) if t in w.columns]

        mean_w = w.mean(axis=0)  # average per ticker over time
        # If everything is zero, avoid division by zero downstream
        total_weight = float(mean_w.sum())
        if math.isclose(total_weight, 0.0, abs_tol=1e-12):
            raise ValueError("All mean weights are zero; cannot compute 99% coverage set.")

        # --- Build the included set greedily ---
        included = set(to_consider)
        # Remaining candidates sorted by mean weight (desc)
        remaining = [c for c in w.columns if c not in included]
        remaining.sort(key=lambda x: float(mean_w.get(x, 0.0)), reverse=True)

        # Start with current ones; compute included coverage
        included_weight = float(mean_w[list(included)].sum()) if included else 0.0

        # Greedily add until reaching ≥ WEIGHT_PLOT_COVERAGE cumulative mean weight
        target = WEIGHT_PLOT_COVERAGE * total_weight
        while included_weight < target and remaining:
            nxt = remaining.pop(0)
            included.add(nxt)
            included_weight += float(mean_w[nxt])

        # Final plotting order: sort included by descending mean weight for readability
        tickers_to_plot = sorted(included, key=lambda x: float(mean_w.get(x, 0.0)), reverse=True)

        data = w[tickers_to_plot].abs().astype(float)
        row_sums = data.sum(axis=1).replace(0, np.nan)
        data = data.div(row_sums, axis=0).fillna(0.0) * 100.0
        fig = go.Figure()
        color_map = self.opti.color_map
        for col in tickers_to_plot:
            # Check if this is a short position (negative mean weight or in optimum as negative)
            is_short = mean_w.get(col, 0.0) < 0 or self.opti.optimum.get(col, 0) < 0
            ticker_label = Data.ticker_display_name(col)
            display_name = 'short ' + ticker_label if is_short else ticker_label
            series_color = color_map.get(col)
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col].values,
                mode='lines',
                name=display_name,
                stackgroup='one',
                stackgaps='interpolate',
                fill='tonexty',
                fillcolor=series_color,
                line=dict(width=0.5, color=series_color),
                opacity=1.0,
                hovertemplate='%{y:.1f}%<extra>%{fullData.name}</extra>'
            ))

        fig.update_layout(
            title='Weights history',
            yaxis_title='%',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5)
        )
        fig.update_yaxes(range=[0, 100], tickformat='.1f')
        return dash_graph(fig)

    def plot_perf_attrib(self):
        returns = self.returns_decomp[self.to_consider]
        fig = go.Figure()
        for col in self.to_consider:
            ticker_label = Data.ticker_display_name(col)
            display_name = ticker_label if self.opti.optimum[col] >= 0 else 'short ' + ticker_label
            fig.add_trace(go.Scatter(
                x=returns.index,
                y=(returns[col].cumsum() * 100).values,
                mode='lines',
                name=display_name,
                line=dict(color=self.opti.color_map[col]),
                hovertemplate='%{y:.1f}%<extra>%{fullData.name}</extra>'
            ))

        fig.add_hline(y=0, line_color='black')
        fig.update_layout(
            title='Backtest Performance Breakdown',
            yaxis_title='%',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5)
        )
        fig.update_yaxes(tickformat='.1f')
        return dash_graph(fig)

    def plot_drawdown(self):
        cumulative = (1 + self.returns).cumprod()
        fig = figure_drawdown(cumulative, title='Backtest Drawdown')
        return dash_graph(fig)

    def plot_info(self):
        info = {}
        explain = {}
        self.cumulative = (1 + self.returns).cumprod()

        # Calculate actual backtest period in years
        start_date = self.returns.index[0]
        end_date = self.returns.index[-1]
        nb_years = (end_date - start_date).days / 365.25
        if nb_years <= 0:
            nb_years = len(self.returns) / Data.NB_PERIOD  # Fallback if days calculation fails
        pa_perf = (round(((self.cumulative.iloc[-1]) ** (1 / nb_years) - 1) * 100, 1))
        info['CAGR'] = str(round(pa_perf, 1)) + ' %'
        explain['CAGR'] = 'Average annual growth rate'

        sharpe = self.returns.mean() / self.returns.std()
        info['Sharpe ratio'] = round(sharpe * np.sqrt(Data.NB_PERIOD), 2)
        explain['Sharpe ratio'] = 'Risk-adjusted return'

        running_max = self.cumulative.cummax()
        drawdown = (self.cumulative - running_max) / running_max
        info['Max drawdown'] = str(round(drawdown.min() * 100, 1)) + ' %'
        info['Avg drawdown'] = str(round(drawdown.mean() * 100, 1)) + ' %'
        explain['Max drawdown'] = 'Largest peak-to-trough loss'
        explain['Avg drawdown'] = 'Typical loss during downturns'

        spy = self.portfolio.data.benchmarks['SPY'].pct_change().dropna()[self.cutoff:]
        beta = self.returns.cov(spy) / spy.var()
        info['Beta (Stocks)'] = round(beta, 2)
        explain['Beta (Stocks)'] = 'Sensitivity to stock market movements'

        vol = self.returns.std() * np.sqrt(Data.NB_PERIOD)
        info['Volatility'] = round(vol, 2)
        explain['Volatility'] = 'Return fluctuations (risk)'

        var95 = np.percentile(self.returns, (1 - VAR_CONFIDENCE) * 100)
        info['VaR 95%'] = str(round(var95 * 100, 1)) + ' %'
        explain['VaR 95%'] = 'Max expected loss at 95% confidence'

        X = sm.add_constant(spy)
        model = sm.OLS(self.returns, X).fit()
        r2 = model.rsquared
        info['R2 (Stocks)'] = str(round(100 * r2)) + ' %'
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
