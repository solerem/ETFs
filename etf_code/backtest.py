import pandas as pd
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
        it = range(self.cutoff, self.n)
        if self.progress_callback is None:
            it = tqdm(it)
        for idx, i in enumerate(it):
            if self.progress_callback is not None:
                self.progress_callback(idx + 1, total)
            # Slice pre-loaded data to this date (no disk read)
            sliced_data = Data(
                self.portfolio.currency,
                Info.etf_list,
                static=True,
                backtest=self.index[i],
                rates=self.portfolio.rates,
                _full_data=full_data,
            )
            portfolio = Portfolio(
                risk=self.portfolio.risk,
                currency=self.portfolio.currency,
                static=True,
                backtest=self.index[i],
                rates=self.portfolio.rates,
                data=sliced_data,
            )
            optimum = Opti(portfolio, long_only=self.opti.long_only, max_assets=self.opti.max_assets).optimum_all
            self.w_opt.loc[self.index[i]] = optimum

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
