"""
Rolling backtest utilities for an optimized portfolio.

This module defines :class:`Backtest`, which:

* Walks forward through monthly dates and re-optimizes the portfolio at each step
  using in-sample data up to that date (via :class:`~portfolio.Portfolio` and
  :class:`~opti.Opti` in *static* mode with an in-sample cutoff).
* Builds a time series of optimal weights (:attr:`w_opt`) across the test window.
* Computes out-of-sample (test) returns from the held-out period.
* Produces diagnostic plots (equity curve, stacked weights, performance
  attribution, and drawdown) as Dash-ready images while saving PNGs to disk.

Notes
-----
* The train/test split is controlled by :attr:`Backtest.ratio_train_test`. It is
  set adaptively from the underlying data horizon:
  ``0.8`` when using the short crypto horizon (``period == '5y'``), otherwise
  ``0.9`` for the long horizon (``'20y'``).
* Re-optimization loops monthly from the cutoff to the end of the data,
  constructing a fresh :class:`~portfolio.Portfolio` with ``static=True`` and
  ``backtest=<current_date>`` at each step.
* Figures are saved under ``graphs/<currency>/`` using
  :meth:`opti.Opti.save_fig_as_dash_img`.
"""

import pandas as pd
from portfolio import Portfolio
from opti import Opti
from tqdm import tqdm
import matplotlib
from dash import dash_table
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from data import Data
import statsmodels.api as sm
import numpy as np

class Backtest:
    """
    Rolling re-optimization backtest.

    Class Attributes
    ----------------
    ratio_train_test : float
        Fraction of the sample used for training/in-sample. Set to ``0.8`` when
        the underlying :class:`data.Data` uses the short horizon (``'5y'``),
        otherwise ``0.9`` for the long horizon (``'20y'``).

    Parameters
    ----------
    opti : Opti
        A fully-initialized optimizer instance whose portfolio defines the
        configuration (universe, risk, currency, shorting, etc.) and whose
        optimized constituents (:attr:`Opti.optimum`) seed the initial
        ``to_consider`` set for plots.

    Attributes
    ----------
    opti : Opti
        The reference optimizer object passed in.
    portfolio : Portfolio
        Convenience alias to ``opti.portfolio``.
    to_consider : dict_keys[str]
        The keys of ``opti.optimum``; used to focus attribution plots.
    w_opt : pandas.DataFrame | None
        Time-indexed weights per ticker for the walk-forward re-optimizations.
        Columns cover the working universe at initialization time.
    returns : pandas.Series | None
        Out-of-sample (test) portfolio returns (simple monthly).
    n : int | None
        Number of rows (time points) in the underlying NAV table.
    cutoff : int | None
        Index position separating train/test based on :attr:`ratio_train_test`.
    index : list[pandas.Timestamp] | None
        Copy of the underlying ``DatetimeIndex`` for iteration.
    returns_decomp : pandas.DataFrame | None
        Per-asset contributions to test-period returns (weights × returns).
    cumulative : pandas.Series | None
        Test-period cumulative equity curve (set in :meth:`plot_info`).
    """

    def __init__(self, opti):
        """
        Initialize the backtest, compute rolling weights, and derive returns.

        The constructor:
        1) Parses data and builds a time series of optimal weights by repeatedly
           re-optimizing up to each test date in *static* mode.
        2) Computes test-period returns via :meth:`get_returns`.

        Parameters
        ----------
        opti : Opti
            Optimizer instance providing the baseline portfolio.

        Returns
        -------
        None
        """
        self.opti = opti
        self.portfolio = self.opti.portfolio
        # Adaptive train/test split (matches implementation)
        self.ratio_train_test = .8 if self.portfolio.data.period == '5y' else .9
        self.to_consider = self.opti.optimum.keys()
        self.w_opt, self.returns, self.n, self.cutoff, self.index, self.returns_decomp = None, None, None, None, None, None
        self.cumulative = None
        self.parse_data()
        self.get_returns()

    def parse_data(self):
        """
        Build rolling optimal weights by re-optimizing through time.

        Steps
        -----
        1. Determine the train/test split using :attr:`ratio_train_test`.
        2. For each test date ``t`` (from cutoff to end):
           - Create a new :class:`~portfolio.Portfolio` with ``static=True`` and
             ``backtest=index[t]`` so that all data are truncated to in-sample
             up to that date.
           - Run :class:`~opti.Opti` on that portfolio and store the optimal
             weight vector into :attr:`w_opt` at timestamp ``index[t]``.

        Side Effects
        ------------
        Sets :attr:`n`, :attr:`cutoff`, :attr:`index`, and fills :attr:`w_opt`.

        Returns
        -------
        None
        """
        self.n = len(self.portfolio.data.nav)
        self.cutoff = int(self.ratio_train_test * self.n)
        self.index = list(self.portfolio.data.nav.index)

        self.w_opt = pd.DataFrame({ticker: [] for ticker in self.opti.portfolio.etf_list})
        for i in tqdm(range(self.cutoff, self.n)):
            portfolio = Portfolio(
                risk=self.portfolio.risk,
                currency=self.portfolio.currency,
                allow_short=self.portfolio.allow_short,
                static=True,
                backtest=self.index[i],
                rates=self.portfolio.rates,
                crypto=self.opti.portfolio.crypto
            )
            optimum = Opti(portfolio).optimum_all
            self.w_opt.loc[self.index[i]] = optimum

    def get_returns(self):
        """
        Compute out-of-sample (test) returns and decomposition.

        * Select the test-period rows from the full return matrix using
          :meth:`data.Data.get_test_data_backtest` with the cutoff timestamp.
        * Multiply by time-aligned weights to obtain per-asset contributions.
        * Sum across columns to obtain the total test return series.

        Side Effects
        ------------
        Sets :attr:`returns_decomp` and :attr:`returns`.

        Returns
        -------
        None
        """
        self.returns_decomp = Data.get_test_data_backtest(self.portfolio.data.returns, self.index[self.cutoff])
        self.returns_decomp *= self.w_opt
        self.returns = self.returns_decomp.sum(axis=1)

    def plot_backtest(self):
        """
        Plot the backtest equity curve vs. benchmark and risk-free leg.

        The title reflects annualized performance (p.a.) and maximum drawdown
        over the test window. The benchmark label switches to ``BTC`` in crypto
        mode; otherwise it is the total stock market proxy.

        Returns
        -------
        dash.html.Img
            Dash image component with the figure embedded and saved to disk.
        """
        cumulative = (1 + self.returns).cumprod()

        fig, ax = plt.subplots()
        ax.plot((cumulative - 1) * 100, label=str(self.portfolio.name) + f' ({self.portfolio.currency})')

        spy = self.portfolio.data.spy.copy()
        spy = spy.loc[self.index[self.cutoff]:]
        spy = (spy / spy.iloc[0] - 1) * 100
        label = 'BTC' if self.portfolio.crypto else 'Total stock market'
        ax.plot(spy, label=f'{label} ({self.portfolio.currency})', linestyle='--')

        rf_rate = ((self.portfolio.data.rf_rate.loc[self.index[self.cutoff]:] + 1).cumprod() - 1) * 100
        ax.plot(rf_rate, label='Rate', linestyle='--')

        ax.axhline(0, color='black')

        nb_years = int(self.opti.portfolio.data.period[:-1]) * (1 - self.ratio_train_test)
        pa_perf = round(((cumulative.iloc[-1]) ** (1 / nb_years) - 1) * 100, 1)

        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = round(drawdown.min() * 100, 1)
        plt.setp(ax.get_xticklabels(), rotation=45)

        ax.set_title('Backtest')
        ax.set_ylabel('%')
        ax.legend()
        ax.grid()

        output_path = Opti.graph_dir_path / f'{self.portfolio.currency}/{self.portfolio.name}- Backtest_backtest.png'
        return Opti.save_fig_as_dash_img(fig, output_path)

    def plot_weights(self):
        """
        Plot a stacked weight history for the most material tickers.

        Heuristic
        ---------
        Start with the *current* optimized constituents (:attr:`to_consider`).
        Greedily add other tickers by descending average weight until the
        cumulative mean weight of the plotted set reaches at least **99%**
        (matches implementation target ``0.99``).

        Returns
        -------
        dash.html.Img
            Dash image component with the figure embedded and saved to disk.

        Raises
        ------
        ValueError
            If weight history is empty or all mean weights are zero.
        """
        import math
        from pathlib import Path
        import matplotlib.pyplot as plt

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

        # Greedily add until reaching ≥ 99% cumulative mean weight
        target = 0.99 * total_weight
        while included_weight < target and remaining:
            nxt = remaining.pop(0)
            included.add(nxt)
            included_weight += float(mean_w[nxt])

        # Final plotting order: sort included by descending mean weight for readability
        tickers_to_plot = sorted(included, key=lambda x: float(mean_w.get(x, 0.0)), reverse=True)

        # --- Colors (fallback to matplotlib cycle if any missing) ---
        color_map = self.opti.color_map
        colors = [color_map[t] for t in tickers_to_plot]

        # --- Build the figure ---
        fig, ax = plt.subplots()
        data = (100.0 * w[tickers_to_plot]).astype(float)

        # Matplotlib's stackplot works best with explicit 1D arrays
        ys = [data[col].values for col in tickers_to_plot]
        ax.stackplot(data.index, *ys, labels=tickers_to_plot, colors=colors)

        # Cosmetics
        ax.set_title("Weights history")
        ax.set_ylabel("%")
        ax.set_ylim(0, 100)
        ax.axhline(100, linewidth=1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Legend: compact & outside if many series
        n = len(tickers_to_plot)
        if n <= 12:
            ax.legend(loc="upper left", frameon=False)
        else:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)

        # --- Save & return Dash image ---
        output_path = Opti.graph_dir_path / f"{self.portfolio.currency}/{self.portfolio.name}- Backtest_weights.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        return Opti.save_fig_as_dash_img(fig, output_path)

    def plot_perf_attrib(self):
        """
        Plot cumulative performance attribution for selected tickers.

        Uses the per-asset return contributions in :attr:`returns_decomp` and
        accumulates them through time, plotted in percent. The set of tickers
        follows :attr:`to_consider`.

        Returns
        -------
        dash.html.Img
            Dash image component with the figure embedded and saved to disk.
        """
        returns = self.returns_decomp[self.to_consider]

        fig, ax = plt.subplots()
        for col in self.to_consider:
            ax.plot(returns.index, (returns[col].cumsum()) * 100, label=col, color=self.opti.color_map[col])

        ax.axhline(0, color='black')
        plt.setp(ax.get_xticklabels(), rotation=45)

        ax.set_title('Backtest Performance Attribution')
        ax.set_ylabel('%')
        ax.legend()
        ax.grid()

        output_path = Opti.graph_dir_path / f'{self.portfolio.currency}/{self.portfolio.name}- Backtest_perf_attrib.png'
        return Opti.save_fig_as_dash_img(fig, output_path)

    def plot_drawdown(self):
        """
        Plot the portfolio drawdown curve (area below zero) over the test window.

        Returns
        -------
        dash.html.Img
            Dash image component with the figure embedded and saved to disk.
        """
        cumulative = (1 + self.returns).cumprod()

        rolling_max = cumulative.cummax()
        drawdown = cumulative / rolling_max - 1

        fig, ax = plt.subplots()
        ax.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=.5)

        ax.set_title('Drawdown')
        ax.set_ylabel('%')
        ax.grid()

        output_path = Opti.graph_dir_path / f'{self.portfolio.currency}/{self.portfolio.name}- Backtest_drawdown.png'
        return Opti.save_fig_as_dash_img(fig, output_path)

    def plot_info(self):
        """
        Assemble a compact metrics table (Dash DataTable) for the backtest window.

        Metrics reported
        ----------------
        * CAGR — average annual growth rate (from the test-period cumulative).
        * Sharpe ratio — monthly mean/std scaled by ``sqrt(12)``.
        * Max/Avg drawdown — from the test-period equity curve.
        * Beta — covariance with benchmark (VTI or BTC-USD) over benchmark variance.
        * Volatility — monthly std scaled by ``sqrt(12)``.
        * VaR 95% — empirical 5th percentile of monthly returns.
        * R² — OLS fit of portfolio returns on the benchmark over the test window.

        Returns
        -------
        dash_table.DataTable
            A ready-to-render Dash table with metric names, values, and short
            descriptions.
        """
        info = {}
        explain = {}
        self.cumulative = (1 + self.returns).cumprod()

        nb_years = int(self.portfolio.data.period[:-1])
        pa_perf = (round(((self.cumulative.iloc[-1]) ** (1 / nb_years) - 1) * 100, 1))
        info['CAGR'] = str(round(pa_perf, 1)) + ' %'
        explain['CAGR'] = 'Average annual growth rate'

        sharpe = self.returns.mean() / self.returns.std()
        info['Sharpe ratio'] = round(sharpe * np.sqrt(12), 2)
        explain['Sharpe ratio'] = 'Risk-adjusted return'

        running_max = self.cumulative.cummax()
        drawdown = (self.cumulative - running_max) / running_max
        info['Max drawdown'] = str(round(drawdown.min() * 100, 1)) + ' %'
        info['Avg drawdown'] = str(round(drawdown.mean() * 100, 1)) + ' %'
        explain['Max drawdown'] = 'Largest peak-to-trough loss'
        explain['Avg drawdown'] = 'Typical loss during downturns'

        label = 'BTC-USD' if self.portfolio.crypto else 'VTI'
        spy = self.portfolio.data.spy[label].pct_change().dropna()[self.cutoff-1:]
        beta = self.returns.cov(spy) / spy.var()
        info['Beta'] = round(beta, 2)
        explain['Beta'] = 'Sensitivity to market movements'

        vol = self.returns.std() * np.sqrt(12)
        info['Volatility'] = round(vol, 2)
        explain['Volatility'] = 'Return fluctuations (risk)'

        var95 = np.percentile(self.returns, (1 - .95) * 100)
        info['VaR 95%'] = str(round(var95 * 100, 1)) + ' %'
        explain['VaR 95%'] = 'Max expected loss at 95% confidence'

        X = sm.add_constant(spy)
        model = sm.OLS(self.returns, X).fit()
        r2 = model.rsquared
        info['R2'] = str(round(100 * r2)) + ' %'
        explain['R2'] = '% of returns explained by benchmark'

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
