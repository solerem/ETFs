"""
Rolling backtest utilities for an optimized portfolio.

This module defines :class:`Backtest`, which:

* Walks forward through monthly dates and re-optimizes the portfolio at each step
  using in-sample data up to that date (via :class:`~portfolio.Portfolio` and
  :class:`~opti.Opti` in *static* mode with an in-sample cutoff).
* Optionally smooths the resulting weight paths.
* Computes out-of-sample (test) returns from the held-out period.
* Produces diagnostic plots (equity curve, weights stack, and performance
  attribution) as Dash-ready images while saving PNGs to disk.

Notes
-----
* The train/test split is controlled by :attr:`Backtest.ratio_train_test`
  (default ``17/20`` i.e., 85% train, 15% test).
* Re-optimization loops over months from the cutoff to the end of the data,
  constructing a fresh :class:`~portfolio.Portfolio` with ``static=True`` and
  ``backtest=<current_date>`` at each step.
"""

import pandas as pd
from portfolio import Portfolio
from opti import Opti
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from data import Data


class Backtest:
    """
    Rolling re-optimization backtest.

    Class Attributes
    ----------------
    ratio_train_test : float
        Fraction of the sample used for training/in-sample (default: ``17/20``).

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
    returns : pandas.Series | None
        Out-of-sample (test) portfolio returns.
    n : int | None
        Number of rows (time points) in the underlying NAV table.
    cutoff : int | None
        Index position separating train/test based on :attr:`ratio_train_test`.
    index : list[pandas.Timestamp] | None
        Copy of the underlying ``DatetimeIndex`` for iteration.
    returns_decomp : pandas.DataFrame | None
        Per-asset contributions to test-period returns (weights × returns).
    """

    ratio_train_test = .95
    ratio_train_test = 17 / 20

    def __init__(self, opti):
        """
        Initialize the backtest, compute rolling weights, and derive returns.

        The constructor:
        1) Parses data and builds a time-series of optimal weights by repeatedly
           re-optimizing up to each test date in *static* mode.
        2) (Optional) allows weight smoothing via :meth:`smoothen_weights`
           (currently commented out).
        3) Computes test-period returns via :meth:`get_returns`.

        :param opti: Optimizer instance providing the baseline portfolio.
        :type opti: Opti
        :returns: ``None``.
        :rtype: None
        """
        self.opti = opti
        self.portfolio = self.opti.portfolio
        self.to_consider = self.opti.optimum.keys()
        self.w_opt, self.returns, self.n, self.cutoff, self.index, self.returns_decomp = None, None, None, None, None, None
        self.parse_data()
        # self.smoothen_weights()
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

        :returns: ``None``.
        :rtype: None
        """

        self.n = len(self.portfolio.data.nav)
        self.cutoff = round(Backtest.ratio_train_test * self.n)
        self.index = list(self.portfolio.data.nav.index)

        self.w_opt = pd.DataFrame({ticker: [] for ticker in self.opti.portfolio.etf_list})
        for i in tqdm(range(self.cutoff, self.n)):
            portfolio = Portfolio(risk=self.portfolio.risk, currency=self.portfolio.currency,
                                  allow_short=self.portfolio.allow_short, static=True, backtest=self.index[i], rates=self.portfolio.rates, crypto=self.opti.portfolio.crypto)
            optimum = Opti(portfolio).optimum_all
            self.w_opt.loc[self.index[i]] = optimum

    def smoothen_weights(self):
        """
        Apply simple exponential smoothing (2/3 previous + 1/3 current).

        This can reduce churn in the weights before computing test returns.

        Side Effects
        ------------
        Overwrites :attr:`w_opt` with the smoothed series.

        :returns: ``None``.
        :rtype: None
        """
        self.w_opt.fillna(0, inplace=True)
        smoothed_df = pd.DataFrame(index=self.w_opt.index, columns=self.w_opt.columns, dtype=float)
        smoothed_df.iloc[0] = self.w_opt.iloc[0]

        for t in range(1, len(self.w_opt)):
            smoothed_df.iloc[t] = (self.w_opt.iloc[t] + 2 * smoothed_df.iloc[t - 1]) / 3

        self.w_opt = smoothed_df

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

        :returns: ``None``.
        :rtype: None
        """
        self.returns_decomp = Data.get_test_data_backtest(self.portfolio.data.returns, self.index[self.cutoff])
        self.returns_decomp *= self.w_opt
        self.returns = self.returns_decomp.sum(axis=1)

    def plot_backtest(self):
        """
        Plot the backtest equity curve vs. benchmark and risk-free leg.

        The title includes annualized performance (p.a.) and maximum drawdown
        over the test window.

        :returns: Dash image component with the figure embedded.
        :rtype: dash.html.Img
        """
        cumulative = (1 + self.returns).cumprod()

        fig, ax = plt.subplots()
        ax.plot((cumulative - 1) * 100, label=str(self.portfolio.name) + f' ({self.portfolio.currency})')

        spy = self.portfolio.data.spy.copy()
        spy = spy.loc[self.index[self.cutoff]:]
        spy = (spy / spy.iloc[0] - 1) * 100
        ax.plot(spy, label=f'Total stock market ({self.portfolio.currency})', linestyle='--')

        rf_rate = ((self.portfolio.data.rf_rate.loc[self.index[self.cutoff]:] + 1).cumprod() - 1) * 100
        ax.plot(rf_rate, label='Rate', linestyle='--')

        ax.axhline(0, color='black')

        nb_years = int(self.opti.portfolio.data.period[:-1]) * (1 - Backtest.ratio_train_test)
        pa_perf = round(((cumulative.iloc[-1]) ** (1 / nb_years) - 1) * 100, 1)

        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = round(drawdown.min() * 100, 1)
        plt.setp(ax.get_xticklabels(), rotation=45)

        ax.set_title(f'Backtest ({pa_perf}% p.a., {max_drawdown}% max drawdown)')

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
        cumulative mean weight of the plotted set reaches at least 90%.

        :returns: Dash image component with the figure embedded.
        :rtype: dash.html.Img
        """
        included = set(self.to_consider)
        all_tickers = set(self.w_opt.columns)
        remaining = list(all_tickers - included)
        mean_weights = self.w_opt.mean()
        sorted_remaining = sorted(remaining, key=lambda x: mean_weights[x], reverse=True)

        total_weight = mean_weights.sum()
        included_weight = mean_weights[list(included)].sum()

        while included_weight / total_weight < 0.9 and sorted_remaining:
            next_ticker = sorted_remaining.pop(0)
            included.add(next_ticker)
            included_weight += mean_weights[next_ticker]

        tickers_to_plot = list(included)
        colors = [self.portfolio.color_map[ticker] for ticker in tickers_to_plot]

        fig, ax = plt.subplots()
        ax.stackplot(
            self.w_opt.index,
            100 * self.w_opt[tickers_to_plot].T,
            labels=tickers_to_plot,
            colors=colors
        )

        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.set_title(f'Weights history')
        ax.axhline(100, color='black')

        ax.set_ylabel('%')
        ax.legend()

        output_path = Opti.graph_dir_path / f'{self.portfolio.currency}/{self.portfolio.name}- Backtest_weights.png'
        return Opti.save_fig_as_dash_img(fig, output_path)

    def plot_perf_attrib(self):
        """
        Plot cumulative performance attribution for selected tickers.

        Uses the per-asset return contributions in :attr:`returns_decomp` and
        accumulates them through time, plotted in percent.

        :returns: Dash image component with the figure embedded.
        :rtype: dash.html.Img
        """
        returns = self.returns_decomp[self.to_consider]

        fig, ax = plt.subplots()
        for col in self.to_consider:
            ax.plot(returns.index, (returns[col].cumsum()) * 100, label=col, color=self.portfolio.color_map[col])

        ax.axhline(0, color='black')
        plt.setp(ax.get_xticklabels(), rotation=45)

        ax.set_title(f'Backtest Performance Attribution')

        ax.set_ylabel('%')
        ax.legend()
        ax.grid()

        output_path = Opti.graph_dir_path / f'{self.portfolio.currency}/{self.portfolio.name}- Backtest_perf_attrib.png'
        return Opti.save_fig_as_dash_img(fig, output_path)


    def plot_drawdown(self):
        """
        Plot the portfolio drawdown curve (area below zero).

        :returns: Dash image component for embedding in a layout.
        :rtype: dash.html.Img
        """
        cumulative = (1 + self.returns).cumprod()

        rolling_max = cumulative.cummax()
        drawdown = cumulative / rolling_max - 1

        fig, ax = plt.subplots()
        ax.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=.5)

        ax.set_title(f'Drawdown')
        ax.set_ylabel('%')
        ax.grid()

        output_path = Opti.graph_dir_path / f'{self.portfolio.currency}/{self.portfolio.name}- Backtest_drawdown.png'
        return Opti.save_fig_as_dash_img(fig, output_path)
