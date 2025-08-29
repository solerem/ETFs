"""
Portfolio construction utilities with correlation clustering and a mean–variance
objective.

This module defines:

* :class:`Info` — configuration and utilities (risk scaling, colors, ticker universe).
* :class:`Portfolio` — data wiring and feature engineering over :class:`~data.Data`,
  including de-duplication of highly correlated tickers via hierarchical clustering
  and a convex mean–variance-style objective you can pass to optimizers.

The workflow is:

1. Instantiate :class:`Portfolio` with a target risk level, currency, holdings, etc.
2. It loads market data through :class:`data.Data`.
3. It removes too-new tickers (with missing history) and prunes clusters of
   highly correlated names, keeping the one with the best (lowest) objective
   value.
4. It exposes :attr:`Portfolio.objective`, a callable that computes
   ``weight_cov * variance - mean_excess`` for a weight vector, suitable for
   SLSQP/L-BFGS-B minimization.

Notes
-----
* Correlation clustering uses average linkage on the distance matrix
  ``1 - |corr|`` with threshold ``1 - threshold_correlation``.
* Colors are assigned deterministically from Matplotlib's ``tab20`` colormap,
  extended with FX pseudo-tickers for non-base currencies.
"""

from data import Data
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


class Info:
    """
    Shared portfolio information and utilities (risk scaling, color maps, universe).

    Class Attributes
    ----------------
    threshold_correlation : float
        Minimum absolute correlation to be considered the "same cluster".
        Used as ``1 - threshold_correlation`` on the correlation distance.
    etf_list : list[str]
        Canonical ETF universe (deduplicated and sorted at import time).
    name : dict[int, str]
        Human labels for discrete risk tiers (may be overridden per instance).

    Parameters
    ----------
    risk : int
        Discrete risk appetite (e.g., 1=low, 2=medium, 3=high). Drives
        :attr:`weight_cov`.
    cash : float
        Available cash (used in :meth:`Portfolio.get_liquidity`).
    holdings : dict[str, float] | None
        Current positions as a mapping ``{ticker: current_value}``. Optional.
    currency : str | None
        Base currency (one of :attr:`data.Data.possible_currencies`). Defaults to
        ``"USD"`` when ``None``.
    allow_short : bool
        Whether shorting is conceptually allowed (does not alter logic here,
        but exposed for downstream optimizers).

    Attributes
    ----------
    color_map : dict[str, str] | None
        Mapping from ticker to HEX color for plotting (set by :meth:`get_color_map`).
    weight_cov : float | None
        Coefficient in the mean–variance objective (set by :meth:`get_weight_cov`).
    risk, cash, holdings, allow_short, currency : see parameters
    n : int
        Current universe size (set after :attr:`etf_list` finalization).
    """

    threshold_correlation = .95

    etf_list = [
        'SPY', 'QQQ', 'DIA', 'MDY', 'IWM', 'XLY', 'XLP', 'XLE', 'XLV', 'XLF', 'XLI', 'XLB', 'XLK', 'XLU', 'EFA', 'EEM',
        'EWA', 'EWC', 'EWH', 'EWG', 'EWI', 'EWJ', 'EWU', 'EWM', 'EWS', 'EWP', 'EWD', 'LQD', 'TLT', 'TIP', 'GLD', 'VTI',
        'IWN', 'IUSG', 'IYJ', 'EWL', 'VHT', 'IWB', 'XLU', 'IGE', 'RTH', 'VWO', 'IWV', 'EWW', 'EWC', 'EWN', 'VPU', 'PWB',
        'VIS', 'IYM', 'SPYV', 'SLYV', 'IUSV', 'AGG', 'IWF', 'EWZ', 'LQD', 'ILCB', 'IXN', 'VDE', 'VOX', 'XLG', 'IVW',
        'IJK', 'XLP', 'XSMO', 'IXC', 'EWY', 'IGM', 'IJH', 'PEJ', 'IVV', 'IYY', 'SOXX', 'EWP', 'VPL', 'IYH', 'VTV',
        'EWT', 'IYW', 'IMCG', 'EWH', 'IGPT', 'PJP', 'SPYG', 'ITOT', 'FXI', 'EWI', 'XLE', 'XLY', 'EWA', 'ILCG', 'IMCV',
        'XLI', 'IWM', 'DVY', 'VBK', 'EWG', 'IGV', 'IJS', 'XNTK', 'IYT', 'SPTM', 'PEY', 'VBR', 'EEM', 'PWV', 'TLT',
        'VFH', 'IEV', 'VB', 'SPEU', 'VGK', 'IYG', 'IWP', 'VTI', 'FEZ', 'EZU', 'IWR', 'VV', 'XLB', 'EWU', 'IJJ', 'IJR',
        'EFA', 'EPP', 'IEF', 'VDC', 'IBB', 'PBW', 'TIP', 'IWS', 'IYE', 'IWO', 'VUG', 'SUSA', 'ILCV', 'IYK', 'XMMO',
        'XLV', 'ONEQ', 'SHY', 'ISCB', 'EWJ', 'VXF', 'EWQ', 'PSI', 'ILF', 'IYR', 'IXG', 'IWD', 'IXP', 'VO', 'IDU', 'VGT',
        'EWD', 'IYZ', 'ISCV', 'ICF', 'IOO', 'SLYG', 'VCR', 'EWS', 'EZA', 'IVE', 'XLF', 'IMCB', 'IYF', 'VAW', 'OEF',
        'IJT', 'RWR', 'IXJ', 'SMH', 'IYC', 'ISCG', 'VNQ', 'XMVM', 'RSP', 'DGT', 'XLK'
    ]

    etf_list = sorted(list(set(etf_list)))

    name = {
        1: 'Low risk',
        2: 'Medium risk',
        3: 'High risk'
    }

    def __init__(self, risk, cash, holdings, currency, allow_short):
        """
        Construct an :class:`Info` object and derive risk/plotting utilities.

        :param risk: Discrete risk appetite (1–3 recommended).
        :type risk: int
        :param cash: Cash on hand (for liquidity calculations).
        :type cash: float
        :param holdings: Current holdings as ``{ticker: value}``.
        :type holdings: dict[str, float] | None
        :param currency: Base currency (defaults to ``"USD"`` if ``None``).
        :type currency: str | None
        :param allow_short: Whether shorting is allowed conceptually.
        :type allow_short: bool
        :returns: ``None``.
        :rtype: None
        """
        self.color_map, self.weight_cov = None, None
        self.risk = risk
        self.cash = cash
        self.holdings = holdings if holdings else {}
        self.allow_short = allow_short
        self.currency = currency if currency else 'USD'
        self.get_weight_cov()
        self.name = 'Risk ' + str(self.risk)
        self.etf_list = Info.etf_list
        self.n = len(self.etf_list)
        self.get_color_map()

    def get_weight_cov(self):
        """
        Derive the risk-aversion coefficient used in the objective.

        The coefficient is computed from the discrete :attr:`risk` as::

            weight_cov = 52 * exp(-0.3259 * risk) - 2

        Larger ``risk`` implies a smaller penalty on variance.

        :returns: ``None``.
        :rtype: None
        """
        self.weight_cov = 52 * np.exp(-0.3259 * self.risk) - 2

    def get_color_map(self):
        """
        Build a deterministic HEX color mapping for the current universe.

        Colors are drawn from Matplotlib's ``tab20`` colormap. FX pseudo-tickers
        for all non-base currencies are appended so that currency series can be
        plotted alongside ETFs.

        :returns: ``None``.
        :rtype: None
        """
        cmap = cm.get_cmap('tab20', self.n)
        self.color_map = {asset: mcolors.to_hex(cmap(i)) for i, asset in enumerate(
            self.etf_list + [ticker for ticker in Data.possible_currencies if ticker != self.currency])}


class Portfolio(Info):
    """
    Portfolio wrapper that loads data, prunes redundancy, and exposes an objective.

    Parameters
    ----------
    risk : int, optional
        Discrete risk appetite (default: ``3``).
    cash : float, optional
        Cash on hand (default: ``100``).
    holdings : dict[str, float] | None, optional
        Current positions as a mapping ``{ticker: value}`` (default: ``None``).
    currency : str | None, optional
        Base currency (defaults to ``"USD"`` if ``None``).
    allow_short : bool, optional
        Whether shorting is allowed conceptually (default: ``False``).
    static : bool, optional
        If ``True``, read cached CSVs instead of downloading (passed to
        :class:`data.Data`).
    backtest : pandas.Timestamp | str | None, optional
        If provided, all series are truncated to ``.loc[:backtest]`` for
        in-sample preparation (passed to :class:`data.Data`).

    Attributes
    ----------
    data : data.Data
        Data access object (FX, RF, ETF prices/returns, crypto weights, etc.).
    etf_list : list[str]
        Working universe (ETF list plus FX pseudo-tickers for non-base currencies).
    n : int
        Current universe size.
    liquidity : float | None
        Cash plus current holdings value (set by :meth:`get_liquidity`).
    objective : callable | None
        Mean–variance-style objective function (set by :meth:`get_objective`).
    cov_excess_returns : numpy.ndarray | None
        Covariance matrix of excess returns (set during initialization).
    crypto_opti : dict[str, float]
        Crypto tangency-portfolio weights copied from :attr:`data.Data.crypto_opti`.
    """

    def __init__(self, risk=3, cash=100, holdings=None, currency=None, allow_short=False, static=False, backtest=None):
        """
        Initialize a :class:`Portfolio`, load data, and prune the universe.

        The constructor performs the following steps:

        1. Initialize super class (:class:`Info`) to set risk, currency, colors.
        2. Load market data via :class:`data.Data`.
        3. Extend the universe with FX pseudo-tickers (non-base currencies).
        4. Drop tickers that are too new (contain missing history).
        5. Instantiate the mean–variance objective.
        6. Cluster by absolute correlation and keep one representative per cluster
           (the one minimizing the objective).
        7. Compute covariance of excess returns and finalize the objective.
        8. Copy crypto-optimized weights from :class:`data.Data`.

        :param risk: Discrete risk appetite.
        :type risk: int
        :param cash: Cash on hand.
        :type cash: float
        :param holdings: Current holdings mapping.
        :type holdings: dict[str, float] | None
        :param currency: Base currency (defaults to ``"USD"`` if ``None``).
        :type currency: str | None
        :param allow_short: Whether shorting is allowed conceptually.
        :type allow_short: bool
        :param static: Use cached CSVs instead of downloading if ``True``.
        :type static: bool
        :param backtest: In-sample cutoff (inclusive) for training data.
        :type backtest: pandas.Timestamp | str | None
        :returns: ``None``.
        :rtype: None
        """
        super().__init__(risk, cash, holdings, currency, allow_short)

        self.liquidity, self.objective, self.cov_excess_returns = None, None, None

        self.data = Data(self.currency, self.etf_list, static=static, backtest=backtest)
        self.etf_list += [ticker for ticker in Data.possible_currencies if ticker != self.currency]
        self.etf_list = sorted(list(set(self.etf_list)))
        self.n = len(self.etf_list)

        self.drop_too_new()

        self.get_objective()
        self.drop_highly_correlated()
        self.get_liquidity()
        self.cov_excess_returns = self.data.excess_returns.cov().values
        self.get_objective()
        self.crypto_opti = self.data.crypto_opti

    def remove_etf(self, ticker):
        """
        Remove a ticker from the working universe and all derived tables.

        This method drops the column from :attr:`data.nav`, :attr:`data.returns`,
        :attr:`data.log_returns`, and :attr:`data.excess_returns`, updates
        :attr:`etf_list`, and decrements :attr:`n`.

        :param ticker: Ticker symbol to remove.
        :type ticker: str
        :returns: ``None``.
        :rtype: None
        :raises KeyError: If the ticker is not present in the tables.
        """
        self.data.nav.drop(ticker, axis=1, inplace=True)
        self.data.returns.drop(ticker, axis=1, inplace=True)
        self.data.log_returns.drop(ticker, axis=1, inplace=True)
        self.data.excess_returns.drop(ticker, axis=1, inplace=True)
        self.etf_list = list(self.data.nav.columns)
        self.n -= 1

    def drop_too_new(self):
        """
        Prune tickers with insufficient history (i.e., containing NaNs).

        Any column in :attr:`data.nav` that contains missing values is removed
        via :meth:`remove_etf`.

        :returns: ``None``.
        :rtype: None
        """
        to_drop = self.data.nav.columns[self.data.nav.isna().any()].tolist()
        for col in to_drop:
            self.remove_etf(col)

    def drop_highly_correlated(self):
        """
        Cluster highly correlated tickers and keep one representative per cluster.

        Procedure
        ---------
        1. Compute ``corr = |corr(log_returns)|``.
        2. Convert to a distance matrix ``1 - corr`` and run average-linkage
           hierarchical clustering.
        3. Cut the dendrogram at distance ``1 - threshold_correlation``.
        4. Within each cluster, evaluate :attr:`objective` as a single-ticker
           portfolio and **keep the ticker that minimizes the objective**, drop
           the others.

        Side Effects
        ------------
        Updates :attr:`etf_list`, :attr:`n`, and removes columns from the data
        frames via :meth:`remove_etf`.

        :returns: ``None``.
        :rtype: None
        """
        correlation_matrix = self.data.log_returns.corr().abs()

        distance_matrix = 1 - correlation_matrix
        linkage_matrix = linkage(squareform(distance_matrix), method='average')

        threshold = 1 - Portfolio.threshold_correlation
        clusters = fcluster(linkage_matrix, threshold, criterion='distance')

        cluster_df = pd.DataFrame({'ETF': self.etf_list, 'Cluster': clusters})

        obj_values = {ticker: self.objective(single_ticker=ticker) for ticker in self.etf_list}
        obj_values = pd.Series(obj_values, name='obj_values')

        cluster_df = cluster_df.set_index('ETF').join(obj_values)
        best_etfs = cluster_df.groupby('Cluster')['obj_values'].idxmin().tolist()

        to_drop = [ticker for ticker in self.etf_list if ticker not in best_etfs]
        for ticker in to_drop:
            self.remove_etf(ticker)

    def get_liquidity(self):
        """
        Compute total liquidity as cash plus current holdings value.

        :returns: Total liquidity.
        :rtype: float
        """
        self.liquidity = self.cash + sum(self.holdings.values())

    def get_objective(self):
        """
        Define and store the mean–variance-style objective function.

        The objective is designed for *minimization* and is defined as::

            f(w) = weight_cov * (w^T Σ_excess w) - mean(ExcessReturns @ w)

        For convenience, the callable also supports a single-ticker evaluation
        mode via ``single_ticker='SPY'`` which computes the same quantity using
        that column's excess returns.

        Notes
        -----
        * :attr:`weight_cov` controls the variance penalty relative to mean.
        * :attr:`cov_excess_returns` is set after data-dependent pruning.

        :returns: ``None`` (sets :attr:`objective` to a callable).
        :rtype: None
        """
        def f(w=np.zeros(self.n), single_ticker=None):
            """
            Objective function handle.

            :param w: Portfolio weights (ignored when ``single_ticker`` is set).
            :type w: numpy.ndarray
            :param single_ticker: If provided, evaluate the objective for a
                                  single column as a 1-asset portfolio.
            :type single_ticker: str | None
            :returns: Objective value (lower is better).
            :rtype: float
            """
            if single_ticker:
                excess_series = self.data.excess_returns[single_ticker]
                mean = excess_series.mean()
                var = excess_series.var()
                return self.weight_cov * var - mean

            excess_series = self.data.excess_returns @ w
            mean = excess_series.mean()
            return self.weight_cov * (w @ self.cov_excess_returns @ w) - mean

        self.objective = f
