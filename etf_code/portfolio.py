"""
Portfolio construction utilities with correlation clustering and a mean–variance
(or Sharpe-style, for crypto) objective.

This module defines:

* :class:`Info` — configuration and utilities (risk scaling, ticker universe).
* :class:`Portfolio` — data wiring and feature engineering over :class:`~data.Data`,
  including de-duplication of highly correlated tickers via hierarchical clustering
  and an objective callable you can pass to optimizers.

Workflow
--------
1. Instantiate :class:`Portfolio` with a target risk level, currency, holdings, etc.
2. It loads market data through :class:`data.Data`.
3. It removes too-new tickers (with missing history) and prunes clusters of
   highly correlated names, keeping the one with the best (lowest) objective value.
4. It exposes :attr:`Portfolio.objective`, a callable that computes a portfolio
   score for a weight vector:

   * Default (non-crypto): ``weight_cov * variance - mean_excess`` (to minimize).
   * Crypto mode: ``- mean_excess / std_excess`` (to *maximize* Sharpe; we minimize
     the negative).

Notes
-----
* Correlation clustering uses average linkage on the distance matrix
  ``1 - |corr|`` with threshold ``1 - threshold_correlation``.
* The working universe can be extended with FX pseudo-tickers (non-base currencies).
  Color maps are not assigned in this module.
"""

from data import Data
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np


class Info:
    """
    Shared portfolio information and utilities (risk scaling, universes).

    Class Attributes
    ----------------
    threshold_correlation : float
        Minimum absolute correlation to be considered the "same cluster".
        Used as ``1 - threshold_correlation`` on the correlation distance.
    etf_list : list[str]
        Canonical ETF universe (deduplicated and sorted at import time).
    crypto_list : list[str]
        Canonical crypto tickers (``*-USD``) for the crypto universe.
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
    rates : dict[str, float] | None
        Optional mapping of currency pseudo-tickers to annualized rates (in %).
        Forwarded to :class:`data.Data` to adjust currency return columns.
    crypto : bool
        If ``True``, use the crypto universe and Sharpe-style objective.

    Attributes
    ----------
    weight_cov : float | None
        Coefficient in the mean–variance objective (set by :meth:`get_weight_cov`).
    risk, cash, holdings, allow_short, currency, rates, crypto : see parameters
    etf_list : list[str]
        Selected universe for this instance (ETFs or crypto).
    n : int
        Current universe size (set after :attr:`etf_list` finalization).
    """

    threshold_correlation = .95

    etf_list = [
        'SPY', 'QQQ', 'DIA', 'MDY', 'IWM', 'XLY', 'XLP', 'XLE', 'XLV', 'XLF', 'XLI', 'XLB', 'XLK', 'XLU', 'EFA', 'EEM',
        'EWA', 'EWC', 'EWH', 'EWG', 'EWI', 'EWU', 'EWM', 'EWS', 'EWP', 'EWD', 'LQD', 'TLT', 'TIP', 'GLD', 'VTI',
        'IWN', 'IUSG', 'IYJ', 'EWL', 'VHT', 'IWB', 'XLU', 'IGE', 'RTH', 'VWO', 'IWV', 'EWW', 'EWC', 'EWN', 'VPU', 'PWB',
        'VIS', 'IYM', 'SPYV', 'SLYV', 'IUSV', 'AGG', 'IWF', 'EWZ', 'LQD', 'ILCB', 'IXN', 'VDE', 'VOX', 'XLG', 'IVW',
        'IJK', 'XLP', 'XSMO', 'IXC', 'EWY', 'IGM', 'IJH', 'PEJ', 'IVV', 'IYY', 'SOXX', 'EWP', 'VPL', 'IYH', 'VTV',
        'EWT', 'IYW', 'IMCG', 'EWH', 'IGPT', 'PJP', 'SPYG', 'ITOT', 'FXI', 'EWI', 'XLE', 'XLY', 'EWA', 'ILCG', 'IMCV',
        'XLI', 'IWM', 'DVY', 'VBK', 'EWG', 'IGV', 'IJS', 'XNTK', 'IYT', 'SPTM', 'PEY', 'VBR', 'EEM', 'PWV', 'TLT',
        'VFH', 'IEV', 'VB', 'SPEU', 'VGK', 'IYG', 'IWP', 'VTI', 'FEZ', 'EZU', 'IWR', 'VV', 'XLB', 'EWU', 'IJJ', 'IJR',
        'EFA', 'EPP', 'IEF', 'VDC', 'IBB', 'PBW', 'TIP', 'IWS', 'IYE', 'IWO', 'VUG', 'SUSA', 'ILCV', 'IYK', 'XMMO',
        'XLV', 'ONEQ', 'SHY', 'ISCB', 'EWJ', 'VXF', 'EWQ', 'PSI', 'ILF', 'IYR', 'IXG', 'IWD', 'IXP', 'VO', 'IDU', 'VGT',
        'EWD', 'IYZ', 'ISCV', 'ICF', 'IOO', 'SLYG', 'VCR', 'EWS', 'EZA', 'IVE', 'XLF', 'IMCB', 'IYF', 'VAW', 'OEF',
        'IJT', 'RWR', 'IXJ', 'SMH', 'IYC', 'ISCG', 'VNQ', 'XMVM', 'RSP', 'DGT', 'XLK',
        'SI=F', 'PL=F', 'PA=F'
    ]
    etf_list += [
        "^GSPC", "^DJI", "^IXIC", "^NDX", "^RUT", "^FTSE", "^GDAXI", "^FCHI",
        "^STOXX50E", "^STOXX", "^N225", "^HSI", "000001.SS", "399001.SZ", "^BSESN",
        "^NSEI", "^AXJO", "^GSPTSE", "^BVSP"
    ]

    crypto_list  = ['BTC', 'ETH']#, 'XRP', 'SOL', 'DOGE', 'ADA', 'LINK', 'AVAX', 'XLM', 'HBAR', 'LTC', 'CRO', 'DOT', 'AAVE', 'NEAR', 'ETC']
    crypto_list = [f'{x}-USD' for x in crypto_list]

    etf_list = sorted(list(set(etf_list)))
    crypto_list = sorted(list(set(crypto_list)))

    name = {
        1: 'Low risk',
        2: 'Medium risk',
        3: 'High risk'
    }

    def __init__(self, risk, cash, holdings, currency, allow_short, rates, crypto):
        """
        Construct an :class:`Info` object and derive risk-related utilities.

        Parameters
        ----------
        risk : int
            Discrete risk appetite (1–3 recommended; larger means more risk).
        cash : float
            Cash on hand (used by :meth:`Portfolio.get_liquidity`).
        holdings : dict[str, float] | None
            Current holdings as ``{ticker: value}``. If ``None``, treated as empty.
        currency : str | None
            Base currency (defaults to ``"USD"`` if ``None``).
        allow_short : bool
            Whether shorting is allowed conceptually (no effect here).
        rates : dict[str, float] | None
            Optional mapping of currency pseudo-tickers to annualized rates (in %);
            passed through to :class:`data.Data`.
        crypto : bool
            If ``True``, use the crypto universe and Sharpe-style objective.

        Notes
        -----
        Calls :meth:`get_weight_cov` to set :attr:`weight_cov` based on ``risk``.
        """
        self.weight_cov = None
        self.risk = risk
        self.cash = cash
        self.holdings = holdings if holdings else {}
        self.rates = rates if rates else {}
        self.allow_short = allow_short
        self.crypto = crypto
        self.currency = currency if currency else 'USD'
        self.get_weight_cov()
        self.name = 'Risk ' + str(self.risk)
        self.etf_list = Info.crypto_list if self.crypto else Info.etf_list
        self.n = len(self.etf_list)

        if self.allow_short:
            self.add_short_ticker()

    def add_short_ticker(self):
        short_list = [f'-- {ticker}' for ticker in self.etf_list]
        self.etf_list += short_list

    def get_weight_cov(self):
        """
        Derive the risk-aversion coefficient used in the (non-crypto) objective.

        The coefficient is computed from the discrete :attr:`risk` approximately as::

            weight_cov = 52 * exp(-0.326 * risk) - 2

        Larger ``risk`` implies a smaller penalty on variance.

        Special cases
        -------------
        * If ``risk == 10``, an extra ``+ 1/3`` is added (legacy calibration).

        Returns
        -------
        None
        """
        self.weight_cov = 52 * np.exp(-0.326 * self.risk) - 2
        if self.risk == 10:
            self.weight_cov += 1/3


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
    rates : dict[str, float] | None, optional
        Optional mapping of currency pseudo-tickers to annualized rates (in %);
        forwarded to :class:`data.Data` to adjust currency return columns.
    crypto : bool, optional
        If ``True``, use the crypto universe and a Sharpe-style objective
        (negative mean over std). Default ``False``.

    Attributes
    ----------
    data : data.Data
        Data access object (FX, RF, prices/returns).
    etf_list : list[str]
        Working universe (ETF list; extended with FX pseudo-tickers when not crypto).
    n : int
        Current universe size.
    liquidity : float | None
        Cash plus current holdings value (set by :meth:`get_liquidity`).
    objective : callable | None
        Objective function (set by :meth:`get_objective`).
    cov_excess_returns : numpy.ndarray | None
        Covariance matrix of excess returns (set during initialization).
    """

    def __init__(self, risk=3, cash=100, holdings=None, currency=None, allow_short=False, static=False, backtest=None, rates=None, crypto=False):
        """
        Initialize a :class:`Portfolio`, load data, and prune the universe.

        Steps
        -----
        1. Initialize :class:`Info` (risk, currency, universe).
        2. Load market data via :class:`data.Data`.
        3. Extend the universe with FX pseudo-tickers (non-crypto mode).
        4. Drop tickers that are too new (contain missing history).
        5. Instantiate the objective.
        6. Cluster by absolute correlation and keep one representative per cluster
           (the one minimizing the objective).
        7. Compute covariance of excess returns and finalize the objective.

        Returns
        -------
        None
        """
        super().__init__(risk, cash, holdings, currency, allow_short, rates, crypto)

        self.liquidity, self.objective, self.cov_excess_returns = None, None, None
        self.data = Data(self.currency, self.etf_list, static=static, backtest=backtest, rates=self.rates, crypto=crypto, allow_short=self.allow_short)

        if not crypto:
            # Extend with currency pseudo-tickers so FX can be considered.
            self.etf_list += [ticker for ticker in Data.possible_currencies]

        self.etf_list = sorted(list(set(self.etf_list)))
        self.n = len(self.etf_list)

        self.drop_too_new()

        self.get_objective()
        self.drop_highly_correlated()

        self.get_liquidity()
        self.cov_excess_returns = self.data.excess_returns.cov().values
        self.get_objective()




    def remove_etf(self, ticker):
        """
        Remove a ticker from the working universe and all derived tables.

        This method drops the column from :attr:`data.nav`, :attr:`data.returns`,
        :attr:`data.log_returns`, and :attr:`data.excess_returns`, updates
        :attr:`etf_list`, and decrements :attr:`n`.

        Parameters
        ----------
        ticker : str
            Ticker symbol to remove.

        Raises
        ------
        KeyError
            If the ticker is not present in the tables.
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
        """
        log_returns_without_currency = self.data.log_returns.copy()

        if not self.crypto:
            # Keep the base currency column out of the clustering universe.
            log_returns_without_currency.drop(self.currency, axis=1, inplace=True)

        corr = log_returns_without_currency.corr(method='pearson', min_periods=2).abs()
        tickers = corr.columns.tolist()
        np.fill_diagonal(corr.values, 1.0)
        corr = corr.fillna(0.0)
        dist = 1.0 - corr
        dist = 0.5 * (dist + dist.T)
        dist = np.clip(dist, 0.0, 2.0)
        np.fill_diagonal(dist.values, 0.0)
        condensed = squareform(dist.values, checks=True)
        Z = linkage(condensed, method='average')
        t = 1.0 - Portfolio.threshold_correlation  # e.g., 0.05 for 0.95
        clusters = fcluster(Z, t, criterion='distance')

        cluster_df = pd.DataFrame({'ETF': tickers, 'Cluster': clusters})

        # Evaluate objective per single ticker (will use Sharpe-style in crypto mode).
        obj_values = {ticker: self.objective(single_ticker=ticker) for ticker in self.etf_list}
        obj_values = pd.Series(obj_values, name='obj_values')

        cluster_df = cluster_df.set_index('ETF').join(obj_values)
        best_etfs = cluster_df.groupby('Cluster')['obj_values'].idxmin().tolist()

        to_drop = [ticker for ticker in self.etf_list if ticker not in best_etfs]
        for ticker in to_drop:
            if ticker != self.currency:
                self.remove_etf(ticker)

    def get_liquidity(self):
        """
        Compute total liquidity as cash plus current holdings value.

        Returns
        -------
        float
            Total liquidity.
        """
        self.liquidity = self.cash + sum(self.holdings.values())

    def get_objective(self):
        """
        Define and store the portfolio objective function.

        Non-crypto mode (mean–variance, minimize):
            ``f(w) = weight_cov * (w^T Σ_excess w) - mean(ExcessReturns @ w)``

        Crypto mode (Sharpe-style, minimize negative Sharpe):
            ``f(w) = - mean(ExcessReturns @ w) / std(ExcessReturns @ w)``

        The callable also supports a single-ticker evaluation mode via
        ``single_ticker='SPY'`` which computes the same quantity using that
        column's excess returns.

        Notes
        -----
        * :attr:`weight_cov` controls the variance penalty in non-crypto mode.
        * :attr:`cov_excess_returns` is set after data-dependent pruning.
        """
        def f(w=np.zeros(self.n), single_ticker=None):
            """
            Mean–variance objective (non-crypto).

            Parameters
            ----------
            w : numpy.ndarray
                Portfolio weights (ignored when ``single_ticker`` is set).
            single_ticker : str | None
                If provided, evaluate the objective for a single column.

            Returns
            -------
            float
                Objective value (lower is better).
            """
            if single_ticker:
                excess_series = self.data.excess_returns[single_ticker]
                mean = excess_series.mean()
                var = excess_series.var()
                return self.weight_cov * var - mean

            excess_series = self.data.excess_returns @ w
            mean = excess_series.mean()
            return self.weight_cov * (w @ self.cov_excess_returns @ w) - mean

        def f_crypto(w=np.zeros(self.n), single_ticker=None):
            """
            Sharpe-style objective (crypto): minimize negative mean/std.

            Parameters
            ----------
            w : numpy.ndarray
                Portfolio weights (ignored when ``single_ticker`` is set).
            single_ticker : str | None
                If provided, evaluate the objective for a single column.

            Returns
            -------
            float
                Objective value (lower is better).
            """
            if single_ticker:
                excess_series = self.data.excess_returns[single_ticker]
                mean = excess_series.mean()
                std = excess_series.std()
                return -mean / std

            excess_series = self.data.excess_returns @ w
            mean = excess_series.mean()
            return -mean / excess_series.std()

        self.objective = f_crypto if self.crypto else f
