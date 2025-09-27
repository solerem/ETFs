"""
Data acquisition and preprocessing utilities for multi-asset portfolio work.

This module centers around :class:`Data`, which downloads (or loads cached)
time series for:

* Foreign exchange (FX) rates to convert assets into a chosen base currency.
* A risk-free rate proxy from ``^IRX`` (13-week T-bill), converted to a
  monthly rate and expressed in the base currency.
* ETF NAV/Close series and derived simple, log, and excess returns.
* A total U.S. equity market proxy (``VTI``) for benchmarking, or ``BTC-USD``
  when ``crypto=True``.
* A simple “alternatives” mapping table loaded from cache to relate tickers to
  preferred substitutes.

It supports a *static* mode that reads/writes CSV caches under
``data_dir_path`` to avoid repeated network calls, and optional backtest
truncation where series are sliced up to a specified date.

Dependencies
------------
``yfinance``, ``pandas``, ``numpy``, ``matplotlib``, and ``scipy`` are used
for retrieval, manipulation, plotting, and optimization.

Examples
--------
Create a dataset in EUR with cached files only:

>>> d = Data(currency="EUR", etf_list=["VWRA.L", "EUNA.L"], static=True)

Create a dataset in USD, download fresh data, and trim in-sample up to
January 2020:

>>> d = Data(currency="USD", etf_list=["VT", "BND"], static=False, backtest="2020-01-01")

Crypto-oriented example (uses BTC as the benchmark and shorter history):

>>> d = Data(currency="USD", etf_list=["BTC-USD"], static=False, crypto=True)
"""

import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import concurrent.futures
from pathlib import Path
from scipy.optimize import minimize


class Data:
    """
    Container for market datasets (FX, risk-free, ETFs; optional crypto) with
    helpers to compute returns, perform currency conversion, and plot series.

    Parameters
    ----------
    currency : str
        Target/base currency code (one of :attr:`possible_currencies`) used to
        express all prices and returns.
    etf_list : list[str]
        List of ETF (or ticker) symbols understood by *yfinance*.
    static : bool, optional
        If ``True``, read from and write to cached CSVs under
        :attr:`data_dir_path` instead of fetching from the network (default:
        ``False``).
    backtest : pandas.Timestamp | str | None, optional
        If provided, time series loaded by this instance are truncated to
        ``.loc[:backtest]`` to form an in-sample set for backtesting. Use
        :meth:`get_test_data_backtest` to obtain the complementary out-of-sample
        slice.
    rates : dict[str, float] | None, optional
        Optional mapping of currency pseudo-tickers to annual rates (in percent).
        If provided, the corresponding currency return columns in :attr:`returns`
        are adjusted to embed these rates as monthly equivalents.
    crypto : bool, optional
        If ``True``, use crypto-oriented caches and settings (e.g., shorter
        history, BTC benchmark). Default is ``False``.

    Attributes
    ----------
    possible_currencies : list[str]
        Whitelist of currency codes for FX handling.
    helper_currencies : list[str]
        Additional FX crosses fetched to improve conversion coverage.
    data_dir_path : pathlib.Path
        Root directory for CSV caches.
    currency_rate : pandas.DataFrame | None
        Monthly FX rates normalized to the base currency. Columns are other
        currencies; values express the conversion **into** the base currency.
        Populated by :meth:`get_currency`.
    etf_currency : dict[str, str] | pandas.Series | None
        Mapping from ticker to its trading currency. Populated by
        :meth:`get_currency`.
    nav : pandas.DataFrame | None
        Monthly close/NAV series for requested tickers, converted into the base
        currency. Populated by :meth:`get_nav_returns`.
    returns : pandas.DataFrame | None
        Simple monthly returns from :attr:`nav`. Populated by
        :meth:`get_nav_returns`.
    log_returns : pandas.DataFrame | None
        Log monthly returns from :attr:`nav`. Populated by
        :meth:`get_nav_returns`.
    rf_rate : pandas.Series | None
        Monthly risk-free rates aligned to month starts and expressed in the
        base currency. Populated by :meth:`get_rf_rate`.
    excess_returns : pandas.DataFrame | None
        Simple returns in excess of :attr:`rf_rate` (broadcast across columns).
        Populated by :meth:`get_nav_returns`.
    spy : pandas.Series | pandas.DataFrame | None
        Benchmark proxy: ``VTI`` (equities) or ``BTC-USD`` when ``crypto=True``,
        expressed in the base currency. Populated by :meth:`get_spy`.
    etf_full_names : pandas.Series | None
        Long names for tickers (and passthrough for currency columns). Populated
        by :meth:`get_full_names`.
    exposure : pandas.DataFrame | None
        Exposures table loaded from cache. Populated by :meth:`get_exposure`.
    alternatives : dict[str, str] | None
        Mapping from ticker (``TICKER``) to a preferred substitute (``BEST``),
        loaded from ``alternatives.csv``. Populated by :meth:`get_alternatives`.
    period : str
        Download lookback used with yfinance (``'20y'`` by default, ``'5y'`` if
        ``crypto=True``).

    Notes
    -----
    This class previously exposed a ``get_crypto``/``crypto_opti`` workflow for
    computing a tangency portfolio. That logic has been removed; ``crypto=True``
    now only toggles data sources, cache names, and the benchmark proxy.
    """

    possible_currencies = ['USD', 'EUR', 'SGD', 'AUD', 'CNH', 'GBP', 'HKD', 'JPY']
    helper_currencies = ['INR', 'CNY', 'BRL', 'CAD']
    data_dir_path = Path(__file__).resolve().parent.parent / "data_dir"

    def __init__(self, currency, etf_list, static=False, backtest=None, rates=None, crypto=False):
        """
        Initialize and eagerly load core datasets.

        This constructor immediately calls, in order:
        :meth:`get_currency`, :meth:`get_rf_rate`, :meth:`get_nav_returns`,
        :meth:`get_spy`, :meth:`get_full_names`, :meth:`get_exposure`,
        :meth:`get_alternatives`.

        :param currency: Base currency code.
        :type currency: str
        :param etf_list: Tickers to download.
        :type etf_list: list[str]
        :param static: If ``True``, use cached CSVs; otherwise fetch and cache.
        :type static: bool
        :param backtest: If set, truncate series to ``.loc[:backtest]``.
        :type backtest: pandas.Timestamp | str | None
        :param rates: Optional mapping of currency columns to annual rates (%).
        :type rates: dict[str, float] | None
        :param crypto: Enable crypto-oriented settings and caches.
        :type crypto: bool
        """
        self.currency_rate, self.nav, self.rf_rate, self.returns, self.excess_returns, self.log_returns, self.etf_currency, self.spy, self.etf_full_names, self.exposure, self.alternatives = None, None, None, None, None, None, None, None, None, None, None
        self.etf_list, self.currency, self.static, self.backtest, self.rates = etf_list, currency, static, backtest, rates
        self.crypto = crypto
        self.period = '5y' if self.crypto else '20y'

        self.get_currency()
        self.get_rf_rate()
        self.get_nav_returns()
        self.get_spy()
        self.get_full_names()
        self.get_exposure()
        self.get_alternatives()

    def drop_test_data_backtest(self, df):
        """
        Trim an object to the in-sample (training) slice for backtesting.

        If :attr:`backtest` is set, this returns ``df.loc[:self.backtest]``;
        otherwise the input is returned unchanged.

        :param df: Time-indexed data (``DatetimeIndex`` recommended).
        :type df: pandas.Series | pandas.DataFrame
        :returns: The trimmed object (same type as ``df``).
        :rtype: pandas.Series | pandas.DataFrame
        """
        if self.backtest:
            df = df.loc[:self.backtest]
        return df

    def get_alternatives(self):
        """
        Load a ticker substitution map from cache into :attr:`alternatives`.

        Reads ``alternatives.csv`` from :attr:`data_dir_path` (semicolon-
        separated). The file must contain at least columns ``TICKER`` and
        ``BEST``. No network call is performed.

        :returns: ``None``.
        :rtype: None
        """
        df = pd.read_csv(Data.data_dir_path / 'alternatives.csv', sep=';')
        self.alternatives = {row['TICKER']: row['BEST'] for _, row in df.iterrows()}

    @staticmethod
    def get_test_data_backtest(df, cutoff):
        """
        Obtain the out-of-sample (test) slice for backtesting.

        Convenience wrapper equivalent to ``df.loc[cutoff:]``.

        :param df: Time-indexed data to slice.
        :type df: pandas.Series | pandas.DataFrame
        :param cutoff: Inclusive start label of the test period.
        :type cutoff: pandas.Timestamp | str
        :returns: The out-of-sample slice from ``cutoff`` onward.
        :rtype: pandas.Series | pandas.DataFrame
        """
        return df.loc[cutoff:]

    def get_currency(self):
        """
        Fetch or load monthly FX rates and detect each ETF's trading currency.

        Behavior
        --------
        * If :attr:`static` is ``True``, read ``currency.csv`` (or
          ``currency_crypto.csv`` when ``crypto=True``) and ``curr_etf.csv``
          from :attr:`data_dir_path`.
        * Otherwise, download monthly FX rates for codes in
          :attr:`possible_currencies` + :attr:`helper_currencies` versus USD
          (e.g., ``USDJPY=X``), normalize everything into the base
          :attr:`currency`, and cache to CSV. ETF trading currencies are
          detected via ``yfinance.Ticker(...).fast_info['currency']`` and cached.

        Side Effects
        ------------
        Sets :attr:`currency_rate` (``DataFrame``) and :attr:`etf_currency`
        (``dict`` or ``Series``).

        :returns: ``None``.
        :rtype: None
        """
        currency_file = 'currency_crypto.csv' if self.crypto else 'currency.csv'
        if self.static:
            self.currency_rate = pd.read_csv(Data.data_dir_path / currency_file, index_col=0)
            self.currency_rate.index = pd.to_datetime(self.currency_rate.index)
        else:
            to_download = [f'USD{ticker}=X' for ticker in Data.possible_currencies+Data.helper_currencies if ticker != 'USD']
            self.currency_rate = yf.download(to_download, period=self.period, interval='1mo', auto_adjust=True)['Close']
            self.currency_rate.to_csv(Data.data_dir_path / currency_file)

        self.currency_rate.columns = self.currency_rate.columns.get_level_values(0)
        self.currency_rate.columns = [col[3:6] for col in self.currency_rate.columns]

        for curr in self.currency_rate:
            self.currency_rate[curr] = self.currency_rate[curr].bfill()

        self.currency_rate['USD'] = [1.] * len(self.currency_rate)
        my_curr_rate = self.currency_rate[self.currency].copy()
        for col in self.currency_rate.columns:
            self.currency_rate[col] /= my_curr_rate

        # self.currency_rate.drop(self.currency, axis=1, inplace=True)

        for curr in self.currency_rate:
            self.currency_rate[curr] = self.drop_test_data_backtest(self.currency_rate[curr])

        if self.crypto:
            self.etf_currency = pd.Series({ticker: 'USD' for ticker in self.etf_list})
        elif self.static:
            self.etf_currency = pd.Series(pd.read_csv(Data.data_dir_path / 'curr_etf.csv', index_col=0)['0'])
        else:
            def get_currency(ticker):
                """
                Helper to retrieve the trading currency for a single ticker.

                :param ticker: The ETF ticker symbol.
                :type ticker: str
                :returns: Pair ``(ticker, currency_code)``; ``'N/A'`` on failure.
                :rtype: tuple[str, str]
                """
                try:
                    return ticker, yf.Ticker(ticker).fast_info['currency']
                except Exception:
                    print('Cant retreive etf currency', ticker)
                    return ticker, 'N/A'

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(get_currency, self.etf_list)

            self.etf_currency = dict(results)
            pd.Series(self.etf_currency).to_csv(Data.data_dir_path / 'curr_etf.csv')

    def get_spy(self):
        """
        Fetch or load a benchmark proxy and convert to base currency.

        * If :attr:`static` is ``True``, read ``spy.csv`` (or
          ``spy_crypto.csv``) from :attr:`data_dir_path`.
        * Otherwise, download monthly closes for ``'VTI'`` (or ``'BTC-USD'``
          when ``crypto=True``) and cache them.

        The series is converted to the instance's base :attr:`currency` using
        :attr:`currency_rate` when that base is not USD.

        Side Effects
        ------------
        Sets :attr:`spy` and writes/reads the corresponding cache file.

        :returns: ``None``.
        :rtype: None
        """
        file_name = 'spy_crypto.csv' if self.crypto else 'spy.csv'
        spy_ticker = 'BTC-USD' if self.crypto else 'VTI'

        if self.static:
            self.spy = pd.read_csv(Data.data_dir_path / file_name, index_col=0)
            self.spy.index = pd.to_datetime(self.spy.index)
        else:
            self.spy = yf.download(spy_ticker, period=self.period, interval='1mo', auto_adjust=True)['Close']
            self.spy[spy_ticker].to_csv(Data.data_dir_path / file_name)

        if self.currency != 'USD':
            self.spy[spy_ticker] /= self.currency_rate['USD']

        self.spy = self.drop_test_data_backtest(self.spy)

    def get_rf_rate(self):
        """
        Fetch or load daily ``^IRX`` and compute a monthly risk-free rate.

        Steps
        -----
        1. Load daily close for ``^IRX`` (in decimal form) or read from cache.
        2. Resample to month starts (``'MS'``) taking the first valid value.
        3. Convert to an equivalent monthly rate via ``(1 + r) ** (1/12) - 1``.
        4. Drop timezone info and apply backtest truncation if configured.
        5. Express the series in base currency by dividing by USD FX (kept for
           compatibility with the rest of the pipeline).

        Side Effects
        ------------
        Sets :attr:`rf_rate` and writes/reads ``rf_rate.csv`` (or
        ``rf_rate_crypto.csv`` when ``crypto=True``).

        :returns: ``None``.
        :rtype: None
        """
        file_name = 'rf_rate_crypto.csv' if self.crypto else 'rf_rate.csv'

        if self.static:
            irx = pd.read_csv(Data.data_dir_path / file_name, index_col=0)
            irx.index = pd.to_datetime(irx.index, utc=True)
        else:
            irx = yf.Ticker('^IRX').history(period=self.period, interval='1d')['Close'] / 100
            irx.to_csv(Data.data_dir_path / file_name)

        rf_monthly = irx.resample('MS').first()
        self.rf_rate = (1 + rf_monthly) ** (1 / 12) - 1
        self.rf_rate.index = self.rf_rate.index.tz_localize(None)
        self.rf_rate = self.drop_test_data_backtest(self.rf_rate)

        if self.static:
            # When loaded from CSV, keep only the rate series.
            self.rf_rate = self.rf_rate['Close']

        # Express risk-free in base currency terms using USD FX.
        self.rf_rate /= self.currency_rate['USD']

    def get_nav_returns(self):
        """
        Fetch or load closes, convert to base currency, and compute returns.

        Behavior
        --------
        * If :attr:`static` is ``True``, read ``nav.csv`` (or
          ``nav_crypto.csv``) from cache.
        * Otherwise, download monthly closes for :attr:`etf_list` and cache.
        * Each ticker series is converted into the base :attr:`currency` using
          :attr:`currency_rate` and detected :attr:`etf_currency`.
        * Unless ``crypto=True``, currency columns for the non-base currencies
          are added to ``nav`` so downstream code can plot FX series alongside
          ETFs.
        * If :attr:`rates` is provided, currency return columns that match keys
          in ``rates`` are adjusted to embed those annualized rates (in %)
          as monthly equivalents.

        Side Effects
        ------------
        Sets :attr:`nav`, :attr:`returns`, :attr:`log_returns`,
        :attr:`excess_returns`.

        :returns: ``None``.
        :rtype: None
        """
        file_name = 'nav_crypto.csv' if self.crypto else 'nav.csv'

        if self.static:
            self.nav = pd.read_csv(Data.data_dir_path / file_name, index_col=0)
            self.nav.index = pd.to_datetime(self.nav.index)
        else:
            self.nav = yf.download(self.etf_list, period=self.period, interval='1mo', auto_adjust=True)['Close']
            self.nav.to_csv(Data.data_dir_path / file_name)

        for ticker in self.nav.columns:
            if ticker not in Data.possible_currencies:
                curr = self.etf_currency[ticker]
                if self.currency != curr:
                    self.nav[ticker] /= self.currency_rate[curr]
        self.nav = self.nav.copy()
        self.nav = self.drop_test_data_backtest(self.nav)

        if not self.crypto:
            for curr in self.currency_rate:
                if curr in Data.possible_currencies:
                    self.nav[curr] = self.currency_rate[curr]

        self.returns = self.nav.pct_change(fill_method=None).fillna(0)

        for curr in self.rates:
            if curr:
                self.returns[curr] = (1+self.returns[curr]) * ((1+self.rates[curr]/100) ** (1/12)) - 1

        self.log_returns = np.log(1 + self.returns)
        self.excess_returns = self.returns.subtract(self.rf_rate, axis=0)

    def plot(self, tickers):
        """
        Plot cumulative performance for selected tickers vs. benchmark and RF.

        The function plots:

        * Each ticker's cumulative total return since its first observation,
          in percent.
        * The benchmark proxy (``VTI`` or ``BTC-USD``) as a dashed line.
        * The compounded risk-free leg (from :attr:`rf_rate`) as a dashed line.

        :param tickers: Tickers/columns to plot from :attr:`nav`.
        :type tickers: list[str] | tuple[str, ...]
        :returns: ``None`` (displays a Matplotlib figure).
        :rtype: None
        :raises KeyError: If a requested ticker is not present in :attr:`nav`.
        """
        for t in tickers:
            historic = (self.nav[t] / self.nav[t].iloc[0] - 1) * 100
            plt.plot(historic, label=t)

        spy = (self.spy / self.spy.iloc[0] - 1) * 100
        plt.plot(spy, label='Total stock market', ls='--')

        rf_rate = ((self.rf_rate + 1).cumprod() - 1) * 100
        plt.plot(rf_rate, label='rate', ls='--')

        plt.axhline(0, color='black')

        plt.ylabel('%')
        plt.grid()
        plt.legend()
        plt.show()

    def get_full_names(self):
        """
        Fetch or load long names for tickers and cache them.

        * If :attr:`static` is ``True``, read from ``full_names.csv`` (or
          ``full_names_crypto.csv``).
        * Otherwise, query ``yfinance`` for ``longName`` of each ticker (or
          ``shortName`` when the symbol looks like a futures ticker with
          ``'=F'``), add passthrough entries for currency pseudo-tickers, and
          cache.

        Side Effects
        ------------
        Sets :attr:`etf_full_names`.

        :returns: ``None``.
        :rtype: None
        """
        file_name = 'full_names_crypto.csv' if self.crypto else 'full_names.csv'

        if self.static:
            etf_full_names = pd.read_csv(Data.data_dir_path / file_name, index_col=0)
        else:
            etf_full_names = pd.Series({ticker: (yf.Ticker(ticker).info['longName']) for ticker in self.etf_list if '=F' not in ticker })
            for ticker in self.etf_list:
                if '=F' in ticker:
                    etf_full_names[ticker] = yf.Ticker(ticker).info['shortName']

            for ticker in Data.possible_currencies:
                etf_full_names[ticker] = ticker
            etf_full_names.to_csv(Data.data_dir_path / file_name)

        self.etf_full_names = etf_full_names

    def get_exposure(self):
        """
        Load the exposures table from cache into :attr:`exposure`.

        This method reads ``exposure.csv`` from :attr:`data_dir_path`. The file
        is expected to exist (no download step is performed here).

        :returns: ``None``.
        :rtype: None
        """
        self.exposure = pd.read_csv(Data.data_dir_path / 'exposure.csv', index_col=0)
