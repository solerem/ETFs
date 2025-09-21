"""
Data acquisition and preprocessing utilities for multi-asset portfolio work.

This module centers around :class:`Data`, which downloads (or loads cached)
time series for:

* Foreign exchange (FX) rates to convert assets into a chosen base currency.
* A risk-free rate proxy from ``^IRX`` (13-week T-bill), converted to a
  monthly rate.
* ETF NAV/Close series and derived simple, log, and excess returns.
* A total U.S. equity market proxy (VTI) for benchmarking.
* A simple long-only tangency portfolio over a small crypto universe.

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
    Container for market datasets (FX, risk-free, ETFs, crypto) with helpers to
    compute returns, perform currency conversion, and plot series.

    Parameters
    ----------
    currency : str
        Target/base currency code (one of :attr:`possible_currencies`) used to
        express all prices and returns.
    etf_list : list[str]
        List of ETF tickers understood by *yfinance*.
    static : bool, optional
        If ``True``, read from and write to cached CSVs under
        :attr:`data_dir_path` instead of fetching from the network (default:
        ``False``).
    backtest : pandas.Timestamp | str | None, optional
        If provided, time series loaded by this instance are truncated to
        ``.loc[:backtest]`` to form an in-sample set for backtesting. Use
        :meth:`get_test_data_backtest` to obtain the complementary out-of-sample
        slice.

    Attributes
    ----------
    currency_rate : pandas.DataFrame | None
        Monthly FX rates normalized to the base currency. Columns are other
        currencies; values express the conversion **into** the base currency.
        Populated by :meth:`get_currency`.
    etf_currency : dict[str, str] | pandas.Series | None
        Mapping from ETF ticker to its trading currency. Populated by
        :meth:`get_currency`.
    nav : pandas.DataFrame | None
        Monthly close/NAV series for requested ETFs, converted into the base
        currency. Populated by :meth:`get_nav_returns`.
    returns : pandas.DataFrame | None
        Simple monthly returns from :attr:`nav`. Populated by
        :meth:`get_nav_returns`.
    log_returns : pandas.DataFrame | None
        Log monthly returns from :attr:`nav`. Populated by
        :meth:`get_nav_returns`.
    rf_rate : pandas.Series | None
        Monthly risk-free rates aligned to month starts. Populated by
        :meth:`get_rf_rate`.
    excess_returns : pandas.DataFrame | None
        Simple returns in excess of :attr:`rf_rate` (broadcast across columns).
        Populated by :meth:`get_nav_returns`.
    spy : pandas.Series | pandas.DataFrame | None
        Total U.S. market proxy (VTI) in the base currency. Populated by
        :meth:`get_spy`.
    etf_full_names : pandas.Series | None
        Long names for tickers (and passthrough for currency columns). Populated
        by :meth:`get_full_names`.
    exposure : pandas.DataFrame | None
        Exposures table loaded from cache. Populated by :meth:`get_exposure`.
    crypto_opti : dict[str, float] | None
        Long-only tangency portfolio weights (percent, rounded to 0.1) for the
        crypto universe. Populated by :meth:`get_crypto`.
    """


    possible_currencies = ['USD', 'EUR', 'SGD', 'GBP', 'JPY', 'CNH', 'HKD', 'AUD', 'CAD', 'NOK', 'NZD', 'SEK', 'THB']
    data_dir_path = Path(__file__).resolve().parent.parent / "data_dir"

    def __init__(self, currency, etf_list, static=False, backtest=None, rates=None, crypto=False):
        """
        Initialize and eagerly load core datasets.

        This constructor immediately calls, in order:
        :meth:`get_currency`, :meth:`get_rf_rate`, :meth:`get_nav_returns`,
        :meth:`get_spy`, :meth:`get_full_names`, :meth:`get_exposure`,
        :meth:`get_crypto`.

        :param currency: Base currency code.
        :type currency: str
        :param etf_list: ETF tickers to download.
        :type etf_list: list[str]
        :param static: If ``True``, use cached CSVs; otherwise fetch and cache.
        :type static: bool
        :param backtest: If set, truncate series to ``.loc[:backtest]``.
        :type backtest: pandas.Timestamp | str | None
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

        df = pd.read_csv(Data.data_dir_path / 'alternatives.csv', sep=';')
        self.alternatives = {row['TICKER']: row['BEST'] for _, row in df.iterrows()}

    @staticmethod
    def get_test_data_backtest(df, cutoff):
        """
        Obtain the out-of-sample (test) slice for backtesting.

        This is a convenience wrapper equivalent to ``df.loc[cutoff:]``.

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
        * If :attr:`static` is ``True``, read ``currency.csv`` and
          ``curr_etf.csv`` from :attr:`data_dir_path`.
        * Otherwise, download monthly FX rates for all currencies listed in
          :attr:`possible_currencies` (vs. USD), normalize everything into the
          base :attr:`currency`, and cache to CSV. ETF trading currencies are
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
            to_download = [f'USD{ticker}=X' for ticker in Data.possible_currencies if ticker != 'USD']
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

        #self.currency_rate.drop(self.currency, axis=1, inplace=True)

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
        Fetch or load a total market proxy (VTI) and convert to base currency.

        * If :attr:`static` is ``True``, read ``spy.csv`` (VTI close series)
          from :attr:`data_dir_path`.
        * Otherwise, download monthly close for ``'VTI'`` and cache it.

        The series is converted to the instance's base :attr:`currency` using
        :attr:`currency_rate` when that base is not USD.

        Side Effects
        ------------
        Sets :attr:`spy` and writes/reads ``spy.csv``.

        :returns: ``None``.
        :rtype: None
        """
        file_name = 'spy_crypto.csv' if self.crypto else 'spy.csv'

        if self.static:
            self.spy = pd.read_csv(Data.data_dir_path / file_name, index_col=0)
            self.spy.index = pd.to_datetime(self.spy.index)
        else:
            self.spy = yf.download('VTI', period=self.period, interval='1mo', auto_adjust=True)['Close']
            self.spy['VTI'].to_csv(Data.data_dir_path / file_name)

        if self.currency != 'USD':
            self.spy['VTI'] /= self.currency_rate['USD']

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

        Side Effects
        ------------
        Sets :attr:`rf_rate` and writes/reads ``rf_rate.csv``.

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
            self.rf_rate = self.rf_rate['Close']

        self.rf_rate /= self.currency_rate['USD']





    def get_nav_returns(self):
        """
        Fetch or load ETF closes, convert to base currency, and compute returns.

        Behavior
        --------
        * If :attr:`static` is ``True``, read ``nav.csv`` from cache.
        * Otherwise, download monthly closes for :attr:`etf_list` and cache.
        * Each ETF series is converted into the base :attr:`currency` using
          :attr:`currency_rate` and detected :attr:`etf_currency`.
        * Currency columns for the non-base currencies are added to ``nav`` so
          downstream code can plot FX series alongside ETFs.

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
        * The total stock market proxy (VTI) as a dashed line.
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
        Fetch or load long names for ETFs and cache them.

        * If :attr:`static` is ``True``, read from ``full_names.csv``.
        * Otherwise, query ``yfinance.Ticker(t).info['longName']`` for each ETF,
          add passthrough entries for currency pseudo-tickers, and cache.

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
