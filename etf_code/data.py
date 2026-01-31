import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import concurrent.futures
from pathlib import Path
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore', category=pd.errors.Pandas4Warning, module='yfinance')


def _download_nav_chunk(etf_chunk, period):
    """Download Close prices for a chunk of tickers (used in separate process to avoid yfinance shared state)."""
    data = yf.download(etf_chunk, period=period, interval='1mo', auto_adjust=True)
    return data['Close'] if isinstance(data.columns, pd.MultiIndex) else data[['Close']].set_axis(etf_chunk, axis=1)


class Data:
    possible_currencies = ['USD', 'EUR', 'SGD', 'AUD', 'GBP', 'HKD', 'JPY']
    helper_currencies = ['INR', 'CNY', 'BRL', 'CAD']
    data_dir_path = Path(__file__).resolve().parent.parent / "data_dir"
    static_dir_path = data_dir_path / "STATIC"
    weight_cov_path = data_dir_path / "weight_cov.parquet"
    _weight_cov_params = None

    @staticmethod
    def _coerce_datetime_index(index):
        if isinstance(index, pd.DatetimeIndex):
            return index
        if pd.api.types.is_integer_dtype(index):
            sample = index[0] if len(index) else 0
            unit = 'ms' if abs(sample) < 10 ** 12 else 'ns'
            return pd.to_datetime(index, unit=unit)
        return pd.to_datetime(index)

    @staticmethod
    def _coerce_datetime_series(series):
        if pd.api.types.is_datetime64_any_dtype(series):
            return pd.to_datetime(series)
        if pd.api.types.is_integer_dtype(series):
            sample = series.dropna().iloc[0] if series.notna().any() else 0
            unit = 'ms' if abs(sample) < 10 ** 12 else 'ns'
            return pd.to_datetime(series, unit=unit)
        return pd.to_datetime(series)

    @staticmethod
    def _read_parquet_timeseries(path):
        df = pd.read_parquet(path, engine='pyarrow')

        if 'Date' in df.columns:
            df['Date'] = Data._coerce_datetime_series(df['Date'])
            df = df.set_index('Date')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = Data._coerce_datetime_index(df.index)
        return df

    @staticmethod
    def _write_parquet_timeseries(data, path, date_col='Date'):
        df = data.to_frame() if isinstance(data, pd.Series) else data
        df = df.copy().reset_index()
        if 'index' in df.columns:
            df = df.rename(columns={'index': date_col})
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
        try:
            df.to_parquet(path, index=False, engine='pyarrow')
        except Exception:
            df.to_parquet(path, index=False)

    @classmethod
    def get_weight_cov_params(cls, static=False, refit_weights=False):
        if not static and refit_weights:
            from weight_tune import save_weights
            save_weights()

        if cls._weight_cov_params is None:
            df = pd.read_parquet(cls.weight_cov_path)
            params = df.set_index("currency")[["a", "b", "c"]]
            cls._weight_cov_params = params.to_dict(orient="index")

        return cls._weight_cov_params

    def __init__(self, currency, etf_list, static=False, backtest=None, rates=None):
        self.currency_rate, self.nav, self.rf_rate, self.returns, self.excess_returns, self.log_returns, self.etf_currency, self.benchmarks, self.etf_full_names, self.exposure = None, None, None, None, None, None, None, None, None, None
        self.etf_list, self.currency, self.static, self.backtest, self.rates = etf_list, currency, static, backtest, rates
        self.period = '20y'

        self.get_currency()
        self.get_rf_rate()
        self.get_nav_returns()
        self.get_benchmarks()
        self.get_full_names()
        self.get_exposure()

    def drop_test_data_backtest(self, df):
        if self.backtest:
            df = df.loc[:self.backtest]
        return df

    @staticmethod
    def get_test_data_backtest(df, cutoff):
        return df.loc[cutoff:]

    def get_currency(self):
        currency_file = 'currency.parquet'
        if self.static:
            self.currency_rate = Data._read_parquet_timeseries(Data.data_dir_path / currency_file)
        else:
            to_download = [f'USD{ticker}=X' for ticker in Data.possible_currencies + Data.helper_currencies if
                           ticker != 'USD']
            self.currency_rate = yf.download(to_download, period=self.period, interval='1mo', auto_adjust=False)[
                'Close'].bfill()
            Data._write_parquet_timeseries(self.currency_rate, Data.data_dir_path / currency_file)

        self.currency_rate.columns = self.currency_rate.columns.get_level_values(0)
        self.currency_rate.columns = [col[3:6] for col in self.currency_rate.columns]

        for curr in self.currency_rate:
            self.currency_rate[curr] = self.currency_rate[curr].bfill()

        self.currency_rate['USD'] = [1.] * len(self.currency_rate)
        my_curr_rate = self.currency_rate[self.currency].copy()
        for col in self.currency_rate.columns:
            self.currency_rate[col] = my_curr_rate / self.currency_rate[col]

        # self.currency_rate.drop(self.currency, axis=1, inplace=True)

        for curr in self.currency_rate:
            self.currency_rate[curr] = self.drop_test_data_backtest(self.currency_rate[curr])

        if self.static:
            df = pd.read_parquet(Data.data_dir_path / 'curr_etf.parquet')
            self.etf_currency = df['currency'] if 'currency' in df.columns else df.iloc[:, 0]
        else:
            def get_currency(ticker):
                try:
                    ticker_to_test = ticker[6:] if ticker.startswith('short') else ticker
                    return ticker, yf.Ticker(ticker_to_test).fast_info['currency']
                except Exception:
                    print('Cant retreive etf currency', ticker)
                    return ticker, 'N/A'

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(get_currency, self.etf_list)

            self.etf_currency = dict(results)
            pd.Series(self.etf_currency).to_frame('currency').to_parquet(Data.data_dir_path / 'curr_etf.parquet',
                                                                         index=True)

    def get_benchmarks(self):
        file_name = 'benchmark.parquet'
        spy_ticker = 'SPY'
        bonds_ticker = 'AGG'
        gold_ticker = 'GLD'

        if self.static:
            self.benchmarks = Data._read_parquet_timeseries(Data.data_dir_path / file_name)
        else:
            tickers = [spy_ticker, bonds_ticker, gold_ticker]
            data = yf.download(tickers, period=self.period, interval='1mo', auto_adjust=True)['Close']
            self.benchmarks = pd.DataFrame({
                'SPY': data[spy_ticker],
                'AGG': data[bonds_ticker],
                'GLD': data[gold_ticker]
            })
            Data._write_parquet_timeseries(self.benchmarks, Data.data_dir_path / file_name)

        # Apply currency conversion if needed
        if self.currency != 'USD':
            for col in self.benchmarks.columns:
                self.benchmarks[col] *= self.currency_rate['USD']

        self.benchmarks = self.drop_test_data_backtest(self.benchmarks)

    def get_rf_rate(self):
        file_name = 'rf_rate.parquet'

        if self.static:
            irx = Data._read_parquet_timeseries(Data.data_dir_path / file_name)
            irx.index = pd.to_datetime(irx.index, utc=True)
        else:
            irx = yf.Ticker('^IRX').history(period=self.period, interval='1d')['Close'] / 100
            Data._write_parquet_timeseries(irx, Data.data_dir_path / file_name)

        rf_monthly = irx.resample('MS').first()
        self.rf_rate = (1 + rf_monthly) ** (1 / 12) - 1
        self.rf_rate.index = self.rf_rate.index.tz_localize(None)
        self.rf_rate = self.drop_test_data_backtest(self.rf_rate)

        if self.static:
            # When loaded from CSV, keep only the rate series.
            self.rf_rate = self.rf_rate['Close']

        # Express risk-free in base currency terms using USD FX.
        self.rf_rate *= self.currency_rate['USD']

    def get_nav_returns(self):
        file_name = 'nav.parquet'

        if self.static:
            self.nav = Data._read_parquet_timeseries(Data.data_dir_path / file_name)
        else:
            n_chunks = min(4, len(self.etf_list))
            chunks = [c.tolist() for c in np.array_split(self.etf_list, n_chunks) if len(c) > 0]

            with concurrent.futures.ProcessPoolExecutor(max_workers=n_chunks) as executor:
                results = list(executor.map(_download_nav_chunk, chunks, [self.period] * len(chunks)))

            cols_ordered = list(dict.fromkeys(self.etf_list))
            self.nav = pd.concat(results, axis=1).reindex(columns=cols_ordered).ffill()
            if len(self.nav) % 10 != 0:
                self.nav = self.nav.iloc[1:]
            Data._write_parquet_timeseries(self.nav, Data.data_dir_path / file_name)

        for ticker in self.nav.columns:
            if ticker not in Data.possible_currencies:
                curr = self.etf_currency[ticker]
                if self.currency != curr:
                    self.nav[ticker] *= self.currency_rate[curr]
        self.nav = self.nav.copy()
        self.nav = self.drop_test_data_backtest(self.nav)

        for curr in self.currency_rate:
            if curr in Data.possible_currencies:
                self.nav[curr] = self.currency_rate[curr]

        self.returns = self.nav.pct_change(fill_method=None).fillna(0)

        for curr in self.rates:
            if curr:
                self.returns[curr] = (1 + self.returns[curr]) * ((1 + self.rates[curr] / 100) ** (1 / 12)) - 1

        self.log_returns = np.log(1 + self.returns)
        self.excess_returns = self.returns.subtract(self.rf_rate, axis=0)


    def get_full_names(self):
        file_name = 'full_names.parquet'

        if self.static:
            df = pd.read_parquet(Data.data_dir_path / file_name)
            # Convert DataFrame back to Series (first column contains the values)
            etf_full_names = df.iloc[:, 0] if len(df.columns) > 0 else pd.Series(dtype=object)
        else:
            def get_name(ticker):
                key = 'shortName' if '=F' in ticker else 'longName'
                return ticker, yf.Ticker(ticker).info.get(key, '')

            with concurrent.futures.ThreadPoolExecutor() as executor:
                etf_full_names = pd.Series(dict(executor.map(get_name, self.etf_list)))
            for ticker in Data.possible_currencies:
                etf_full_names[ticker] = ticker
            etf_full_names.to_frame('name').to_parquet(Data.data_dir_path / file_name, index=True)

        self.etf_full_names = etf_full_names

    def get_exposure(self):
        self.exposure = pd.read_csv(Data.static_dir_path / 'exposure.csv', index_col=0)
