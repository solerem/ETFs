import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import concurrent.futures
from pathlib import Path
from scipy.optimize import minimize


class Data:
    possible_currencies = ['USD', 'EUR', 'SGD', 'AUD', 'GBP', 'HKD', 'JPY']
    helper_currencies = ['INR', 'CNY', 'BRL', 'CAD']
    data_dir_path = Path(__file__).resolve().parent.parent / "data_dir"
    static_dir_path = data_dir_path / "STATIC"

    @staticmethod
    def _coerce_datetime_index(index):
        if isinstance(index, pd.DatetimeIndex):
            return index
        if pd.api.types.is_integer_dtype(index):
            sample = index[0] if len(index) else 0
            unit = 'ms' if abs(sample) < 10**12 else 'ns'
            return pd.to_datetime(index, unit=unit)
        return pd.to_datetime(index)

    @staticmethod
    def _coerce_datetime_series(series):
        if pd.api.types.is_datetime64_any_dtype(series):
            return pd.to_datetime(series)
        if pd.api.types.is_integer_dtype(series):
            sample = series.dropna().iloc[0] if series.notna().any() else 0
            unit = 'ms' if abs(sample) < 10**12 else 'ns'
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
        df = df.reset_index()
        if 'index' in df.columns:
            df = df.rename(columns={'index': date_col})
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
        try:
            df.to_parquet(path, index=False, engine='pyarrow')
        except Exception:
            df.to_parquet(path, index=False)

    def __init__(self, currency, etf_list, static=False, backtest=None, rates=None, crypto=False):
        self.currency_rate, self.nav, self.rf_rate, self.returns, self.excess_returns, self.log_returns, self.etf_currency, self.spy, self.etf_full_names, self.exposure = None, None, None, None, None, None, None, None, None, None
        self.etf_list, self.currency, self.static, self.backtest, self.rates = etf_list, currency, static, backtest, rates
        self.crypto = crypto
        self.period = '5y' if self.crypto else '20y'

        self.get_currency()
        self.get_rf_rate()
        self.get_nav_returns()
        self.get_spy()
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
        currency_file = 'currency_crypto.parquet' if self.crypto else 'currency.parquet'
        if self.static:
            self.currency_rate = Data._read_parquet_timeseries(Data.data_dir_path / currency_file)
        else:
            to_download = [f'USD{ticker}=X' for ticker in Data.possible_currencies+Data.helper_currencies if ticker != 'USD']
            self.currency_rate = yf.download(to_download, period=self.period, interval='1mo', auto_adjust=False)['Close'].bfill()
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

        if self.crypto:
            self.etf_currency = pd.Series({ticker: 'USD' for ticker in self.etf_list})
        elif self.static:
            self.etf_currency = pd.Series(pd.read_csv(Data.static_dir_path / 'curr_etf.csv', index_col=0)['0'])
        else:
            def get_currency(ticker):
                try:
                    ticker_to_test = ticker[3:] if ticker.startswith('-- ') else ticker
                    return ticker, yf.Ticker(ticker_to_test).fast_info['currency']
                except Exception:
                    print('Cant retreive etf currency', ticker)
                    return ticker, 'N/A'

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(get_currency, self.etf_list)

            self.etf_currency = dict(results)
            pd.Series(self.etf_currency).to_csv(Data.static_dir_path / 'curr_etf.csv')

    def get_spy(self):
        file_name = 'spy_crypto.parquet' if self.crypto else 'spy.parquet'
        spy_ticker = 'BTC-USD' if self.crypto else 'VTI'

        if self.static:
            self.spy = Data._read_parquet_timeseries(Data.data_dir_path / file_name)
        else:
            self.spy = yf.download(spy_ticker, period=self.period, interval='1mo', auto_adjust=True)['Close']
            Data._write_parquet_timeseries(self.spy[spy_ticker], Data.data_dir_path / file_name)

        if self.currency != 'USD':
            self.spy[spy_ticker] *= self.currency_rate['USD']


        self.spy = self.drop_test_data_backtest(self.spy)

    def get_rf_rate(self):
        file_name = 'rf_rate_crypto.parquet' if self.crypto else 'rf_rate.parquet'

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
        file_name = 'nav_crypto.parquet' if self.crypto else 'nav.parquet'

        if self.static:
            self.nav = Data._read_parquet_timeseries(Data.data_dir_path / file_name)
        else:
            self.nav = yf.download(self.etf_list, period=self.period, interval='1mo', auto_adjust=True)['Close'].ffill()
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
        file_name = 'full_names_crypto.csv' if self.crypto else 'full_names.csv'

        if self.static:
            etf_full_names = pd.read_csv(Data.static_dir_path / file_name, index_col=0)
        else:
            etf_full_names = pd.Series({ticker: (yf.Ticker(ticker).info['longName']) for ticker in self.etf_list if '=F' not in ticker })
            for ticker in self.etf_list:
                if '=F' in ticker:
                    etf_full_names[ticker] = yf.Ticker(ticker).info['shortName']

            for ticker in Data.possible_currencies:
                etf_full_names[ticker] = ticker
            etf_full_names.to_csv(Data.static_dir_path / file_name)

        self.etf_full_names = etf_full_names

    def get_exposure(self):
        self.exposure = pd.read_csv(Data.static_dir_path / 'exposure.csv', index_col=0)
