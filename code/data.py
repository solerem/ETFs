from locale import currency

import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import concurrent.futures
from scipy.optimize import minimize

class Data:

    period = '20y'
    possible_currencies = ['USD', 'EUR', 'SGD', 'GBP', 'JPY', 'CHF', 'CNY', 'HKD']
    data_dir_path = '/Users/maximesolere/PycharmProjects/ETF/data_dir/'

    def __init__(self, currency, etf_list, static=False, backtest=None):

        self.currency_rate, self.nav, self.rf_rate, self.returns, self.excess_returns, self.log_returns, self.etf_currency, self.spy, self.etf_full_names, self.exposure, self.crypto_opti = None, None, None, None, None, None, None, None, None, None, None
        self.etf_list, self.currency, self.static, self.backtest = etf_list, currency, static, backtest

        self.get_currency()
        self.get_rf_rate()
        self.get_nav_returns()
        self.get_spy()
        self.get_full_names()
        self.get_exposure()
        self.get_crypto()


    def drop_test_data_backtest(self, df):

        if self.backtest:
            #df = df.loc[self.backtest - pd.DateOffset(months=180): self.backtest]
            df = df.loc[:self.backtest]
        return df


    @staticmethod
    def get_test_data_backtest(df, cutoff):
        return df.loc[cutoff:]


    def get_currency(self):

        if self.static:
            self.currency_rate = pd.read_csv(Data.data_dir_path + f'currency.csv', index_col=0)
            self.currency_rate.index = pd.to_datetime(self.currency_rate.index)
        else:
            to_download = [f'USD{ticker}=X' for ticker in Data.possible_currencies if ticker != 'USD']
            self.currency_rate = yf.download(to_download, period=Data.period, interval='1mo', auto_adjust=True)['Close']
            self.currency_rate.to_csv(Data.data_dir_path + f'currency.csv')


        self.currency_rate.columns = self.currency_rate.columns.get_level_values(0)
        self.currency_rate.columns = [col[3:6] for col in self.currency_rate.columns]

        for curr in self.currency_rate:
            self.currency_rate[curr] = self.currency_rate[curr].bfill()

        self.currency_rate['USD'] = [1.] * len(self.currency_rate)
        my_curr_rate = self.currency_rate[self.currency].copy()
        for col in self.currency_rate.columns:
            self.currency_rate[col] /= my_curr_rate

        self.currency_rate.drop(self.currency, axis=1, inplace=True)

        for curr in self.currency_rate:
            self.currency_rate[curr] = self.drop_test_data_backtest(self.currency_rate[curr])

        if self.static:
            self.etf_currency = pd.Series(pd.read_csv(Data.data_dir_path+'curr_etf.csv', index_col=0)['0'])

        else:

            def get_currency(ticker):

                try:
                    return ticker, yf.Ticker(ticker).fast_info['currency']
                except Exception:
                    print('Cant retreive etf currency', ticker)
                    return ticker, 'N/A'

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(get_currency, self.etf_list)

            self.etf_currency = dict(results)
            pd.Series(self.etf_currency).to_csv(Data.data_dir_path+'curr_etf.csv')


    def get_spy(self):

        if self.static:
            self.spy = pd.read_csv(Data.data_dir_path+'spy.csv', index_col=0)
            self.spy.index = pd.to_datetime(self.spy.index)
        else:
            self.spy = yf.download('VTI', period=Data.period, interval='1mo', auto_adjust=True)['Close']
            self.spy['VTI'].to_csv(Data.data_dir_path+'spy.csv')

        if self.currency != 'USD':
            self.spy['VTI'] /= self.currency_rate['USD']

        self.spy = self.drop_test_data_backtest(self.spy)


    def get_rf_rate(self):

        if self.static:
            irx = pd.read_csv(Data.data_dir_path + 'rf_rate.csv', index_col=0)
            irx.index = pd.to_datetime(irx.index, utc=True)
        else:
            irx = yf.Ticker("^IRX").history(period=self.period, interval="1d")['Close'] / 100
            irx.to_csv(Data.data_dir_path + 'rf_rate.csv')

        rf_monthly = irx.resample("MS").first()
        self.rf_rate = (1 + rf_monthly) ** (1 / 12) - 1
        self.rf_rate.index = self.rf_rate.index.tz_localize(None)
        self.rf_rate = self.drop_test_data_backtest(self.rf_rate)

        if self.static:
            self.rf_rate = self.rf_rate['Close']


    def get_crypto(self):
        rf = 0.

        if self.static:
            nav = pd.read_csv(Data.data_dir_path+'crypto.csv', index_col=0)
            nav.index = pd.to_datetime(nav.index)
        else:
            nav = yf.download([f'{t}-USD' for t in ['BTC', 'ETH', 'XRP', 'BNB', 'ADA', 'DOGE', 'TRX']], period='max',
                             interval='1mo')['Close'].loc['2017-11-01 00:00:00':]
            nav.to_csv(Data.data_dir_path+'crypto.csv')


        if self.currency != 'USD':
            for col in nav:
                nav[col] /= self.currency_rate['USD']


        rets_m = nav.pct_change().dropna(how='all')  # simple monthly returns
        rets_m = rets_m.dropna(axis=1)  # drop assets with all-NaN returns
        mu = rets_m.mean() * 12  # annualized mean returns
        Sigma = rets_m.cov() * 12  # annualized covariance
        assets = mu.index.to_list()
        n = len(assets)

        # --- Helpers ------------------------------------------------------------
        def portfolio_stats(w, mu, Sigma, rf):
            mu_p = float(np.dot(w, mu))
            var_p = float(np.dot(w, Sigma @ w))
            vol_p = np.sqrt(var_p)
            sharpe = (mu_p - rf) / vol_p if vol_p > 0 else -np.inf
            return mu_p, vol_p, sharpe

        # Objective: negative Sharpe (for minimization)
        def neg_sharpe(w, mu, Sigma, rf):
            _, vol, _ = portfolio_stats(w, mu, Sigma, rf)
            if vol <= 0:
                return 1e6
            return - (np.dot(w, mu) - rf) / vol

        # --- A) Long-only max-Sharpe via SLSQP ----------------------------------
        w0 = np.repeat(1.0 / n, n)
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # fully invested
        )
        bounds = tuple((0.0, 1.0) for _ in range(n))  # long-only

        res = minimize(
            fun=neg_sharpe,
            x0=w0,
            args=(mu.values, Sigma.values, rf),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 10_000, "ftol": 1e-12, "disp": False},
        )

        w_star_long = pd.Series(res.x, index=assets).sort_values(ascending=False)
        self.crypto_opti = {x[:3]: float(round(w_star_long[x]*100,1)) for x in w_star_long.index}


    def get_nav_returns(self):

        if self.static:
            self.nav = pd.read_csv(Data.data_dir_path + 'nav.csv', index_col=0)
            self.nav.index = pd.to_datetime(self.nav.index)
        else:
            self.nav = yf.download(self.etf_list, period=Data.period, interval='1mo', auto_adjust=True)['Close']
            self.nav.to_csv(Data.data_dir_path + 'nav.csv')

        for ticker in self.nav.columns:
            curr = self.etf_currency[ticker]
            if self.currency != curr:
                self.nav[ticker] /= self.currency_rate[curr]
        self.nav = self.nav.copy()
        self.nav = self.drop_test_data_backtest(self.nav)

        for curr in self.currency_rate:
            self.nav[curr] = self.currency_rate[curr]

        self.returns = self.nav.pct_change().fillna(0)
        self.log_returns = np.log(1+self.returns)
        self.excess_returns = self.returns.subtract(self.rf_rate, axis=0)


    def plot(self, tickers):

        for t in tickers:
            historic = (self.nav[t]/self.nav[t].iloc[0] - 1)*100
            plt.plot(historic, label=t)

        spy = (self.spy / self.spy.iloc[0] - 1)*100
        plt.plot(spy, label='Total stock market', ls='--')

        rf_rate = ((self.rf_rate+1).cumprod()-1) * 100
        plt.plot(rf_rate, label='rate', ls='--')

        plt.axhline(0, color='black')

        plt.ylabel('%')
        plt.grid()
        plt.legend()
        plt.show()


    def get_full_names(self):

        if self.static:
            etf_full_names = pd.read_csv(Data.data_dir_path + 'full_names.csv', index_col=0)
        else:
            etf_full_names = pd.Series({ticker: (yf.Ticker(ticker).info['longName'] ) for ticker in self.etf_list})
            for ticker in Data.possible_currencies:
                etf_full_names[ticker] = ticker
            etf_full_names.to_csv(Data.data_dir_path + 'full_names.csv')

        self.etf_full_names = etf_full_names


    def get_exposure(self):

        self.exposure = pd.read_csv(Data.data_dir_path+'exposure.csv', index_col=0)


