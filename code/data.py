from locale import currency

import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import concurrent.futures
from sklearn.linear_model import LinearRegression




class Data:

    period = '20y'
    possible_currencies = ['USD', 'EUR', 'SGD']
    data_dir_path = '/Users/maximesolere/PycharmProjects/ETF/data_dir/'

    def __init__(self, currency, etf_list, static=False, backtest=None):

        self.currency_rate, self.nav, self.rf_rate, self.returns, self.excess_returns, self.log_returns, self.etf_currency, self.spy, self.etf_full_names, self.exposure, self.btc = None, None, None, None, None, None, None, None, None, None, None
        self.etf_list, self.currency, self.static, self.backtest = etf_list, currency, static, backtest

        self.get_currency()
        self.get_rf_rate()
        self.get_nav_returns()
        self.get_spy()
        self.get_full_names()
        self.get_exposure()


    def drop_test_data_backtest(self, df):

        if self.backtest:
            #df = df.loc[self.backtest - pd.DateOffset(months=180): self.backtest]
            df = df.loc[:self.backtest]
        return df


    @staticmethod
    def get_test_data_backtest(df, cutoff):
        return df.loc[cutoff:]


    def get_currency(self):

        to_download = [f'{self.currency}{ticker}=X' for ticker in Data.possible_currencies if ticker != self.currency]

        if self.static:
            self.currency_rate = pd.read_csv(Data.data_dir_path + f'currency_{self.currency}.csv', index_col=0)
            self.currency_rate.index = pd.to_datetime(self.currency_rate.index)
        else:
            self.currency_rate = yf.download(to_download, period=Data.period, interval='1mo', auto_adjust=True)['Close']
            self.currency_rate.to_csv(Data.data_dir_path + f'currency_{self.currency}.csv')

        self.currency_rate.columns = self.currency_rate.columns.get_level_values(0)
        self.currency_rate.columns = [col[3:6] for col in self.currency_rate.columns]

        if self.currency == 'SGD':
            self.currency_rate['EUR'] = self.currency_rate['EUR'].bfill()

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

        self.add_btc()

        self.nav = self.drop_test_data_backtest(self.nav)

        self.returns = self.nav.pct_change().fillna(0)
        self.log_returns = np.log(1+self.returns)
        self.excess_returns = self.returns.subtract(self.rf_rate, axis=0)



    def backfill_btc(self):

        def to_logret(p):
            return np.log(p).diff().dropna()

        etf_r = to_logret(self.nav.copy())
        btc_r = to_logret(self.btc['BTC-USD'].copy())

        df = etf_r.join(btc_r.rename("BTC"), how="outer")

        X, y = df[etf_r.columns], df["BTC"]
        cutoff = self.btc.index[0]
        mask_train = X.index > cutoff

        model = LinearRegression()
        model.fit(X[mask_train], y[mask_train])

        # Predicted BTC returns for pre-2011 period
        mask_back = X.index <= cutoff

        btc_backfill = pd.Series(model.predict(X[mask_back]), index=X[mask_back].index, name="BTC_synthetic")
        scale = y[mask_train].std() / btc_backfill.std()
        btc_backfill *= scale

        new_row = pd.Series({pd.Timestamp("2005-09-01"): btc_backfill.iloc[0]})
        btc_backfill = pd.concat([new_row, btc_backfill]).sort_index()

        # Combine synthetic pre-2011 with real post-2011
        self.btc = pd.concat([btc_backfill, y[mask_train]]).sort_index()

    def add_btc(self):


        if self.static:
            self.btc = pd.read_csv(Data.data_dir_path+'btc.csv', index_col=0)['0']
            self.btc.index = pd.to_datetime(self.btc.index)
        else:
            self.btc = yf.download('BTC-USD', period=Data.period, interval='1mo', auto_adjust=True)['Close']#['BTC-USD']
            self.backfill_btc()
            self.btc.to_csv(Data.data_dir_path+'btc.csv')


        if self.currency != 'USD':
            self.btc /= self.currency_rate['USD']


        #self.nav['BTC'] = self.btc
        #self.etf_list.append('BTC')


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
            etf_full_names = pd.Series({ticker: yf.Ticker(ticker).info['longName'] for ticker in self.etf_list})
            etf_full_names.to_csv(Data.data_dir_path + 'full_names.csv')

        self.etf_full_names = etf_full_names


    def get_exposure(self):

        self.exposure = pd.read_csv(Data.data_dir_path+'exposure.csv', index_col=0)



