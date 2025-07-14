import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Data:

    period = '5y'
    possible_currencies = ['USD', 'EUR', 'HKD', 'JPY', 'GBP', 'CHF', 'AUD', 'CNY', 'SEK']

    def __init__(self, currency, etf_list):

        self.currency_rate, self.nav, self.rf_rate, self.returns, self.excess_returns, self.log_returns, self.etf_currency = None, None, None, None, None, None, None

        self.etf_list = etf_list
        self.currency = currency
        self.get_currency()
        self.get_rf_rate()
        self.get_nav_returns()
        self.spy = yf.download('VOO', period=Data.period, interval='1mo', auto_adjust=True)['Close']


    def get_currency(self):

        to_download = [f'{self.currency}{ticker}=X' for ticker in Data.possible_currencies if ticker != self.currency]
        self.currency_rate = yf.download(to_download, period=Data.period, interval='1mo', auto_adjust=True)['Close']
        self.currency_rate.columns = [col[3:6] for col in self.currency_rate.columns]
        self.currency_rate[self.currency] = pd.Series([1]*len(self.currency_rate))

        self.etf_currency = {}

        for ticker in self.etf_list:
            y_ticker = yf.Ticker(ticker)
            currency = y_ticker.info.get('currency', 'N/A')
            self.etf_currency[ticker] = currency


    def get_rf_rate(self):

        irx = yf.Ticker("^IRX").history(period=self.period, interval="1d")['Close'] / 100
        rf_monthly = irx.resample("MS").first()
        self.rf_rate = (1 + rf_monthly) ** (1 / 12) - 1
        self.rf_rate.index = self.rf_rate.index.tz_localize(None)


    def get_nav_returns(self):

        self.nav = yf.download(self.etf_list, period=Data.period, interval='1mo', auto_adjust=True)['Close']
        for ticker in self.nav.columns:
            self.nav[ticker] *= self.currency_rate[self.etf_currency[ticker]]
        self.nav = self.nav.copy()

        self.add_btc()
        self.returns = self.nav.pct_change().iloc[1:]
        self.log_returns = np.log(1+self.returns)
        self.excess_returns = self.returns.subtract(self.rf_rate, axis=0)


    def add_btc(self):

        btc = yf.download('BTC-USD', period=Data.period, interval='1mo', auto_adjust=True)['Close']['BTC-USD']

        if self.currency != 'USD':
            btc *= self.currency_rate['USD']

        self.nav = self.nav.copy()
        self.nav['BTC'] = btc
        self.etf_list.append('BTC')


    def plot(self, tickers):

        for t in tickers:
            historic = (self.nav[t]/self.nav[t].iloc[0] - 1)*100
            plt.plot(historic, label=t)

        spy = (self.spy / self.spy.iloc[0] - 1)*100
        plt.plot(spy, label='SPY', ls='--')

        rf_rate = ((self.rf_rate+1).cumprod()-1) * 100
        plt.plot(rf_rate, label='rate', ls='--')

        plt.axhline(0, color='black')

        plt.ylabel('%')
        plt.grid()
        plt.legend()
        plt.show()



