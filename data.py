import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np


class Data:

    period = '5y'

    def __init__(self, currency, etf_list):

        self.sgd_rate, self.nav, self.rf_rate, self.returns, self.excess_returns, self.log_returns = None, None, None, None, None, None

        self.etf_list = etf_list
        self.currency = currency
        self.get_sgd_rate()
        self.get_rf_rate()
        self.get_nav_returns()
        self.spy = yf.download('VOO', period=Data.period, interval='1mo', auto_adjust=True)['Close']


    def get_sgd_rate(self):

        if self.currency == 'SGD':
            self.sgd_rate = 1

        else:
            ticker = yf.Ticker(f'{self.currency}SGD=X')
            self.sgd_rate = ticker.history(period='1d', interval='1h')['Close'].iloc[-1]


    def get_rf_rate(self):

        irx = yf.Ticker("^IRX").history(period=self.period, interval="1d")['Close'] / 100
        rf_monthly = irx.resample("MS").last()
        self.rf_rate = (1 + rf_monthly) ** (1 / 12) - 1
        self.rf_rate.index = self.rf_rate.index.tz_localize(None)


    def get_nav_returns(self):

        self.nav = yf.download(self.etf_list, period=Data.period, interval='1mo', auto_adjust=True)['Close']
        for ticker in self.nav.columns:
            self.nav[ticker] *= self.sgd_rate
        self.nav = self.nav.copy()

        self.add_btc()
        self.returns = self.nav.pct_change().iloc[1:]
        self.log_returns = np.log(1+self.returns)
        self.excess_returns = self.returns.subtract(self.rf_rate, axis=0)


    def add_btc(self):

        btc = yf.download('BTC-USD', period=Data.period, interval='1mo', auto_adjust=True)['Close']['BTC-USD']

        if self.currency != 'USD':
            ticker = f'USD{self.currency}=X'
            rate = yf.download(ticker, period='5y', interval='1mo', auto_adjust=True)['Close'][ticker]
            btc *= rate

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



