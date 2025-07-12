import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


class Data:

    period = '5y'

    def __init__(self, currency, etf_list, etf_preference):

        self.sgd_rate = None
        self.nav = None
        self.rf_rate = None

        self.etf_list = etf_list
        self.currency = currency
        self.get_sgd_rate()
        self.get_rf_rate()
        self.get_nav()
        self.returns = self.nav.pct_change().iloc[1:]
        self.spy = yf.download('VOO', period=Data.period, interval='1mo', auto_adjust=True)['Close']


    def get_sgd_rate(self):

        if self.currency == 'SGD':
            self.sgd_rate = 1

        else:
            ticker = yf.Ticker(f'{self.currency}SGD=X')
            self.sgd_rate = ticker.history(period='1d', interval='1h')['Close'].iloc[-1]


    def get_rf_rate(self):

        irx = yf.download('^IRX', period=Data.period, interval='1wk', auto_adjust=True)['Close'].resample('MS').ffill()[1:]
        self.rf_rate = ((irx/100)+1) ** (1/12) - 1


    def get_nav(self):

        self.nav = yf.download(self.etf_list, period=Data.period, interval='1mo', auto_adjust=True)['Close']
        self.add_BTC()


    def add_BTC(self):

        btc = yf.download('BTC-USD', period=Data.period, interval='1mo', auto_adjust=True)['Close']['BTC-USD']

        if self.currency != 'USD':
            ticker = f'USD{self.currency}=X'
            rate = yf.download(ticker, period='5y', interval='1mo', auto_adjust=True)['Close'][ticker]
            btc *= rate

        self.nav['BTC'] = btc


    def plot(self, tickers):

        for t in tickers:
            historic = (self.nav[t]/self.nav[t].iloc[0] - 1)*100
            plt.plot(historic, label=t)

        spy = (self.spy / self.spy.iloc[0] - 1)*100
        plt.plot(spy, label='SPY', ls='--')

        rf_rate = ((self.rf_rate+1).cumprod()-1) * 100
        plt.plot(rf_rate, label='rate', ls='--')

        plt.axhline(0, color='black')


        plt.legend()
        plt.show()
