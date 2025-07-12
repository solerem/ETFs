from data import Data


class Info:

    threshold_correlation = .9

    etf_list = {
        'SGD': ['VOO'],
        'EUR': ['VOO'],
        'USD': ['VOO', 'IVV'],
    }

    etf_preference = {
        'SGD': {},
        'EUR': {},
        'USD': {}
    }

    currency_config = {
        1: 'SGD',
        2: 'EUR',
        3: 'USD'
    }


    def __init__(self, risk, cash_sgd, holdings, currency, allow_short):

        self.risk = risk
        self.cash_sgd = cash_sgd
        self.holdings = holdings if holdings else {}
        self.allow_short = allow_short
        self.currency = currency if currency else Info.currency_config[self.risk]
        self.etf_list = Info.etf_list[self.currency]
        self.etf_preference = Info.etf_preference[self.currency]
        self.transform_etf_preference()


    def transform_etf_preference(self):

        temp = {}
        for preferred in self.etf_preference:
            temp[preferred] = True
            temp[self.etf_preference[preferred]] = False

        self.etf_preference = temp




class Portfolio(Info):


    def __init__(self, risk=3, cash_sgd=100, holdings=None, currency=None, allow_short=False):

        super().__init__(risk, cash_sgd, holdings, currency, allow_short)

        self.data = Data(self.currency, self.etf_list)
        self.cash = self.cash_sgd / self.data.sgd_rate
        self.drop_highly_correlated()


    def remove_etf(self, ticker):

        self.etf_list.remove(ticker)
        self.data.nav.drop(ticker, axis=1, inplace=True)
        self.data.returns.drop(ticker, axis=1, inplace=True)


    def drop_highly_correlated(self):

        while True:
            detected = False
            corr_matrix = self.data.returns.corr()
            for i, col in enumerate(self.etf_list):
                correlated = corr_matrix[col][self.etf_list[i+1:]]
                high_corr = correlated[correlated > Info.threshold_correlation]
                if not high_corr.empty:
                    detected = True
                    if col in self.etf_preference and self.etf_preference[col]:
                        for ticker in high_corr:
                            if ticker in self.etf_preference and not self.etf_preference[ticker]:
                                self.remove_etf(ticker)
                                break
                    else:
                        self.remove_etf(col)
                    break
            if not detected:
                break




#Portfolio()







