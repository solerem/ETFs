from data import Data


class Info:


    etf_list = {
        'SGD': ['VOO'],
        'EUR': ['VOO'],
        'USD': ['VOO'],
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

        self.data = Data(self.currency, self.etf_list, self.etf_preference)
        self.cash = self.cash_sgd / self.data.sgd_rate





Portfolio()







