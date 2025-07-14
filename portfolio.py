from data import Data
from scipy.cluster.hierarchy import linkage, fcluster, single
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np

class Info:

    threshold_correlation = .95

    etf_list = {
        'SGD': ['VOO'],
        'EUR': ['VOO', 'SMH', 'SGOL', 'SGOV'],
        'USD': [
            "VOO", "VTI", "QQQ", "VUG", "VEA", "IEFA", "VTV", "BND",
            "AGG", "IWF", "SGOL", "IJH", "VXUS", "VIG", "IEMG", "VGT", "VWO", "VO",
            "IJR", "RSP", "XLK", "SCHD", "ITOT",  "BNDX", "IWM", "VB",
            "EFA", "IWD", "VYM", "IVW", "SCHX", "VCIT", "XLF", "QUAL", "TLT", "QQQM",
            "SCHF", "SGOV", "VT", "IXUS", "VEU", "SCHG", "IWR", "VV",
            "IWB", "JEPI", "MUB", "BSV", "MBB", "DIA", "IVE", "VTEB", "VCSH", "SPYG",
            "XLV", "IEF", "VNQ", "DFAC", "IUSB", "SCHB", "VGIT", "DGRO", "JPST", "LQD",
            "VBR", "XLE", "VONG", "GOVT", "SPDW", "SPYV", "MGK", "JEPQ", "TQQQ", "VGK",
            "USMV", "SHY", "BIV", "MDY", "USHY", "VXF", "VGSH", "SMH", "IUSG", "XLY",
            "COWZ", "EFV", "XLC", "ACWI", "XLI", "JAAA", "IGSB",  "IUSV",
            "IYW", "SDY", "DVY", "IDEV", "IWP", "XLU", "VBK", "FBND",
            "VOE", "FNDX", "OEF", "EEM", "DYNF", "SCHA", "MTUM", "VOOG", "CGDV", "VOT",
            "IEI", "XLP", "AVUV", "FNDF", "EMXC"]#['VOO', 'SMH'],

    }
    etf_list['SGD'] = etf_list['USD']
    etf_list['EUR'] = etf_list['USD']

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

    weight_cov = {
        1: 20,
        2: 10,
        3: 2
    }

    color_plot = {
        1: 'green',
        2: 'orange',
        3: 'blue'
    }

    name = {
        1: 'Low risk',
        2: 'Medium risk',
        3: 'High risk'
    }


    def __init__(self, risk, cash_sgd, holdings, currency, allow_short):

        self.risk = risk
        self.cash_sgd = cash_sgd
        self.holdings = holdings if holdings else {}
        self.allow_short = allow_short
        self.currency = currency if currency else Info.currency_config[self.risk]
        self.weight_cov = Info.weight_cov[self.risk]
        self.name = Info.name[self.risk]
        self.color_plot = f'tab:{Info.color_plot[self.risk]}'
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

        self.liquidity, self.objective = None, None

        self.n = len(self.etf_list) + 1
        self.data = Data(self.currency, self.etf_list)

        self.drop_too_new()
        self.cov_excess_returns = self.data.excess_returns.cov().values
        self.get_objective()
        self.cash = self.cash_sgd # / self.data.sgd_rate
        self.drop_highly_correlated()
        self.get_liquidity()
        self.cov_excess_returns = self.data.excess_returns.cov().values
        self.get_objective()



    def remove_etf(self, ticker):

        self.data.nav.drop(ticker, axis=1, inplace=True)
        self.data.returns.drop(ticker, axis=1, inplace=True)
        self.data.log_returns.drop(ticker, axis=1, inplace=True)
        self.data.excess_returns.drop(ticker, axis=1, inplace=True)
        self.etf_list = list(self.data.nav.columns)
        self.n -= 1


    def drop_too_new(self):

        to_drop = self.data.nav.columns[self.data.nav.isna().any()].tolist()
        for col in to_drop:
            self.remove_etf(col)


    def drop_highly_correlated(self):

        correlation_matrix = self.data.log_returns.corr().abs()
        distance_matrix = 1 - correlation_matrix
        linkage_matrix = linkage(squareform(distance_matrix), method='average')

        threshold = 1 - Portfolio.threshold_correlation
        clusters = fcluster(linkage_matrix, threshold, criterion='distance')

        cluster_df = pd.DataFrame({'ETF': self.etf_list, 'Cluster': clusters})

        obj_values = {ticker: self.objective(single_ticker=ticker) for ticker in self.etf_list}
        obj_values = pd.Series(obj_values, name='obj_values')

        cluster_df = cluster_df.set_index('ETF').join(obj_values)


        (cluster_df.sort_values(by='Cluster')).to_csv('/Users/maximesolere/desktop/cluster.csv')
        to_plot = cluster_df[cluster_df['Cluster'].isin([57, ])].index.to_list()
        self.data.plot(to_plot)



        best_etfs = cluster_df.groupby('Cluster')['obj_values'].idxmin().tolist()

        to_drop = [ticker for ticker in self.etf_list if ticker not in best_etfs]
        for ticker in to_drop:
            self.remove_etf(ticker)


    def get_liquidity(self):

        self.liquidity = self.cash + sum(self.holdings.values())


    def get_objective(self,):

        def f(w=np.zeros(self.n), single_ticker=None):

            if single_ticker:
                excess_series = self.data.excess_returns[single_ticker]
                mean = excess_series.mean()
                var = excess_series.var()
                return self.weight_cov * var - mean

            excess_series = self.data.excess_returns @ w
            mean = excess_series.mean()
            return self.weight_cov * (w @ self.cov_excess_returns @ w) - mean

        self.objective = f



#x = Portfolio(risk=1, currency='USD')