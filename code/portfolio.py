from data import Data
from scipy.cluster.hierarchy import linkage, fcluster, single
from scipy.spatial.distance import squareform
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

class Info:

    threshold_correlation = .95

    etf_list = [
        'SPY', 'QQQ', 'DIA', 'MDY', 'IWM', 'XLY', 'XLP', 'XLE', 'XLV', 'XLF', 'XLI', 'XLB', 'XLK', 'XLU', 'EFA', 'EEM',
        'EWA', 'EWC', 'EWH', 'EWG', 'EWI', 'EWJ', 'EWU', 'EWM', 'EWS', 'EWP', 'EWD', 'LQD', 'TLT', 'TIP', 'GLD', 'VTI',
        'IWN', 'IUSG', 'IYJ', 'EWL', 'VHT', 'IWB', 'XLU', 'IGE', 'RTH', 'VWO', 'IWV', 'EWW', 'EWC', 'EWN', 'VPU', 'PWB',
        'VIS', 'IYM', 'SPYV', 'SLYV', 'IUSV', 'AGG', 'IWF', 'EWZ', 'LQD', 'ILCB', 'IXN', 'VDE', 'VOX', 'XLG', 'IVW',
        'IJK', 'XLP', 'XSMO', 'IXC', 'EWY', 'IGM', 'IJH', 'PEJ', 'IVV', 'IYY', 'SOXX', 'EWP', 'VPL', 'IYH', 'VTV',
        'EWT', 'IYW', 'IMCG', 'EWH', 'IGPT', 'PJP', 'SPYG', 'ITOT', 'FXI', 'EWI', 'XLE', 'XLY', 'EWA', 'ILCG', 'IMCV',
        'XLI', 'IWM', 'DVY', 'VBK', 'EWG', 'IGV', 'IJS', 'XNTK', 'IYT', 'SPTM', 'PEY', 'VBR', 'EEM', 'PWV', 'TLT',
        'VFH', 'IEV', 'VB', 'SPEU', 'VGK', 'IYG', 'IWP', 'VTI', 'FEZ', 'EZU', 'IWR', 'VV', 'XLB', 'EWU', 'IJJ', 'IJR',
        'EFA', 'EPP', 'IEF', 'VDC', 'IBB', 'PBW', 'TIP', 'IWS', 'IYE', 'IWO', 'VUG', 'SUSA', 'ILCV', 'IYK', 'XMMO',
        'XLV', 'ONEQ', 'SHY', 'ISCB', 'EWJ', 'VXF', 'EWQ', 'PSI', 'ILF', 'IYR', 'IXG', 'IWD', 'IXP', 'VO', 'IDU', 'VGT',
        'EWD', 'IYZ', 'ISCV', 'ICF', 'IOO', 'SLYG', 'VCR', 'EWS', 'EZA', 'IVE', 'XLF', 'IMCB', 'IYF', 'VAW', 'OEF',
        'IJT', 'RWR', 'IXJ', 'SMH', 'IYC', 'ISCG', 'VNQ', 'XMVM', 'RSP', 'DGT', 'XLK'
    ]

    etf_list = sorted(list(set(etf_list)))

    name = {
        1: 'Low risk',
        2: 'Medium risk',
        3: 'High risk'
    }

    def __init__(self, risk, cash, holdings, currency, allow_short):

        self.color_map, self.weight_cov = None, None
        self.risk = risk
        self.cash = cash
        self.holdings = holdings if holdings else {}
        self.allow_short = allow_short
        self.currency = currency if currency else 'USD'
        self.get_weight_cov()
        self.name = 'Risk ' + str(self.risk + 4)
        self.etf_list = Info.etf_list
        self.n = len(self.etf_list)
        self.get_color_map()


    def get_weight_cov(self):

        self.weight_cov = 52*np.exp(-0.3259*self.risk)-2


    def get_color_map(self):
        cmap = cm.get_cmap('tab20', self.n)
        self.color_map = {asset: mcolors.to_hex(cmap(i)) for i, asset in enumerate(self.etf_list+[ticker for ticker in Data.possible_currencies if ticker != self.currency])}


class Portfolio(Info):


    def __init__(self, risk=3, cash=100, holdings=None, currency=None, allow_short=False, static=False, backtest=None):

        super().__init__(risk, cash, holdings, currency, allow_short)

        self.liquidity, self.objective, self.cov_excess_returns = None, None, None

        self.data = Data(self.currency, self.etf_list, static=static, backtest=backtest)
        self.etf_list += [ticker for ticker in Data.possible_currencies if ticker != self.currency]
        self.etf_list = sorted(list(set(self.etf_list)))
        self.n = len(self.etf_list)




        self.drop_too_new()

        self.get_objective()
        self.drop_highly_correlated()
        self.get_liquidity()
        self.cov_excess_returns = self.data.excess_returns.cov().values
        self.get_objective()
        self.crypto_opti = self.data.crypto_opti


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
        best_etfs = cluster_df.groupby('Cluster')['obj_values'].idxmin().tolist()

        to_drop = [ticker for ticker in self.etf_list if ticker not in best_etfs]
        for ticker in to_drop:
            self.remove_etf(ticker)


    def get_liquidity(self):

        self.liquidity = self.cash + sum(self.holdings.values())


    def get_objective(self):

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
