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
            "IEI", "XLP", "AVUV", "FNDF", "EMXC"'VOO', 'SMH']

    etf_list = [
    # Broad U.S. Equity
    "SPY",   # SPDR S&P 500 ETF Trust — launched Jan 22, 1993 :contentReference[oaicite:1]{index=1}
    "QQQ",   # Invesco QQQ Trust (NASDAQ‑100) — launched Mar 1999 :contentReference[oaicite:2]{index=2}
    "DIA",   # SPDR Dow Jones Industrial Average (“Diamonds”) — launched 1998 :contentReference[oaicite:3]{index=3}

    # U.S. Cap segments
    "MDY",   # SPDR S&P MidCap 400 — launched Apr 1995 :contentReference[oaicite:4]{index=4}
    "IWM",   # iShares Russell 2000 — launched 2000 :contentReference[oaicite:5]{index=5}

    # Sector ETFs (SPDR)
    "XLY", "XLP", "XLE", "XLV", "XLF", "XLI", "XLB", "XLK", "XLU",  # launched 1998 :contentReference[oaicite:6]{index=6}

    # International Developed Equity
    "EFA",   # iShares MSCI EAFE — launched Aug 2001 :contentReference[oaicite:7]{index=7}
    #"EFA Growth/Value variants (EFG/EFV)",
    "EEM",   # iShares MSCI Emerging Markets — launched 2003 :contentReference[oaicite:8]{index=8}

    # Country-specific equity (iShares WEBS line)
    "EWA", "EWC", "EWH", "EWG", "EWI", "EWJ", "EWU", "EWM", "EWS", "EWP", "EWD",  # launched 1996–2000 :contentReference[oaicite:9]{index=9}

    # Bond ETFs
    "LQD",   # iShares iBoxx $ Investment Grade Corporate Bond — launched Jul 2002 :contentReference[oaicite:10]{index=10}
    "TLT",   # iShares 20+ Year Treasury Bond — launched Jul 2002 :contentReference[oaicite:11]{index=11}
    "TIP",   # iShares TIPS Bond ETF — launched Dec 2003 :contentReference[oaicite:12]{index=12}

    # Commodity/Precious metals
    "GLD",   # SPDR Gold Shares — launched Nov 2004 :contentReference[oaicite:13]{index=13}

    # Currency

    # Thematic & Long‑short early movers

    # Broad-market Vanguard
    "VTI",   # Vanguard Total Stock Market ETF — launched 2001
]+['IWN', 'IUSG', 'IYJ', 'EWL', 'VHT', 'IWB', 'XLU', 'IGE', 'RTH', 'VWO', 'IWV', 'EWW', 'EWC', 'EWN', 'VPU', 'PWB', 'VIS', 'IYM', 'SPYV', 'SLYV', 'IUSV', 'AGG', 'IWF', 'EWZ', 'LQD', 'ILCB', 'IXN', 'VDE', 'VOX', 'XLG', 'IVW', 'IJK', 'XLP', 'XSMO', 'IXC', 'EWY', 'IGM', 'IJH', 'PEJ', 'IVV', 'IYY', 'SOXX', 'EWP', 'VPL', 'IYH', 'VTV', 'EWT', 'IYW', 'IMCG', 'EWH', 'IGPT', 'PJP', 'SPYG', 'ITOT', 'FXI', 'EWI', 'XLE', 'XLY', 'EWA', 'ILCG', 'IMCV', 'XLI', 'IWM', 'DVY', 'VBK', 'EWG', 'IGV', 'IJS', 'XNTK', 'IYT', 'SPTM', 'PEY', 'VBR', 'EEM', 'PWV', 'TLT', 'VFH', 'IEV', 'VB', 'SPEU', 'VGK', 'IYG', 'IWP', 'VTI', 'FEZ', 'EZU', 'IWR', 'VV', 'XLB', 'EWU', 'IJJ', 'IJR', 'EFA', 'EPP', 'IEF', 'VDC', 'IBB', 'PBW', 'TIP', 'IWS', 'IYE', 'IWO', 'VUG', 'SUSA', 'ILCV', 'IYK', 'XMMO', 'XLV', 'ONEQ', 'SHY', 'ISCB', 'EWJ', 'VXF', 'EWQ', 'PSI', 'ILF', 'IYR', 'IXG', 'IWD', 'IXP', 'VO', 'IDU', 'VGT', 'EWD', 'IYZ', 'ISCV', 'ICF', 'IOO', 'SLYG', 'VCR', 'EWS', 'EZA', 'IVE', 'XLF', 'IMCB', 'IYF', 'VAW', 'OEF', 'IJT', 'RWR', 'IXJ', 'SMH', 'IYC', 'ISCG', 'VNQ', 'XMVM', 'RSP', 'DGT', 'XLK']
    etf_list = sorted(list(set(etf_list)))

    etf_preference = {
        'SGD': {},
        'EUR': {},
        'USD': {}
    }

    currency_config = {
        #1: 'SGD',
        2: 'EUR',
        3: 'USD'
    }


    name = {
        1: 'Low risk',
        2: 'Medium risk',
        3: 'High risk'
    }


    def __init__(self, risk, cash_sgd, holdings, currency, allow_short):

        self.color_map, self.weight_cov = None, None
        self.risk = risk
        self.cash_sgd = cash_sgd
        self.holdings = holdings if holdings else {}
        self.allow_short = allow_short
        self.currency = currency if currency else 'USD'#Info.currency_config[self.risk]
        self.get_weight_cov()
        self.name = 'Risk ' + str(self.risk + 4)
        self.etf_list = Info.etf_list
        self.etf_preference = Info.etf_preference[self.currency]
        self.n = len(self.etf_list)
        self.transform_etf_preference()
        self.get_color_map()


    def transform_etf_preference(self):

        temp = {}
        for preferred in self.etf_preference:
            temp[preferred] = True
            temp[self.etf_preference[preferred]] = False

        self.etf_preference = temp


    def get_weight_cov(self):
        self.weight_cov = np.exp(self.risk-4)


    def get_color_map(self):
        cmap = cm.get_cmap('tab20', self.n)
        self.color_map = {asset: mcolors.to_hex(cmap(i)) for i, asset in enumerate(self.etf_list)}


class Portfolio(Info):


    def __init__(self, risk=3, cash_sgd=100, holdings=None, currency=None, allow_short=False, static=False, backtest=None):

        super().__init__(risk, cash_sgd, holdings, currency, allow_short)

        self.liquidity, self.objective, self.cov_excess_returns = None, None, None

        self.data = Data(self.currency, self.etf_list, static=static, backtest=backtest)

        self.drop_too_new()
        #self.cov_excess_returns = self.data.excess_returns.cov().values
        self.get_objective()
        self.cash = self.cash_sgd # / self.data.sgd_rate
        self.drop_highly_correlated()
        self.get_liquidity()
        #self.cov_excess_returns = self.data.excess_returns.cov().values
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
        best_etfs = cluster_df.groupby('Cluster')['obj_values'].idxmin().tolist()

        to_drop = [ticker for ticker in self.etf_list if ticker not in best_etfs]
        for ticker in to_drop:
            self.remove_etf(ticker)


    def get_liquidity(self):

        self.liquidity = self.cash + sum(self.holdings.values())


    def get_objective(self):

        def old_f(w=np.zeros(self.n), single_ticker=None):

            if single_ticker:
                excess_series = self.data.excess_returns[single_ticker]
                mean = excess_series.mean()
                var = excess_series.var()
                return self.weight_cov * var - mean

            excess_series = self.data.excess_returns @ w
            mean = excess_series.mean()
            return self.weight_cov * (w @ self.cov_excess_returns @ w) - mean

        def f(w=np.zeros(self.n), single_ticker=None):

            if single_ticker:
                excess_series = self.data.excess_returns[single_ticker]
                returns = self.data.returns[single_ticker]
            else:
                excess_series = self.data.excess_returns @ w
                returns = self.data.returns @ w

            prod = (1 + excess_series).product()
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            if single_ticker:
                return - self.weight_cov * (prod ** (
                            1 / 20) - 1) - max_drawdown  # self.weight_cov * (w @ self.cov_excess_returns @ w) - mean

            return -self.weight_cov * (prod**(1/20)-1) - max_drawdown #self.weight_cov * (w @ self.cov_excess_returns @ w) - mean

        self.objective = f



#x = Portfolio(risk=1, currency='SGD', static=False)