from data import Data
import time
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np


class Info:
    threshold_correlation = .9

    etf_list = [
        'SPY', 'QQQ', 'DIA', 'MDY', 'IWM', 'XLY', 'XLP', 'XLE', 'XLV', 'XLF', 'XLI', 'XLB', 'XLK', 'XLU', 'EFA', 'EEM',
        'EWA', 'EWC', 'EWH', 'EWG', 'EWI', 'EWU', 'EWM', 'EWS', 'EWP', 'EWD', 'LQD', 'TLT', 'TIP', 'GLD', 'VTI',
        'IWN', 'IUSG', 'IYJ', 'EWL', 'VHT', 'IWB', 'XLU', 'IGE', 'RTH', 'VWO', 'IWV', 'EWW', 'EWC', 'EWN', 'VPU', 'PWB',
        'VIS', 'IYM', 'SPYV', 'SLYV', 'IUSV', 'AGG', 'IWF', 'EWZ', 'LQD', 'ILCB', 'IXN', 'VDE', 'VOX', 'XLG', 'IVW',
        'IJK', 'XLP', 'XSMO', 'IXC', 'EWY', 'IGM', 'IJH', 'PEJ', 'IVV', 'IYY', 'SOXX', 'EWP', 'VPL', 'IYH', 'VTV',
        'EWT', 'IYW', 'IMCG', 'EWH', 'IGPT', 'PJP', 'SPYG', 'ITOT', 'FXI', 'EWI', 'XLE', 'XLY', 'EWA', 'ILCG', 'IMCV',
        'XLI', 'IWM', 'DVY', 'VBK', 'EWG', 'IGV', 'IJS', 'XNTK', 'IYT', 'SPTM', 'PEY', 'VBR', 'EEM', 'PWV', 'TLT',
        'VFH', 'IEV', 'VB', 'SPEU', 'VGK', 'IYG', 'IWP', 'VTI', 'FEZ', 'EZU', 'IWR', 'VV', 'XLB', 'EWU', 'IJJ', 'IJR',
        'EFA', 'EPP', 'IEF', 'VDC', 'IBB', 'PBW', 'TIP', 'IWS', 'IYE', 'IWO', 'VUG', 'SUSA', 'ILCV', 'IYK', 'XMMO',
        'XLV', 'ONEQ', 'SHY', 'ISCB', 'EWJ', 'VXF', 'EWQ', 'PSI', 'ILF', 'IYR', 'IXG', 'IWD', 'IXP', 'VO', 'IDU', 'VGT',
        'EWD', 'IYZ', 'ISCV', 'ICF', 'IOO', 'SLYG', 'VCR', 'EWS', 'EZA', 'IVE', 'XLF', 'IMCB', 'IYF', 'VAW', 'OEF',
        'IJT', 'RWR', 'IXJ', 'SMH', 'IYC', 'ISCG', 'VNQ', 'XMVM', 'RSP', 'DGT', 'XLK',
        'SI=F', 'PL=F', 'PA=F'
    ]
    etf_list += [
        "^GSPC", "^DJI", "^IXIC", "^RUT", "^GDAXI", "^FCHI",
        "^STOXX50E", "^STOXX", "^HSI", "399001.SZ", "^BSESN",
        "^NSEI", "^AXJO", "^GSPTSE", "^BVSP"
    ]

    crypto_list = ['BTC',
                   'ETH']  # , 'XRP', 'SOL', 'DOGE', 'ADA', 'LINK', 'AVAX', 'XLM', 'HBAR', 'LTC', 'CRO', 'DOT', 'AAVE', 'NEAR', 'ETC']
    crypto_list = [f'{x}-USD' for x in crypto_list]

    etf_list = sorted(list(set(etf_list)))
    crypto_list = sorted(list(set(crypto_list)))

    name = {
        1: 'Low risk',
        2: 'Medium risk',
        3: 'High risk'
    }

    def __init__(self, risk, cash, holdings, currency, rates, crypto, static=False, override_weight_cov=None):
        self.weight_cov = None
        self.risk = risk
        self.cash = cash
        self.holdings = holdings if holdings else {}
        self.rates = rates if rates else {}
        self.crypto = crypto
        self.currency = currency if currency else 'USD'
        self.static = static
        self.name = 'Risk ' + str(self.risk)
        self.etf_list = Info.crypto_list if self.crypto else Info.etf_list
        self.n = len(self.etf_list)

    def get_weight_cov(self, override_weight_cov=None):
        if override_weight_cov is not None:
            self.weight_cov = float(override_weight_cov)
            return
        params = Data.get_weight_cov_params(static=self.static)
        coeffs = params[self.currency]
        self.weight_cov = (
                float(coeffs["a"]) * np.exp(float(coeffs["b"]) * self.risk)
                + float(coeffs["c"])
        )


class Portfolio(Info):
    def __init__(
            self,
            risk=3,
            cash=100,
            holdings=None,
            currency=None,
            static=False,
            backtest=None,
            rates=None,
            crypto=False,
            override_weight_cov=None,
    ):
        super().__init__(
            risk,
            cash,
            holdings,
            currency,
            rates,
            crypto,
            static=static,
            override_weight_cov=override_weight_cov,
        )

        self.liquidity, self.objective, self.cov_excess_returns = None, None, None
        self.data = Data(self.currency, self.etf_list, static=static, backtest=backtest, rates=self.rates,
                         crypto=crypto)
        self.get_weight_cov(override_weight_cov=override_weight_cov)
        if not crypto:
            # Extend with currency pseudo-tickers so FX can be considered.
            self.etf_list += [ticker for ticker in Data.possible_currencies]

        self.etf_list = sorted(list(set(self.etf_list)))
        self.n = len(self.etf_list)
        self.drop_too_new()
        self.get_objective()
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
        log_returns_without_currency = self.data.log_returns.copy()

        if not self.crypto:
            if self.currency in log_returns_without_currency.columns:
                log_returns_without_currency.drop(self.currency, axis=1, inplace=True)

        corr = log_returns_without_currency.corr(method='pearson', min_periods=2).abs()
        tickers = corr.columns.tolist()
        diag_mask = np.eye(len(corr), dtype=bool)
        corr = corr.mask(diag_mask, 1.0)
        corr = corr.fillna(0.0)
        dist = 1.0 - corr
        dist = 0.5 * (dist + dist.T)
        dist = np.clip(dist, 0.0, 2.0)
        dist = dist.mask(diag_mask, 0.0)
        condensed = squareform(dist.values, checks=True)
        Z = linkage(condensed, method='average')
        t = 1.0 - Portfolio.threshold_correlation  # e.g., 0.05 for 0.95
        clusters = fcluster(Z, t, criterion='distance')

        cluster_df = pd.DataFrame({'ETF': tickers, 'Cluster': clusters})

        # Evaluate objective per single ticker (will use Sharpe-style in crypto mode).
        obj_values = {ticker: self.objective(single_ticker=ticker) for ticker in self.etf_list}
        obj_values = pd.Series(obj_values, name='obj_values')

        cluster_df = cluster_df.set_index('ETF').join(obj_values)
        best_etfs = cluster_df.groupby('Cluster')['obj_values'].idxmin().tolist()

        to_drop = [ticker for ticker in self.etf_list if ticker not in best_etfs and ticker != self.currency]
        if to_drop:
            self.data.nav.drop(columns=to_drop, inplace=True, errors='ignore')
            self.data.returns.drop(columns=to_drop, inplace=True, errors='ignore')
            self.data.log_returns.drop(columns=to_drop, inplace=True, errors='ignore')
            self.data.excess_returns.drop(columns=to_drop, inplace=True, errors='ignore')
            self.etf_list = list(self.data.nav.columns)
            self.n = len(self.etf_list)

    def get_liquidity(self):
        self.liquidity = self.cash + sum(abs(v) for v in self.holdings.values())

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

        def f_crypto(w=np.zeros(self.n), single_ticker=None):
            if single_ticker:
                excess_series = self.data.excess_returns[single_ticker]
                mean = excess_series.mean()
                std = excess_series.std()
                return -mean / std

            excess_series = self.data.excess_returns @ w
            mean = excess_series.mean()
            return -mean / excess_series.std()

        self.objective = f_crypto if self.crypto else f
