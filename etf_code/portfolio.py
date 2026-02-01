from data import Data
import time
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np


class Info:
    threshold_correlation = .85

    etf_list = [
        'SPY', 'QQQ', 'DIA', 'MDY',
        'EWM', 'GLD',
        'IWN', 'IUSG', 'IYJ', 'EWL', 'VHT', 'IWB', 'XLU', 'IGE', 'RTH', 'VWO', 'IWV', 'EWW', 'EWC', 'EWN', 'VPU', 'PWB',
        'VIS', 'IYM', 'SPYV', 'SLYV', 'IUSV', 'AGG', 'IWF', 'EWZ', 'LQD', 'ILCB', 'IXN', 'VDE', 'VOX', 'XLG', 'DBC',
        'IJK', 'XLP', 'XSMO', 'IXC', 'EWY', 'IGM', 'IJH', 'PEJ', 'IYY', 'SOXX', 'EWP', 'VPL', 'IYH', 'VTV',
        'EWT', 'IYW', 'IMCG', 'EWH', 'IGPT', 'PJP', 'SPYG', 'FXI', 'EWI', 'XLE', 'XLY', 'EWA', 'ILCG', 'IMCV',
        'XLI', 'IWM', 'DVY', 'VBK', 'EWG', 'IGV', 'IJS', 'XNTK', 'IYT', 'SPTM', 'PEY', 'VBR', 'EEM', 'PWV', 'TLT',
        'VFH', 'IEV', 'VB', 'SPEU', 'VGK', 'IYG', 'IWP', 'VTI', 'FEZ', 'EZU', 'IWR', 'VV', 'XLB', 'EWU', 'IJJ', 'IJR',
        'EFA', 'EPP', 'IEF', 'VDC', 'IBB', 'PBW', 'TIP', 'IWS', 'IYE', 'IWO', 'VUG', 'SUSA', 'ILCV', 'IYK', 'XMMO',
        'XLV', 'ONEQ', 'SHY', 'ISCB', 'EWJ', 'VXF', 'EWQ', 'PSI', 'ILF', 'IYR', 'IXG', 'IWD', 'IXP', 'VO', 'IDU', 'VGT',
        'EWD', 'IYZ', 'ISCV', 'ICF', 'IOO', 'SLYG', 'VCR', 'EWS', 'EZA', 'XLF', 'IMCB', 'IYF', 'VAW', 'OEF',
        'IJT', 'RWR', 'IXJ', 'SMH', 'IYC', 'ISCG', 'VNQ', 'XMVM', 'RSP', 'DGT', 'XLK', 'EWK', 'EWO', 'SDY', 'FVD',
        'SI=F', 'PL=F', 'PA=F', "^IXIC", "^GDAXI", "^FCHI",
        "^STOXX", "^HSI", "399001.SZ", "^BSESN",
        "^AXJO", "^BVSP", 'KBE', 'PRF', 'OIH', 'PFM', 'PHO', 'PBJ', 'BBH', 'PPH', 'IWC', 'KIE', 'FXE'
    ]

    etf_list = sorted(list(set(etf_list)))

    def __init__(self, risk, cash, holdings, currency, rates, static=False, refit_weights = False):
        self.weight_cov = None
        self.risk = risk
        self.cash = cash
        self.holdings = holdings if holdings else {}
        self.rates = rates if rates else {}
        self.refit_weights = refit_weights
        self.currency = currency if currency else 'USD'
        self.static = static
        self.etf_list = Info.etf_list.copy()
        self.n = len(self.etf_list)

    def get_weight_cov(self, override_weight_cov=None):
        self._used_linear_table = False
        if override_weight_cov is not None:
            self.weight_cov = float(override_weight_cov)
            return
        table = Data.get_risk_weight_cov_table(
            self.currency, static=self.static, refit_weights=self.refit_weights
        )
        if table is not None:
            risks, weight_covs = table
            r = np.clip(float(self.risk), 0.0, 10.0)
            self.weight_cov = float(np.interp(r, risks, weight_covs))
            self._used_linear_table = True
            return
        raise RuntimeError(
            f"risk_weight_cov_table.csv missing or has no entry for currency {self.currency}. "
            "Run weight_tune.save_weights_linear() to generate it."
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
            override_weight_cov=None,
            refit_weights=False,
            data=None,
    ):
        super().__init__(
            risk,
            cash,
            holdings,
            currency,
            rates,
            static=static,
            refit_weights=refit_weights,
        )

        self.liquidity, self.objective, self.cov_excess_returns = None, None, None
        if data is not None:
            self.data = data
            self.etf_list = list(self.data.nav.columns)
            self.n = len(self.etf_list)
        else:
            self.data = Data(self.currency, self.etf_list, static=static, backtest=backtest, rates=self.rates)
        self.get_weight_cov(override_weight_cov=override_weight_cov)
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
        self.w_opt_override = None
        if getattr(self, "_used_linear_table", False):
            wt = Data.get_risk_weights_table(
                self.currency, static=self.static, refit_weights=False
            )
            if wt is not None:
                risks, pivot = wt
                r = np.clip(float(self.risk), 0.0, 10.0)
                w_vec = np.array([
                    float(np.interp(r, risks, pivot[ticker].reindex(risks).fillna(0).values))
                    if ticker in pivot.columns else 0.0
                    for ticker in self.etf_list
                ], dtype=float)
                s = np.sum(np.abs(w_vec))
                if s > 0:
                    w_vec /= s
                self.w_opt_override = w_vec

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

        if self.currency in log_returns_without_currency.columns:
            log_returns_without_currency.drop(self.currency, axis=1, inplace=True)

        arr = log_returns_without_currency.to_numpy()  # (T, N)
        corr = np.corrcoef(arr.T)  # (N, N), rows of arr.T = variables (ETFs)
        corr = np.abs(corr)
        corr = pd.DataFrame(corr, index=log_returns_without_currency.columns,
                            columns=log_returns_without_currency.columns)
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

        # Evaluate objective per single ticker.
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
                mean_er = getattr(self.data, 'mean_er', None)
                var_er = getattr(self.data, 'var_er', None)
                if mean_er is not None and var_er is not None and single_ticker in mean_er.index:
                    mean = mean_er[single_ticker]
                    var = var_er[single_ticker]
                else:
                    excess_series = self.data.excess_returns[single_ticker]
                    mean = excess_series.mean()
                    var = excess_series.var()
                return self.weight_cov * var - mean

            excess_series = self.data.excess_returns @ w
            mean = excess_series.mean()
            return self.weight_cov * (w @ self.cov_excess_returns @ w) - mean

        self.objective = f
