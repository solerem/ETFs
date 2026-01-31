import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm
from data import Data
from portfolio import Portfolio
from opti import Opti
from pathlib import Path


class WeightTune:
    def __init__(
            self,
            risk=0,
            cash=100,
            holdings=None,
            currency='USD',
            static=True,
            backtest=None,
            rates=None,
            max_assets=20,
            borrow_years=1/Data.NB_PERIOD,
    ):
        self.portfolio = Portfolio(
            override_weight_cov=0,
            risk=risk,
            cash=cash,
            holdings=holdings,
            currency=currency,
            static=static,
            backtest=backtest,
            rates=rates,
        )
        self._portfolio_kwargs = {
            "cash": cash,
            "holdings": holdings,
            "currency": currency,
            "static": static,
            "backtest": backtest,
            "rates": rates,
        }
        self.max_assets = max_assets
        self.borrow_years = borrow_years
        self._mu, self._sigma = self._get_mu_sigma()
        self._borrow_rate_annual = self._get_borrow_rates()

    def _get_mu_sigma(self):
        rets = self.portfolio.data.returns[self.portfolio.etf_list].copy()
        rets[self.portfolio.currency] += ((1.01) ** (1 / Data.NB_PERIOD)) - 1
        mu = rets.mean().values
        sigma = rets.cov().values
        return mu, sigma

    def _get_borrow_rates(self):
        borrow_rate_annual = {tick: 0.0 for tick in self.portfolio.etf_list}
        try:
            df_repo = pd.read_csv(Data.static_dir_path / "repo.csv", sep=";")
            df_repo["REPO"] = df_repo["REPO"].str.replace(",", ".").astype(float)
            for _, row in df_repo.iterrows():
                borrow_rate_annual[row["TICKER"]] = row["REPO"] / 100
        except Exception:
            pass
        return borrow_rate_annual

    def _solve(self, weight_cov, mode):
        mu = self._mu
        sigma = self._sigma
        n = self.portfolio.n

        m = gp.Model("portfolio")
        m.Params.OutputFlag = 0

        M = 1.0
        w_plus = m.addVars(n, lb=0.0, ub=M, name="w_plus")
        w_minus = m.addVars(n, lb=0.0, ub=M, name="w_minus")
        z = m.addVars(n, vtype=GRB.BINARY, name="z")

        for i in range(n):
            m.addConstr(w_plus[i] <= M * z[i])
            m.addConstr(w_minus[i] <= M * z[i])

        m.addConstr(
            gp.quicksum(w_plus[i] + w_minus[i] for i in range(n)) == 1.0,
            name="budget",
        )
        m.addConstr(
            gp.quicksum(z[i] for i in range(n)) <= self.max_assets,
            name="cardinality",
        )

        w_expr = {i: w_plus[i] - w_minus[i] for i in range(n)}
        var_term = gp.quicksum(
            sigma[i, j] * w_expr[i] * w_expr[j] for i in range(n) for j in range(n)
        )
        ret_term = gp.quicksum(mu[i] * w_expr[i] for i in range(n))
        borrow_cost = gp.quicksum(
            self._borrow_rate_annual[self.portfolio.etf_list[i]]
            * self.borrow_years
            * w_minus[i]
            for i in range(n)
        )

        if mode == "min_variance":
            objective = var_term
        elif mode == "max_return":
            objective = -ret_term + borrow_cost
        else:
            objective = var_term * weight_cov - ret_term + borrow_cost

        m.setObjective(objective, GRB.MINIMIZE)
        m.optimize()

        if m.status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi optimization failed with status {m.status}")

        w_opt = np.array(
            [w_plus[i].X - w_minus[i].X for i in range(n)],
            dtype=float,
        )
        w_opt /= np.sum(np.abs(w_opt))
        return w_opt

    @staticmethod
    def _weights_match(w_a, w_b, tol=1e-4):
        return np.allclose(w_a, w_b, atol=tol, rtol=0.0)

    def find_safest_weight_cov(self, max_weight_cov=1e6, tol=1e-4, max_iter=40):
        w_safest = self._solve(weight_cov=max_weight_cov, mode="mean_variance")
        low = 0.0
        high = max_weight_cov

        for _ in range(max_iter):
            mid = (low + high) / 2.0
            w_mid = self._solve(weight_cov=mid, mode="mean_variance")
            if self._weights_match(w_mid, w_safest, tol=tol):
                high = mid
            else:
                low = mid

        return high

    def find_riskiest_weight_cov(self, max_weight_cov=1e6, tol=1e-4, max_iter=40):
        w_max_ret = self._solve(weight_cov=0.0, mode="max_return")

        low = 0.0
        high = 1.0
        while high < max_weight_cov:
            w_high = self._solve(weight_cov=high, mode="mean_variance")
            if not self._weights_match(w_high, w_max_ret, tol=tol):
                break
            low = high
            high *= 10.0

        if high >= max_weight_cov and self._weights_match(
                self._solve(weight_cov=high, mode="mean_variance"), w_max_ret, tol=tol
        ):
            return high

        for _ in range(max_iter):
            mid = (low + high) / 2.0
            w_mid = self._solve(weight_cov=mid, mode="mean_variance")
            if self._weights_match(w_mid, w_max_ret, tol=tol):
                low = mid
            else:
                high = mid

        return low

    def get_weight_cov_bounds(self, max_weight_cov=1e6, tol=1e-4, max_iter=40):
        safest = self.find_safest_weight_cov(
            max_weight_cov=max_weight_cov, tol=tol, max_iter=max_iter
        )
        safest = np.sqrt(safest) / 5
        riskiest = self.find_riskiest_weight_cov(
            max_weight_cov=max_weight_cov, tol=tol, max_iter=max_iter
        )
        riskiest *= 1.1
        return safest, riskiest

    @staticmethod
    def fit_exp_params(x0, y0, x1, y1, c=0.0):
        y0_adj = y0 + c
        y1_adj = y1 + c
        if y0_adj <= 0 or y1_adj <= 0:
            raise ValueError("y + c must be positive to fit exp form.")
        b = np.log(y1_adj / y0_adj) / (x1 - x0)
        a = y0_adj / np.exp(b * x0)
        return a, b, c

    @classmethod
    def fit_exp_params_with_anchor(cls, x0, y0, x1, y1, x2, y2, max_iter=60):
        eps = 1e-9
        c_low = -min(y0, y1) + eps
        c_high = max(y0, y1, y2) * 10.0

        def y_at_anchor(c_val):
            a_val, b_val, _ = cls.fit_exp_params(x0, y0, x1, y1, c=c_val)
            return a_val * np.exp(b_val * x2) - c_val

        def f(c_val):
            return y_at_anchor(c_val) - y2

        f_low = f(c_low)
        f_high = f(c_high)

        if np.sign(f_low) != np.sign(f_high):
            for _ in range(max_iter):
                c_mid = (c_low + c_high) / 2.0
                f_mid = f(c_mid)
                if np.sign(f_mid) == np.sign(f_low):
                    c_low = c_mid
                    f_low = f_mid
                else:
                    c_high = c_mid
                    f_high = f_mid
            c_best = (c_low + c_high) / 2.0
        else:
            candidates = np.linspace(c_low, c_high, 200)
            best_err = float("inf")
            c_best = 0.0
            for c_val in candidates:
                err = abs(f(c_val))
                if err < best_err:
                    best_err = err
                    c_best = c_val

        a, b, _ = cls.fit_exp_params(x0, y0, x1, y1, c=c_best)
        return a, b, c_best

    def _portfolio_mean_for_weight_cov(self, risk, weight_cov):
        portfolio = self._build_portfolio_for_weight_cov(risk, weight_cov)
        opti = Opti(portfolio)
        returns = portfolio.data.returns[portfolio.etf_list]
        series = returns @ opti.w_opt
        return float(series.mean())

    def _find_weight_cov_for_mean_target(
            self,
            risk,
            target_mean,
            weight_cov_min,
            weight_cov_max,
            num_points=25,
    ):
        if weight_cov_min <= 0:
            grid = np.linspace(weight_cov_min, weight_cov_max, num_points)
        else:
            grid = np.logspace(
                np.log10(weight_cov_min), np.log10(weight_cov_max), num_points
            )

        best_cov = None
        best_mean = None
        best_err = float("inf")

        for weight_cov in grid:
            mean = self._portfolio_mean_for_weight_cov(risk, weight_cov)
            err = abs(mean - target_mean)
            if err < best_err:
                best_err = err
                best_cov = weight_cov
                best_mean = mean

        refine_min = max(weight_cov_min, best_cov / 2.0)
        refine_max = min(weight_cov_max, best_cov * 2.0)
        if refine_min < refine_max:
            if refine_min <= 0:
                refine_grid = np.linspace(
                    refine_min, refine_max, max(10, num_points // 2)
                )
            else:
                refine_grid = np.logspace(
                    np.log10(refine_min),
                    np.log10(refine_max),
                    max(10, num_points // 2),
                )
            for weight_cov in refine_grid:
                mean = self._portfolio_mean_for_weight_cov(risk, weight_cov)
                err = abs(mean - target_mean)
                if err < best_err:
                    best_err = err
                    best_cov = weight_cov
                    best_mean = mean

        return best_cov, best_mean

    def get_weight_cov_formula(
            self,
            risk_low=0,
            risk_high=10,
            c=None,
            anchor_risk=5.5,
            anchor_mean=0.008,
            max_weight_cov=1e6,
            tol=1e-4,
            max_iter=40,
    ):
        safest, riskiest = self.get_weight_cov_bounds(
            max_weight_cov=max_weight_cov, tol=tol, max_iter=max_iter
        )
        if safest is None:
            raise RuntimeError("Could not find a safest weight_cov bound.")

        if c is None:
            anchor_weight_cov, anchor_mean_real = self._find_weight_cov_for_mean_target(
                risk=anchor_risk,
                target_mean=anchor_mean,
                weight_cov_min=safest,
                weight_cov_max=riskiest,
            )
            a, b, c = self.fit_exp_params_with_anchor(
                risk_low, safest, risk_high, riskiest, anchor_risk, anchor_weight_cov
            )
        else:
            anchor_weight_cov = None
            anchor_mean_real = None
            a, b, c = self.fit_exp_params(risk_low, safest, risk_high, riskiest, c=c)

        return {
            "a": a,
            "b": b,
            "c": -c,
        }

    def _build_portfolio_for_weight_cov(self, risk, weight_cov):
        return Portfolio(
            risk=risk,
            override_weight_cov=weight_cov,
            **self._portfolio_kwargs,
        )


def save_weights():
    rows = []
    for currency in tqdm(Data.possible_currencies):
        tune = WeightTune(currency=currency)
        weights = tune.get_weight_cov_formula()
        rows.append({"currency": currency, **weights})

    df = pd.DataFrame(rows)
    output_path = Path(Data.data_dir_path / "weight_cov.parquet")
    df.to_parquet(output_path, index=False, engine="pyarrow")
