"""
Optimization and plotting for portfolio weights.

This module defines :class:`Opti`, a helper that:

* Builds bounds and constraints for a portfolio optimization (long-only or
  long/short with an L1 weight budget).
* Minimizes a user-provided objective exposed by a
  :class:`~etf_code.portfolio.Portfolio` instance (mean–variance or Sharpe-style,
  depending on how the portfolio defines ``objective``).
* Computes in-sample cumulative performance and several diagnostic visuals
  (allocation pie, cumulative vs. benchmark, contribution, and drawdown),
  returning each plot as a Dash-ready ``html.Img`` element while also saving
  PNGs to disk.

Notes
-----
* The solver is SciPy's ``minimize`` with SLSQP by default.
* For long/short, the equality constraint is ``sum(|w|) = 1``; for long-only,
  it is ``sum(w) = 1``.
* Colors for plots are generated deterministically from Matplotlib's ``tab20``.
"""

import math

import numpy as np
import matplotlib
from dash import dash_table
from scipy.optimize import curve_fit
from etf_code.portfolio import Portfolio
import matplotlib.cm as cm
import matplotlib.colors as mcolors
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from data import Data
from scipy.optimize import minimize
import io
import base64
from dash import html
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import gurobipy as gp
from gurobipy import GRB

class Opti:
    """
    Portfolio optimizer and plotting utility.

    Class Attributes
    ----------------
    solver_method : str
        Optimization algorithm passed to :func:`scipy.optimize.minimize`
        (default: ``"SLSQP"``).
    graph_dir_path : pathlib.Path
        Root directory where PNG plots will be saved.

    Parameters
    ----------
    portfolio : Portfolio
        A portfolio object exposing:
        * ``n`` (universe size),
        * ``allow_short`` (bool),
        * ``objective(w=..., single_ticker=None)`` (callable for minimization),
        * ``etf_list`` (tickers),
        * ``data`` with ``returns``, ``spy``, and ``rf_rate``,
        * ``currency`` (base currency code),
        * ``name`` (label for titles),
        * ``crypto`` (bool; used only for labeling the benchmark line).

    Attributes
    ----------
    optimum : dict[str, float] | None
        Sparse weight mapping after thresholding small weights and renormalizing.
    optimum_all : dict[str, float] | None
        Full weight vector (including near-zeros) as a mapping.
    w_opt : numpy.ndarray | None
        Optimized weight vector.
    constraints : list[dict] | None
        Nonlinear equality constraint(s) for the optimizer.
    bounds : list[tuple[float, float]] | None
        Per-asset bounds, long-only or long/short per portfolio settings.
    cumulative : pandas.Series | None
        In-sample cumulative performance of the optimized portfolio (gross of RF).
    returns : pandas.Series | pandas.DataFrame | None
        Subset of simple returns used to compute :attr:`cumulative`.
    color_map : dict[str, str] | None
        Ticker-to-HEX color mapping used by plotting helpers.
    portfolio : Portfolio
        Reference to the provided portfolio object.
    w0 : numpy.ndarray
        Starting point for optimization (uniform weights).
    """

    solver_method = 'SLSQP'
    graph_dir_path = Path(__file__).resolve().parent.parent / "graphs"

    def __init__(self, portfolio):
        """
        Initialize the optimizer, solve for weights, and compute performance.

        The constructor:
        1) builds bounds and constraints,
        2) sets a uniform initial guess,
        3) runs the optimization,
        4) computes cumulative in-sample performance,
        5) builds a deterministic color map.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio-like object exposing an ``objective`` and market data.

        Returns
        -------
        None
        """
        self.optimum, self.optimum_all, self.w_opt, self.constraints, self.bounds, self.cumulative, self.returns, self.color_map = None, None, None, None, None, None, None, None
        self.portfolio = portfolio
        self.get_bounds()
        self.get_constraints()
        self.w0 = np.full(self.portfolio.n, 1 / self.portfolio.n)
        self.optimize()
        self.get_cumulative()

        self.get_color_map()

    def get_color_map(self):
        """
        Build a deterministic HEX color mapping for the optimized universe.

        Colors are drawn from Matplotlib's ``tab20`` colormap and assigned in the
        order of :attr:`optimum_all` keys to ensure reproducibility.

        Returns
        -------
        None
        """
        cmap = cm.get_cmap('tab20', len(self.optimum_all))
        self.color_map = {asset: mcolors.to_hex(cmap(i)) for i, asset in enumerate(self.optimum_all.keys())}

    def get_bounds(self):
        """
        Build per-asset bounds based on shorting permission.

        * If shorting is allowed: ``(-1, 1)``.
        * If long-only: ``(0, 1)``.

        Returns
        -------
        None
        """
        self.bounds = [(-1, 1)] * self.portfolio.n

    @staticmethod
    def abs_sum(lst):
        """
        L1 norm (sum of absolute values).

        Parameters
        ----------
        lst : list[float] | numpy.ndarray | tuple[float, ...]
            Iterable of numbers.

        Returns
        -------
        float
            Sum of absolute values.
        """
        return sum([abs(x) for x in lst])

    @staticmethod
    def save_fig_as_dash_img(fig, output_path):
        """
        Convert a Matplotlib figure to a Dash ``html.Img`` (and save to disk).

        If ``output_path`` is not ``None``, the PNG is written to that path.
        The function always returns an inline base64-encoded ``html.Img`` element.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Matplotlib figure to serialize.
        output_path : str | pathlib.Path | None
            File path for saving the PNG (or ``None`` to skip saving).

        Returns
        -------
        dash.html.Img
            Image component with the figure embedded as a data URI.
        """
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(output_path, format="png", bbox_inches="tight")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        encoded = base64.b64encode(buf.read()).decode("utf-8")
        img_src = f"data:image/png;base64,{encoded}"

        return html.Img(src=img_src, style={"maxWidth": "100%", "height": "auto"})

    @staticmethod
    def softcount(w, tau=2e-3, alpha=200.0, eps=1e-12):
        abs_smooth = np.sqrt(w * w + eps)
        return 1.0 / (1.0 + np.exp(-alpha * (abs_smooth - tau)))



    def get_constraints(self):
        """
        Construct the weight-budget equality constraint.

        * Long-only: enforce ``sum(w) = 1``.
        * Long/short: enforce ``sum(|w|) = 1``.

        Returns
        -------
        None
        """
        func = sum
        func = Opti.abs_sum

        K = 15
        self.constraints = [
            {'type': 'eq', 'fun': lambda w: func(w) - 1, 'tol': 1e-3},
            {'type': 'ineq', 'fun': lambda w: K - np.sum(Opti.softcount(w, tau=1e-2, alpha=100.0))}
        ]

        self.constraints = [{'type': 'eq', 'fun': lambda w: func(w) - 1, 'tol': 1e-3}]


    def optimize(self, max_assets=10, borrow_years=1/12):
        """
        Solve the portfolio optimization problem using Gurobi with a cardinality constraint
        and a borrow-cost penalty for short positions, modeled via w_plus/w_minus.

        Parameters
        ----------
        max_assets : int
            Maximum number of assets allowed in the portfolio (cardinality).
        borrow_rate_annual : float
            Annual borrow fee for short positions (e.g. 0.02 for 2% per year).
        borrow_years : float
            Effective number of years over which to charge the borrow fee
            in the objective (1.0 = 1 year; use 20.0 if you want 20-year cost).
        """

        borrow_rate_annual = {tick:0. for tick in self.portfolio.etf_list}
        # https://www.interactivebrokers.com/en/pricing/reference-benchmark-rates-int.php



        df_repo = pd.read_csv(Data.data_dir_path / 'repo.csv', sep=';')
        df_repo['REPO'] = df_repo['REPO'].str.replace(',', '.').astype(float)

        for _, row in df_repo.iterrows():
            borrow_rate_annual[row['TICKER']] = row['REPO']/100


        # Build data for a mean-variance objective from historical returns
        rets = self.portfolio.data.returns[self.portfolio.etf_list]
        rets[self.portfolio.currency] += ((1.01)**(1/12))-1
        mu = rets.mean().values  # expected returns (vector)
        Sigma = rets.cov().values  # covariance matrix (n x n)
        n = self.portfolio.n

        # Create model
        m = gp.Model("portfolio")

        # Big-M consistent with bounds on w_plus, w_minus
        # Since |w_i| <= 1 and L1 sum = 1, M = 1.0 is fine.
        M = 1.0

        # Decompose weights into long/short parts:
        #   w_i = w_plus[i] - w_minus[i]
        #   w_plus[i] >= 0, w_minus[i] >= 0
        # Short exposure is exactly w_minus[i].
        w_plus = m.addVars(n, lb=0.0, ub=M, name="w_plus")
        w_minus = m.addVars(n, lb=0.0, ub=M, name="w_minus")

        # Binary variables for selection (cardinality)
        z = m.addVars(n, vtype=GRB.BINARY, name="z")

        # Link w_plus, w_minus to selection z:
        # If z_i = 0 => w_plus[i] = w_minus[i] = 0
        for i in range(n):
            m.addConstr(w_plus[i] <= M * z[i])
            m.addConstr(w_minus[i] <= M * z[i])

        # L1 budget: sum(|w_i|) = sum(w_plus[i] + w_minus[i]) = 1
        m.addConstr(
            gp.quicksum(w_plus[i] + w_minus[i] for i in range(n)) == 1.0,
            name="budget"
        )

        # Cardinality: at most max_assets non-zero positions
        m.addConstr(
            gp.quicksum(z[i] for i in range(n)) <= max_assets,
            name="cardinality"
        )

        # Define the actual weights as expressions for convenience
        w_expr = {i: w_plus[i] - w_minus[i] for i in range(n)}

        # Mean-variance objective:  w' Σ w  - μ' w  + borrow_cost(shorts)
        var_term = gp.quicksum(
            Sigma[i, j] * w_expr[i] * w_expr[j] for i in range(n) for j in range(n)
        )
        ret_term = gp.quicksum(mu[i] * w_expr[i] for i in range(n))


        # Build per-asset borrow cost
        borrow_cost = gp.quicksum(
            borrow_rate_annual[self.portfolio.etf_list[i]] * borrow_years * w_minus[i]
            for i in range(n)
        )

        # Final objective
        m.setObjective(
            var_term * self.portfolio.weight_cov - ret_term + borrow_cost,
            GRB.MINIMIZE
        )

        # Solve
        m.Params.OutputFlag = 0  # silence solver; remove if you want logs
        m.update()
        m.optimize()

        if m.status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi optimization failed with status {m.status}")

        # Extract weights: w_i = w_plus - w_minus
        w_opt = np.array(
            [w_plus[i].X - w_minus[i].X for i in range(n)],
            dtype=float
        )

        w_opt /= Opti.abs_sum(w_opt)

        self.w_opt = w_opt
        self.optimum_all = {
            tick: w for tick, w in zip(self.portfolio.etf_list, self.w_opt)
        }

        self.optimum = {
            ticker: weight
            for ticker, weight in self.optimum_all.items()
            if abs(weight) >= 0.01
        }

    def get_cumulative(self):
        """
        Compute in-sample cumulative performance for the optimized weights.

        Uses simple returns from ``self.portfolio.data.returns`` and the
        sparse weight mapping in :attr:`optimum`.

        Notes
        -----
        This method *re-runs* the optimization internally (duplicating the logic
        in :meth:`optimize`) before computing cumulative performance. This mirrors
        the existing behavior and ensures :attr:`optimum` and :attr:`optimum_all`
        are refreshed, but may be redundant if called repeatedly.

        Returns
        -------
        None
        """
        self.returns = self.portfolio.data.returns[self.optimum.keys()]
        weights = list(self.optimum.values())
        self.cumulative = (1 + self.returns @ weights).cumprod()

    def plot_optimum(self):
        """
        Plot the optimized allocation as a pie chart.

        Colors are pulled from the instance color map built at construction.
        The image is saved under
        ``graphs/<currency>/<name>- optimal_allocation.png`` and also returned
        as a Dash image.

        Returns
        -------
        dash.html.Img
            Dash component containing the pie chart.
        """
        sorted_optimum = dict(sorted(self.optimum.items(), key=lambda item: item[1], reverse=True))

        fig, ax = plt.subplots()
        colors = [self.color_map[k] for k in sorted_optimum.keys()]
        ax.pie(
            [abs(sorted_optimum[x]) for x in sorted_optimum],
            labels=[x if sorted_optimum[x]>=0 else '--'+x for x in sorted_optimum],
            colors=colors,
            autopct=lambda pct: f'{int(round(pct))}%'
        )
        ax.set_title('Optimal Allocation')

        output_path = Opti.graph_dir_path / f'{self.portfolio.currency}/{self.portfolio.name}- optimal_allocation.png'
        return Opti.save_fig_as_dash_img(fig, output_path)

    def plot_in_sample(self):
        """
        Plot in-sample cumulative performance vs. market proxy and RF leg.

        The title additionally includes annualized performance and max drawdown
        computed from :attr:`cumulative`. The benchmark line label switches to
        ``BTC`` when the portfolio is in crypto mode.

        Returns
        -------
        dash.html.Img
            Dash component containing the time-series chart.
        """
        fig, ax = plt.subplots()
        ax.plot((self.cumulative - 1) * 100, label=str(self.portfolio.name) + f' ({self.portfolio.currency})')

        spy = (self.portfolio.data.spy / self.portfolio.data.spy.iloc[0] - 1) * 100
        label = 'BTC' if self.portfolio.crypto else 'Total stock market'
        ax.plot(spy, label=f'{label} ({self.portfolio.currency})', linestyle='--')

        rf_rate = ((self.portfolio.data.rf_rate + 1).cumprod() - 1) * 100
        ax.plot(rf_rate, label='Rate', linestyle='--')

        ax.axhline(0, color='black')

        nb_years = int(self.portfolio.data.period[:-1])
        pa_perf = round(((self.cumulative.iloc[-1]) ** (1 / nb_years) - 1) * 100, 1)

        running_max = self.cumulative.cummax()
        drawdown = (self.cumulative - running_max) / running_max
        max_drawdown = round(drawdown.min() * 100, 1)

        ax.set_title(f'In-Sample')
        ax.set_ylabel('%')
        ax.legend()
        ax.grid()

        output_path = Opti.graph_dir_path / f'{self.portfolio.currency}/{self.portfolio.name}- in_sample.png'
        return Opti.save_fig_as_dash_img(fig, output_path)

    def plot_weighted_perf(self):
        """
        Plot in-sample performance attribution by constituent.

        The contribution per asset is the weighted cumulative excess over 1
        (in percent). Colors follow the portfolio color map.

        Returns
        -------
        dash.html.Img
            Dash component containing the attribution chart.
        """
        returns = self.portfolio.data.returns[self.optimum.keys()]
        weights = pd.Series(self.optimum)

        cumulative_returns = (1 + returns).cumprod() - 1
        contribution = cumulative_returns.multiply(weights, axis=1) * 100

        fig, ax = plt.subplots()
        for col in contribution.columns:
            ax.plot(contribution.index, contribution[col], label=col, color=self.color_map[col])

        ax.legend()
        ax.set_title('In-Sample Performance Attribution')
        ax.axhline(0, color='black')
        ax.set_ylabel('%')
        ax.grid()

        output_path = Opti.graph_dir_path / f'{self.portfolio.currency}/{self.portfolio.name}- perf_attrib.png'
        return Opti.save_fig_as_dash_img(fig, output_path)

    def plot_drawdown(self):
        """
        Plot the portfolio drawdown curve as an area chart below zero.

        Returns
        -------
        dash.html.Img
            Dash component containing the drawdown chart.
        """
        rolling_max = self.cumulative.cummax()
        drawdown = self.cumulative / rolling_max - 1

        fig, ax = plt.subplots()
        ax.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=.5)

        ax.set_title(f'Drawdown')
        ax.set_ylabel('%')
        ax.grid()

        output_path = Opti.graph_dir_path / f'{self.portfolio.currency}/{self.portfolio.name}- drawdown.png'
        return Opti.save_fig_as_dash_img(fig, output_path)

    def plot_info(self):
        """
        Assemble a compact metrics table (Dash DataTable) for the optimized portfolio.

        Metrics reported
        ----------------
        * CAGR — average annual growth rate (derived from cumulative).
        * Sharpe ratio — monthly mean/std scaled by ``sqrt(12)``.
        * Max/Avg drawdown — from the cumulative equity curve.
        * Beta — covariance with benchmark (VTI or BTC-USD) over benchmark variance.
        * Volatility — monthly std scaled by ``sqrt(12)``.
        * VaR 95% — empirical 5th percentile of monthly returns.
        * R² — OLS fit of portfolio returns on the benchmark.

        Returns
        -------
        dash_table.DataTable
            A ready-to-render Dash table with metric names, values, and short
            descriptions.
        """
        info = {}
        explain = {}
        weights = list(self.optimum.values())
        #self.returns = self.returns[self.optimum.keys()]
        returns = self.returns @ weights

        nb_years = int(self.portfolio.data.period[:-1])
        pa_perf = (round(((self.cumulative.iloc[-1]) ** (1 / nb_years) - 1) * 100, 1))
        info['CAGR'] = str(round(pa_perf, 1)) + ' %'
        explain['CAGR'] = 'Average annual growth rate'

        sharpe = returns.mean() / returns.std()
        info['Sharpe ratio'] = round(sharpe * math.sqrt(12), 2)
        explain['Sharpe ratio'] = 'Risk-adjusted return'

        running_max = self.cumulative.cummax()
        drawdown = (self.cumulative - running_max) / running_max
        info['Max drawdown'] = str(round(drawdown.min() * 100, 1)) + ' %'
        info['Avg drawdown'] = str(round(drawdown.mean() * 100, 1)) + ' %'
        explain['Max drawdown'] = 'Largest peak-to-trough loss'
        explain['Avg drawdown'] = 'Typical loss during downturns'

        label = 'BTC-USD' if self.portfolio.crypto else 'VTI'
        spy = self.portfolio.data.spy[label].pct_change().dropna()
        beta = returns[1:].cov(spy) / spy.var()
        info['Beta'] = round(beta, 2)
        explain['Beta'] = 'Sensitivity to market movements'

        vol = returns.std() * math.sqrt(12)
        info['Volatility'] = round(vol, 2)
        explain['Volatility'] = 'Return fluctuations (risk)'

        var95 = np.percentile(returns, (1 - .95) * 100)
        info['VaR 95%'] = str(round(var95 * 100, 1)) + ' %'
        explain['VaR 95%'] = 'Max expected loss at 95% confidence'

        X = sm.add_constant(spy)
        #model = sm.OLS(returns[1:], X).fit()
        r2 = 0#model.rsquared
        info['R2'] = str(round(100 * r2)) + ' %'
        explain['R2'] = '% of returns explained by benchmark'

        # Convert dict into list of dicts for DataTable
        data = [{"Metric": k, "Value": info[k], 'Detail': explain[k]} for k in info]

        return dash_table.DataTable(
            data=data,
            columns=[
                {"name": "Metric", "id": "Metric"},
                {"name": "Value", "id": "Value"},
                {"name": "Detail", "id": "Detail"}
            ],
            page_size=10,
            sort_action='native',
            style_table={'overflowX': 'auto'},
            style_as_list_view=True,
            style_header={
                'fontWeight': '600',
                'border': 'none',
                'textAlign': 'center'
            },
            style_cell={
                'padding': '14px',
                'border': 'none',
                'textAlign': 'center',
                'fontSize': '16px'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgba(0,0,0,0.02)'
                }
            ]
        )


def sanity_check_transform_weight():
    """
    Quick visual sanity check for the risk-to-performance mapping.

    Runs the full optimization for discrete risk levels (0..10) using a
    USD, long-only portfolio with static data and sample currency rates.
    It then fits a line to the resulting annualized performance across
    risk levels and shows a scatter + fitted line with MSE.

    Notes
    -----
    This function *displays* a Matplotlib figure and is intended for manual,
    ad-hoc sanity checks rather than automated testing.
    """
    R = list(np.linspace(0, 10, 11))
    returns = []

    for risk in R:
        opti = Opti(Portfolio(risk=risk, cash=100, holdings=None, currency='USD', static=True, backtest=None, rates={'EUR': 1.7, 'SGD': 1.8}, crypto=False))
        pa_perf = round(((opti.cumulative.iloc[-1]) ** (1 / 20) - 1) * 100, 1)
        returns.append(pa_perf)

    R = np.array(R)
    returns = np.array(returns)
    a, b = np.polyfit(R, returns, 1)

    x = np.linspace(-1, 11, 1000)
    y = [a * i + b for i in x]

    mse = 0
    index, biggest = 0, 0
    for i in range(11):
        component = (returns[i] - (a * i + b))**2
        mse += component
        if component > biggest:
            index, biggest = i, component

    plt.scatter(R, returns)
    plt.plot(x, y)
    plt.title(f'MSE: {round(mse/11, 3)}, largest: {index}')

    plt.show()


import numpy as np
from scipy.optimize import curve_fit

# Model function: a * exp(bx) + c
def exp_func(x, a, b, c):
    """
    Exponential model used for curve fitting: ``a * exp(b * x) + c``.

    Parameters
    ----------
    x : array-like
        Input domain.
    a : float
    b : float
    c : float

    Returns
    -------
    numpy.ndarray
        Model values at ``x``.
    """
    return a * np.exp(b * x) + c


def fit_exponential(points):
    """
    Fit a function of the form ``a * exp(b * x) + c`` to a set of points.

    Parameters
    ----------
    points : list[tuple[float, float]]
        Sequence of (x, y) observations.

    Returns
    -------
    tuple[float, float, float]
        The fitted parameters ``(a, b, c)``.

    Notes
    -----
    This helper prints the fitted parameters to stdout and returns them.
    """
    # Convert to numpy arrays
    x_data = np.array([p[0] for p in points], dtype=float)
    y_data = np.array([p[1] for p in points], dtype=float)

    # Fit using curve_fit
    popt, _ = curve_fit(exp_func, x_data, y_data, p0=(1, 0.1, 0))  # initial guess

    print(popt)
    return tuple(popt)


# Sample calibration points used by :func:`fit_exponential` when testing.
points = [(0, 49), (1, 35.5), (2, 25), (3, 17.5), (4, 12), (5, 8), (6, 4.75), (7, 3.33), (8, 2.5), (9, 1.2), (10, 0)]

# sanity_check_transform_weight()
# fit_exponential(points)
