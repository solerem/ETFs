"""
Optimization and plotting for portfolio weights.

This module defines :class:`Opti`, a small helper that:

* Builds bounds and constraints for a portfolio optimization (long-only or
  long/short with L1 weight budget).
* Minimizes a user-provided mean–variance-style objective exposed by a
  :class:`~portfolio.Portfolio` instance.
* Computes in-sample cumulative performance and a few diagnostic plots
  (allocation pie, cumulative vs. benchmark, contribution, and drawdown),
  returning each plot as a Dash-ready ``html.Img`` element while also saving
  PNGs to disk.

Notes
-----
* The solver is SciPy's ``minimize`` with SLSQP by default.
* For long/short, the equality constraint is ``sum(|w|) = 1``; for long-only,
  it is ``sum(w) = 1``.
"""

import numpy as np
import matplotlib
from scipy.optimize import curve_fit
from etf_code.portfolio import Portfolio

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from data import Data
from scipy.optimize import minimize
import io
import base64
from dash import html
import pandas as pd
from pathlib import Path


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
        * ``objective(w=...)`` (callable for minimization),
        * ``etf_list`` (tickers),
        * ``color_map`` (ticker -> HEX),
        * ``data`` with ``returns``, ``spy``, and ``rf_rate``,
        * ``currency`` (base currency code),
        * ``name`` (label for titles).

    Attributes
    ----------
    optimum : dict[str, float] | None
        Sparse weight mapping after thresholding small weights and renormalizing.
    optimum_all : dict[str, float] | None
        Full weight vector (including zeros) as a mapping.
    w_opt : numpy.ndarray | None
        Optimized weight vector.
    constraints : list[dict] | None
        Nonlinear equality constraint(s) for the optimizer.
    bounds : list[tuple[float, float]] | None
        Per-asset bounds, long-only or long/short per portfolio settings.
    cumulative : pandas.Series | None
        In-sample cumulative performance of the optimized portfolio.
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
        4) computes cumulative in-sample performance.

        :param portfolio: Portfolio-like object exposing an ``objective`` and data.
        :type portfolio: Portfolio
        :returns: ``None``.
        :rtype: None
        """
        self.optimum, self.optimum_all, self.w_opt, self.constraints, self.bounds, self.cumulative, self.returns = None, None, None, None, None, None, None
        self.portfolio = portfolio
        self.get_bounds()
        self.get_constraints()
        self.w0 = np.full(self.portfolio.n, 1 / self.portfolio.n)
        self.optimize()
        self.get_cumulative()

    def get_bounds(self):
        """
        Build per-asset bounds based on shorting permission.

        * If shorting is allowed: ``(-1, 1)``.
        * If long-only: ``(0, 1)``.

        :returns: ``None``.
        :rtype: None
        """
        self.bounds = ([(-1, 1)] if self.portfolio.allow_short else [(0, 1)]) * self.portfolio.n

    @staticmethod
    def abs_sum(lst):
        """
        L1 norm (sum of absolute values).

        :param lst: Iterable of numbers.
        :type lst: list[float] | numpy.ndarray | tuple[float, ...]
        :returns: Sum of absolute values.
        :rtype: float
        """
        return sum([abs(x) for x in lst])

    @staticmethod
    def save_fig_as_dash_img(fig, output_path):
        """
        Convert a Matplotlib figure to a Dash ``html.Img`` (and save to disk).

        If ``output_path`` is not ``None``, the PNG is written to that path.
        The function always returns an inline base64-encoded ``html.Img`` element.

        :param fig: Matplotlib figure to serialize.
        :type fig: matplotlib.figure.Figure
        :param output_path: File path for saving the PNG (or ``None``).
        :type output_path: str | pathlib.Path | None
        :returns: Dash image component with the figure embedded.
        :rtype: dash.html.Img
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

    def get_constraints(self):
        """
        Construct the weight-budget equality constraint.

        * Long-only: enforce ``sum(w) = 1``.
        * Long/short: enforce ``sum(|w|) = 1``.

        :returns: ``None``.
        :rtype: None
        """
        func = Opti.abs_sum if self.portfolio.allow_short else sum
        self.constraints = [{'type': 'eq', 'fun': lambda w: func(w) - 1, 'tol': 1e-3}]

    def optimize(self):
        """
        Solve the portfolio optimization problem.

        Minimizes ``self.portfolio.objective(w=w)`` under the configured
        bounds and equality constraint. Post-processes the solution by:
        * thresholding very small absolute weights (< 1%) to zero, then
        * renormalizing by the L1 norm so the budget equals 1.

        Side Effects
        ------------
        Sets :attr:`w_opt`, :attr:`optimum_all`, and :attr:`optimum`. Prints a
        message if SciPy reports failure.

        :returns: ``None`` (updates instance attributes).
        :rtype: None
        """
        opt = minimize(lambda w: self.portfolio.objective(w=w), self.w0, method=Opti.solver_method, bounds=self.bounds,
                       constraints=self.constraints, options={'ftol': 1e-6, 'maxiter': 1000})

        if not opt.success:
            print(f'Optimization failed: {opt.message}')
            return None

        self.w_opt = np.array([0. if abs(w) < .01 else float(w) for w in opt.x])
        self.w_opt /= Opti.abs_sum(self.w_opt)

        self.optimum_all = {tick: w for tick, w in zip(self.portfolio.etf_list, self.w_opt)}
        self.optimum = {ticker: self.optimum_all[ticker] for ticker in self.optimum_all if
                        self.optimum_all[ticker] != 0}

    def get_cumulative(self):
        """
        Compute in-sample cumulative performance for the optimized weights.

        Uses simple returns from ``self.portfolio.data.returns`` and the
        sparse weight mapping in :attr:`optimum`.

        :returns: ``None`` (sets :attr:`cumulative`).
        :rtype: None
        """
        self.returns = self.portfolio.data.returns[self.optimum.keys()]
        weights = list(self.optimum.values())
        self.cumulative = (1 + self.returns @ weights).cumprod()


        opt = minimize(lambda w: self.portfolio.objective(w=w), self.w0, method=Opti.solver_method, bounds=self.bounds,
                       constraints=self.constraints, options={'ftol': 1e-6, 'maxiter': 1000})

        if not opt.success:
            print(f'Optimization failed: {opt.message}')
            return None

        self.w_opt = np.array([0. if abs(w) < .01 else float(w) for w in opt.x])
        self.w_opt /= Opti.abs_sum(self.w_opt)

        self.optimum_all = {tick: w for tick, w in zip(self.portfolio.etf_list, self.w_opt)}
        self.optimum = {ticker: self.optimum_all[ticker] for ticker in self.optimum_all if
                        self.optimum_all[ticker] != 0}


    def plot_optimum(self):
        """
        Plot the optimized allocation as a pie chart.

        Colors are pulled from ``self.portfolio.color_map``. The image is saved
        under ``graphs/<currency>/<name>- optimal_allocation.png`` and also
        returned as a Dash image.

        :returns: Dash image component for embedding in a layout.
        :rtype: dash.html.Img
        """
        sorted_optimum = dict(sorted(self.optimum.items(), key=lambda item: item[1], reverse=True))

        fig, ax = plt.subplots()
        colors = [self.portfolio.color_map[k] for k in sorted_optimum.keys()]
        ax.pie(
            sorted_optimum.values(),
            labels=sorted_optimum.keys(),
            colors=colors,
            autopct=lambda pct: f'{int(round(pct))}%'
        )
        ax.set_title('Optimal Allocation')

        output_path = Opti.graph_dir_path / f'{self.portfolio.currency}/{self.portfolio.name}- optimal_allocation.png'
        return Opti.save_fig_as_dash_img(fig, output_path)

    def plot_in_sample(self):
        """
        Plot in-sample cumulative performance vs. market proxy and RF leg.

        The title includes the annualized performance (p.a.) and maximum
        drawdown computed from :attr:`cumulative`.

        :returns: Dash image component for embedding in a layout.
        :rtype: dash.html.Img
        """
        fig, ax = plt.subplots()
        ax.plot((self.cumulative - 1) * 100, label=str(self.portfolio.name) + f' ({self.portfolio.currency})')

        spy = (self.portfolio.data.spy / self.portfolio.data.spy.iloc[0] - 1) * 100
        ax.plot(spy, label=f'Total stock market ({self.portfolio.currency})', linestyle='--')

        rf_rate = ((self.portfolio.data.rf_rate + 1).cumprod() - 1) * 100
        ax.plot(rf_rate, label='Rate', linestyle='--')

        ax.axhline(0, color='black')

        nb_years = int(self.portfolio.data.period[:-1])
        pa_perf = round(((self.cumulative.iloc[-1]) ** (1 / nb_years) - 1) * 100, 1)

        running_max = self.cumulative.cummax()
        drawdown = (self.cumulative - running_max) / running_max
        max_drawdown = round(drawdown.min() * 100, 1)

        ax.set_title(f'In-Sample ({pa_perf}% p.a., {max_drawdown}% max drawdown)')
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

        :returns: Dash image component for embedding in a layout.
        :rtype: dash.html.Img
        """
        returns = self.portfolio.data.returns[self.optimum.keys()]
        weights = pd.Series(self.optimum)

        cumulative_returns = (1 + returns).cumprod()
        weighted_cumulative = cumulative_returns.multiply(weights, axis=1)
        contribution = weighted_cumulative.subtract(1 * weights, axis=1) * 100

        fig, ax = plt.subplots()
        for col in contribution.columns:
            ax.plot(contribution.index, contribution[col], label=col, color=self.portfolio.color_map[col])

        ax.legend()
        ax.set_title('In-Sample Performance Attribution')
        ax.axhline(0, color='black')
        ax.set_ylabel('%')
        ax.grid()

        output_path = Opti.graph_dir_path / f'{self.portfolio.currency}/{self.portfolio.name}- perf_attrib.png'
        return Opti.save_fig_as_dash_img(fig, output_path)

    def plot_drawdown(self):
        """
        Plot the portfolio drawdown curve (area below zero).

        :returns: Dash image component for embedding in a layout.
        :rtype: dash.html.Img
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


def sanity_check_transform_weight():

    R = list(np.linspace(0, 10, 11))
    returns = []

    for risk in R:
        opti = Opti(Portfolio(risk=risk, cash=100, holdings=None, currency='USD', allow_short=False, static=True, backtest=None, rates={'EUR': 1.7, 'SGD': 1.8}, crypto=False))
        pa_perf = round(((opti.cumulative.iloc[-1]) ** (1 / 20) - 1) * 100, 1)
        returns.append(pa_perf)

    R = np.array(R)
    returns = np.array(returns)
    a,b = np.polyfit(R, returns, 1)

    x = np.linspace(-1, 11, 1000)
    y = [a*i+b for i in x]

    mse = 0
    index, biggest = 0,0
    for i in range(11):
        component = (returns[i] - (a*i+b))**2
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
    return a * np.exp(b * x) + c

def fit_exponential(points):
    """
    Fit a function of the form a*exp(bx) + c to a set of points.

    Parameters:
        points (list of tuples): [(x1, y1), (x2, y2), ...]

    Returns:
        (a, b, c) : fitted parameters
    """
    # Convert to numpy arrays
    x_data = np.array([p[0] for p in points], dtype=float)
    y_data = np.array([p[1] for p in points], dtype=float)

    # Fit using curve_fit
    popt, _ = curve_fit(exp_func, x_data, y_data, p0=(1, 0.1, 0))  # initial guess

    print(popt)



points = [(0, 49), (1, 35.5), (2, 25), (3, 17.5), (4, 12), (5, 8), (6, 4.75), (7, 3.33), (8, 2.5), (9, 1.2), (10, 0)]

#sanity_check_transform_weight()
#fit_exponential(points)