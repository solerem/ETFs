import math

import numpy as np
import matplotlib
from dash import dash_table, dcc
from scipy.optimize import curve_fit
from portfolio import Portfolio
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
import plotly.graph_objects as go

class Opti:
    solver_method = 'SLSQP'
    graph_dir_path = Path(__file__).resolve().parent.parent / "graphs"

    def __init__(self, portfolio):
        self.optimum, self.optimum_all, self.w_opt, self.constraints, self.bounds, self.cumulative, self.returns, self.color_map = None, None, None, None, None, None, None, None
        self.portfolio = portfolio
        self.get_bounds()
        self.get_constraints()
        self.w0 = np.full(self.portfolio.n, 1 / self.portfolio.n)
        self.optimize()
        self.get_cumulative()

        self.get_color_map()

    def get_color_map(self):
        cmap = cm.get_cmap('tab20', len(self.optimum_all))
        self.color_map = {asset: mcolors.to_hex(cmap(i)) for i, asset in enumerate(self.optimum_all.keys())}

    def get_bounds(self):
        self.bounds = [(-1, 1)] * self.portfolio.n

    @staticmethod
    def abs_sum(lst):
        return sum([abs(x) for x in lst])

    @staticmethod
    def save_fig_as_dash_img(fig, output_path):
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
        func = sum
        func = Opti.abs_sum

        K = 15
        self.constraints = [
            {'type': 'eq', 'fun': lambda w: func(w) - 1, 'tol': 1e-3},
            {'type': 'ineq', 'fun': lambda w: K - np.sum(Opti.softcount(w, tau=1e-2, alpha=100.0))}
        ]

        self.constraints = [{'type': 'eq', 'fun': lambda w: func(w) - 1, 'tol': 1e-3}]


    def optimize(self, max_assets=10, borrow_years=1/12):

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
        self.returns = self.portfolio.data.returns[self.optimum.keys()]
        weights = list(self.optimum.values())
        self.cumulative = (1 + self.returns @ weights).cumprod()

    def plot_optimum(self):
        sorted_optimum = dict(sorted(self.optimum.items(), key=lambda item: item[1], reverse=True))
        values = [abs(sorted_optimum[x]) for x in sorted_optimum]
        labels = [x if sorted_optimum[x] >= 0 else '--' + x for x in sorted_optimum]
        colors = [self.color_map[k] for k in sorted_optimum.keys()]
        full_name_list = [self.portfolio.data.etf_full_names.loc[ticker] for ticker in sorted_optimum]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            customdata=full_name_list
        )])
        fig.update_traces(
            textinfo='label+percent',
            texttemplate='%{label}: %{percent:.1%}',
            hovertemplate='%{label}: %{percent:.1%}<br>%{customdata}<extra></extra>'
        )
        fig.update_layout(title='Optimal Allocation')

        output_path = Opti.graph_dir_path / f'{self.portfolio.currency}/{self.portfolio.name}- optimal_allocation.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(output_path))
        except Exception:
            pass

        return dcc.Graph(
            figure=fig,
            config={'displaylogo': False, 'scrollZoom': True},
            style={'height': '420px'}
        )

    def plot_in_sample(self):
        cumulative_pct = (self.cumulative - 1) * 100

        spy = (self.portfolio.data.spy / self.portfolio.data.spy.iloc[0] - 1) * 100
        spy_col = 'BTC-USD' if self.portfolio.crypto else 'VTI'
        spy_name = 'BTC' if self.portfolio.crypto else 'Total stock market'
        if isinstance(spy, pd.DataFrame):
            if spy_col in spy.columns:
                spy = spy[spy_col]
            else:
                spy = spy.iloc[:, 0]

        rf_rate = ((self.portfolio.data.rf_rate + 1).cumprod() - 1) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumulative_pct.index,
            y=cumulative_pct.values,
            mode='lines',
            name="Strategy",
            hovertemplate='%{y:.1f}%<extra>%{fullData.name}</extra>'
        ))
        fig.add_trace(go.Scatter(
            x=spy.index,
            y=spy.values,
            mode='lines',
            name="Stocks",
            hovertemplate='%{y:.1f}%<extra>%{fullData.name}</extra>'
        ))
        fig.add_trace(go.Scatter(
            x=rf_rate.index,
            y=rf_rate.values,
            mode='lines',
            name='Rate',
            hovertemplate='%{y:.1f}%<extra>%{fullData.name}</extra>'
        ))

        fig.update_layout(
            title='In-sample Performance',
            yaxis_title='%',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5)
        )
        fig.update_yaxes(tickformat='.1f')
        fig.add_hline(y=0, line_color='black')

        # Preserve image export for existing static file output
        output_path = Opti.graph_dir_path / f'{self.portfolio.currency}/{self.portfolio.name}- in_sample.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(output_path))
        except Exception:
            pass

        return dcc.Graph(
            figure=fig,
            config={'displaylogo': False, 'scrollZoom': True},
            style={'height': '420px'}
        )

    def plot_weighted_perf(self):
        returns = self.portfolio.data.returns[self.optimum.keys()]
        weights = pd.Series(self.optimum)

        cumulative_returns = (1 + returns).cumprod() - 1
        contribution = cumulative_returns.multiply(weights, axis=1) * 100
        fig = go.Figure()
        for col in contribution.columns:
            fig.add_trace(go.Scatter(
                x=contribution.index,
                y=contribution[col].values,
                mode='lines',
                name=col,
                line=dict(color=self.color_map[col]),
                hovertemplate='%{y:.1f}%<extra>%{fullData.name}</extra>'
            ))

        fig.add_hline(y=0, line_color='black')
        fig.update_layout(
            title='In-sample Performance Breakdown',
            yaxis_title='%',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5)
        )
        fig.update_yaxes(tickformat='.1f')

        output_path = Opti.graph_dir_path / f'{self.portfolio.currency}/{self.portfolio.name}- perf_attrib.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(output_path))
        except Exception:
            pass

        return dcc.Graph(
            figure=fig,
            config={'displaylogo': False, 'scrollZoom': True},
            style={'height': '420px'}
        )

    def plot_drawdown(self):
        rolling_max = self.cumulative.cummax()
        drawdown = self.cumulative / rolling_max - 1
        drawdown_pct = drawdown * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown_pct.index,
            y=drawdown_pct.values,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red'),
            hovertemplate='%{y:.1f}%<extra>%{fullData.name}</extra>'
        ))

        fig.update_layout(
            title='Drawdown',
            yaxis_title='%',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5)
        )
        fig.update_yaxes(tickformat='.1f')

        output_path = Opti.graph_dir_path / f'{self.portfolio.currency}/{self.portfolio.name}- drawdown.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(output_path))
        except Exception:
            pass

        return dcc.Graph(
            figure=fig,
            config={'displaylogo': False, 'scrollZoom': True},
            style={'height': '420px'}
        )

    def plot_info(self):
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
        model = sm.OLS(returns[1:], X).fit()
        r2 = model.rsquared
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
    return a * np.exp(b * x) + c


def fit_exponential(points):
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
