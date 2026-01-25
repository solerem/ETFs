import pandas as pd
from portfolio import Portfolio
from opti import Opti
from tqdm import tqdm
from dash import dash_table, dcc
from data import Data
import statsmodels.api as sm
import numpy as np
import plotly.graph_objects as go


class Backtest:
    def __init__(self, opti):
        self.opti = opti
        self.portfolio = self.opti.portfolio
        # Adaptive train/test split (matches implementation)
        self.ratio_train_test = .8 if self.portfolio.data.period == '5y' else .9
        self.to_consider = self.opti.optimum.keys()
        self.w_opt, self.returns, self.n, self.cutoff, self.index, self.returns_decomp = None, None, None, None, None, None
        self.cumulative = None
        self.parse_data()
        self.get_returns()

    def parse_data(self):
        self.n = len(self.portfolio.data.nav)
        self.cutoff = int(self.ratio_train_test * self.n)
        self.index = list(self.portfolio.data.nav.index)

        self.w_opt = pd.DataFrame({ticker: [] for ticker in self.opti.portfolio.etf_list})
        for i in tqdm(range(self.cutoff, self.n)):
            portfolio = Portfolio(
                risk=self.portfolio.risk,
                currency=self.portfolio.currency,
                static=True,
                backtest=self.index[i],
                rates=self.portfolio.rates,
                crypto=self.opti.portfolio.crypto
            )
            optimum = Opti(portfolio, long_only=self.opti.long_only).optimum_all
            self.w_opt.loc[self.index[i]] = optimum

    def get_returns(self):
        self.returns_decomp = Data.get_test_data_backtest(self.portfolio.data.returns, self.index[self.cutoff])
        self.returns_decomp *= self.w_opt
        self.returns = self.returns_decomp.sum(axis=1)

    def plot_backtest(self):
        cumulative = (1 + self.returns).cumprod()
        cumulative_pct = (cumulative - 1) * 100

        spy_col = 'BTC-USD' if self.portfolio.crypto else 'SPY'
        spy = self.portfolio.data.benchmarks[spy_col].copy()
        spy = spy.loc[self.index[self.cutoff]:]
        spy = (spy / spy.iloc[0] - 1) * 100

        bonds_col = 'BTC-USD' if self.portfolio.crypto else 'AGG'
        bonds = self.portfolio.data.benchmarks[bonds_col].copy()
        bonds = bonds.loc[self.index[self.cutoff]:]
        bonds = (bonds / bonds.iloc[0] - 1) * 100

        gold_col = 'BTC-USD' if self.portfolio.crypto else 'GLD'
        gold = self.portfolio.data.benchmarks[gold_col].copy()
        gold = gold.loc[self.index[self.cutoff]:]
        gold = (gold / gold.iloc[0] - 1) * 100


        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=spy.index,
            y=spy.values,
            mode='lines',
            name="Stocks",
            line=dict(color='red'),
            opacity=0.6,
            hovertemplate='%{y:.1f}%<extra>%{fullData.name}</extra>'
        ))
        fig.add_trace(go.Scatter(
            x=bonds.index,
            y=bonds.values,
            mode='lines',
            name="Bonds",
            line=dict(color='blue'),
            opacity=0.6,
            hovertemplate='%{y:.1f}%<extra>%{fullData.name}</extra>'
        ))
        fig.add_trace(go.Scatter(
            x=gold.index,
            y=gold.values,
            mode='lines',
            name="Gold",
            line=dict(color='orange'),
            opacity=0.6,
            hovertemplate='%{y:.1f}%<extra>%{fullData.name}</extra>'
        ))
        fig.add_trace(go.Scatter(
            x=cumulative_pct.index,
            y=cumulative_pct.values,
            mode='lines',
            name="Portfolio",
            line=dict(width=4, color='green'),
            hovertemplate='%{y:.1f}%<extra>%{fullData.name}</extra>'
        ))
        fig.add_hline(y=0, line_color='black')
        fig.update_layout(
            title='Backtest Performance vs Benchmark',
            yaxis_title='%',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5)
        )
        fig.update_yaxes(tickformat='.1f')

        return dcc.Graph(
            figure=fig,
            config={'displaylogo': False, 'scrollZoom': True},
            style={'height': '420px'}
        )

    def plot_weights(self):
        import math
        from pathlib import Path

        # --- Prep & safety ---
        w = self.w_opt.copy()
        if w.empty:
            raise ValueError("w_opt is empty; nothing to plot.")

        # Ensure numeric & fill NaNs so means/cum sums behave
        w = w.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Only keep columns that actually exist
        to_consider = [t for t in getattr(self, "to_consider", []) if t in w.columns]

        mean_w = w.mean(axis=0)  # average per ticker over time
        # If everything is zero, avoid division by zero downstream
        total_weight = float(mean_w.sum())
        if math.isclose(total_weight, 0.0, abs_tol=1e-12):
            raise ValueError("All mean weights are zero; cannot compute 99% coverage set.")

        # --- Build the included set greedily ---
        included = set(to_consider)
        # Remaining candidates sorted by mean weight (desc)
        remaining = [c for c in w.columns if c not in included]
        remaining.sort(key=lambda x: float(mean_w.get(x, 0.0)), reverse=True)

        # Start with current ones; compute included coverage
        included_weight = float(mean_w[list(included)].sum()) if included else 0.0

        # Greedily add until reaching ≥ 99% cumulative mean weight
        target = 0.99 * total_weight
        while included_weight < target and remaining:
            nxt = remaining.pop(0)
            included.add(nxt)
            included_weight += float(mean_w[nxt])

        # Final plotting order: sort included by descending mean weight for readability
        tickers_to_plot = sorted(included, key=lambda x: float(mean_w.get(x, 0.0)), reverse=True)

        data = w[tickers_to_plot].abs().astype(float)
        row_sums = data.sum(axis=1).replace(0, np.nan)
        data = data.div(row_sums, axis=0).fillna(0.0) * 100.0
        fig = go.Figure()
        color_map = self.opti.color_map
        for col in tickers_to_plot:
            # Check if this is a short position (negative mean weight or in optimum as negative)
            is_short = mean_w.get(col, 0.0) < 0 or self.opti.optimum.get(col, 0) < 0
            display_name = 'short ' + col if is_short else col
            series_color = color_map.get(col)
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col].values,
                mode='lines',
                name=display_name,
                stackgroup='one',
                stackgaps='interpolate',
                fill='tonexty',
                fillcolor=series_color,
                line=dict(width=0.5, color=series_color),
                opacity=1.0,
                hovertemplate='%{y:.1f}%<extra>%{fullData.name}</extra>'
            ))

        fig.update_layout(
            title='Weights history',
            yaxis_title='%',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5)
        )
        fig.update_yaxes(range=[0, 100], tickformat='.1f')


        return dcc.Graph(
            figure=fig,
            config={'displaylogo': False, 'scrollZoom': True},
            style={'height': '420px'}
        )

    def plot_perf_attrib(self):
        returns = self.returns_decomp[self.to_consider]
        fig = go.Figure()
        for col in self.to_consider:
            display_name = col if self.opti.optimum[col] >= 0 else 'short ' + col
            fig.add_trace(go.Scatter(
                x=returns.index,
                y=(returns[col].cumsum() * 100).values,
                mode='lines',
                name=display_name,
                line=dict(color=self.opti.color_map[col]),
                hovertemplate='%{y:.1f}%<extra>%{fullData.name}</extra>'
            ))

        fig.add_hline(y=0, line_color='black')
        fig.update_layout(
            title='Backtest Performance Breakdown',
            yaxis_title='%',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5)
        )
        fig.update_yaxes(tickformat='.1f')

        return dcc.Graph(
            figure=fig,
            config={'displaylogo': False, 'scrollZoom': True},
            style={'height': '420px'}
        )

    def plot_drawdown(self):
        cumulative = (1 + self.returns).cumprod()

        rolling_max = cumulative.cummax()
        drawdown = cumulative / rolling_max - 1
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
            title='Backtest Drawdown',
            yaxis_title='%',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5)
        )
        fig.update_yaxes(tickformat='.1f')

        return dcc.Graph(
            figure=fig,
            config={'displaylogo': False, 'scrollZoom': True},
            style={'height': '420px'}
        )

    def plot_info(self):
        info = {}
        explain = {}
        self.cumulative = (1 + self.returns).cumprod()

        # Calculate actual backtest period in years
        start_date = self.returns.index[0]
        end_date = self.returns.index[-1]
        nb_years = (end_date - start_date).days / 365.25
        if nb_years <= 0:
            nb_years = len(self.returns) / 12.0  # Fallback to months/12 if days calculation fails
        pa_perf = (round(((self.cumulative.iloc[-1]) ** (1 / nb_years) - 1) * 100, 1))
        info['CAGR'] = str(round(pa_perf, 1)) + ' %'
        explain['CAGR'] = 'Average annual growth rate'

        sharpe = self.returns.mean() / self.returns.std()
        info['Sharpe ratio'] = round(sharpe * np.sqrt(12), 2)
        explain['Sharpe ratio'] = 'Risk-adjusted return'

        running_max = self.cumulative.cummax()
        drawdown = (self.cumulative - running_max) / running_max
        info['Max drawdown'] = str(round(drawdown.min() * 100, 1)) + ' %'
        info['Avg drawdown'] = str(round(drawdown.mean() * 100, 1)) + ' %'
        explain['Max drawdown'] = 'Largest peak-to-trough loss'
        explain['Avg drawdown'] = 'Typical loss during downturns'

        label = 'BTC-USD' if self.portfolio.crypto else 'SPY'
        spy = self.portfolio.data.benchmarks[label].pct_change().dropna()[self.cutoff - 1:]
        beta = self.returns.cov(spy) / spy.var()
        info['Beta (Stocks)'] = round(beta, 2)
        explain['Beta (Stocks)'] = 'Sensitivity to stock market movements'

        vol = self.returns.std() * np.sqrt(12)
        info['Volatility'] = round(vol, 2)
        explain['Volatility'] = 'Return fluctuations (risk)'

        var95 = np.percentile(self.returns, (1 - .95) * 100)
        info['VaR 95%'] = str(round(var95 * 100, 1)) + ' %'
        explain['VaR 95%'] = 'Max expected loss at 95% confidence'

        X = sm.add_constant(spy)
        model = sm.OLS(self.returns, X).fit()
        r2 = model.rsquared
        info['R2 (Stocks)'] = str(round(100 * r2)) + ' %'
        explain['R2 (Stocks)'] = '% of returns explained by stock benchmark'

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
