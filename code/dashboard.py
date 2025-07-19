from datetime import datetime

import dash
import numpy as np
from dash import html, dcc, Input, Output, ctx, State
from dash.dependencies import ALL

from backtest import Backtest
from portfolio import Portfolio
from opti import Opti
import datetime


class Dashboard(dash.Dash):

    def __init__(self, static=False):

        super().__init__()
        self.static = static

        self.layout_functions = [
            Dashboard.text_title, Dashboard.radio_risk, Dashboard.radio_currency, Dashboard.radio_short,
            Dashboard.input_cash, Dashboard.button_holdings, Dashboard.button_create_portfolio,
            Dashboard.button_rebalance, Dashboard.button_create_backtest
                                 ]

        self.main_div = None
        self.risk, self.currency, self.allow_short, self.cash_sgd, self.holdings = None, None, None, None, None
        self.portfolio, self.opti, self.backtest = None, None, None
        self.get_layout()
        self.callbacks()


    def get_layout(self):

        self.main_div = []
        for func in self.layout_functions:
            self.main_div.extend(func())

        self.layout = html.Div(self.main_div)


    @staticmethod
    def text_title():
        return [html.H1("ETF Portfolio Optimization")]


    @staticmethod
    def radio_risk():
        return [
            html.H4("Select risk level:"),
            dcc.Input(
                id='risk-input',
                type='number',
                value=5,
                min=0,
                max=10,
                step=1
            )
        ]


    @staticmethod
    def radio_currency():
        return [html.H4("Select currency:"),
                dcc.Dropdown(
                    id='radio-currency',
                    options=[{'label': x, 'value': x} for x in ['SGD', 'EUR', 'USD']],
                    value='USD',
                    clearable=False,
                    style={'width': '100px'}
                )]


    @staticmethod
    def radio_short():
        return [
            dcc.Checklist(
                id='switch-short',
                options=[{'label': 'Allow Short', 'value': 'short'}],
                value=[],  # empty list means unchecked
                inputStyle={"margin-right": "5px"},
            )
        ]


    @staticmethod
    def input_cash():
        return [html.H4(id='cash-label'),
                dcc.Input(id='cash', type='number', value=100, step='any')]


    @staticmethod
    def button_holdings():
        return [
            html.H4("Current Holdings:"),
            html.Button("Add Holding", id='button-holdings', n_clicks=0),
            html.Div(id='holdings-container', children=[])
        ]


    @staticmethod
    def button_create_portfolio():
        return [html.H4("Optimal Portfolio:"),
                html.Button("Create Portfolio", id='create-portfolio', n_clicks=0),
                html.Div(id='portfolio-distrib')]


    @staticmethod
    def button_rebalance():
        return [html.H4("Portfolio Rebalancing:"),
                html.Button("Rebalance", id='rebalance-button', n_clicks=0),
                html.Div(id='rebalance-div')]


    @staticmethod
    def button_create_backtest():
        return [html.H4("Backtest:"),
                html.Button("Launch Backtest", id='create-backtest', n_clicks=0),
                html.Div(id='backtest-graphs')]


    def callbacks(self):


        @self.callback(
            Input('risk-input', 'value'),
            Input('radio-currency', 'value'),
            Input('switch-short', 'value'),
            Input('cash', 'value'),
            Input({'type': 'ticker-input', 'index': ALL}, 'value'),
            Input({'type': 'value-input', 'index': ALL}, 'value'),
        )
        def input_callbacks(risk, currency, allow_short, cash_sgd, holdings_tickers, holdings_values):
            self.risk = risk
            self.currency = currency
            self.allow_short = allow_short
            self.cash_sgd = cash_sgd
            self.holdings = {ticker: value for ticker, value in zip(holdings_tickers, holdings_values)}


        @self.callback(
            Output('cash-label', 'children'),
            Input('radio-currency', 'value')
        )
        def update_cash_label(selected_currency):
            return f"Input Cash (in {selected_currency})"


        @self.callback(
            Output('holdings-container', 'children'),
            Input('button-holdings', 'n_clicks'),
            Input('radio-currency', 'value'),
            State({'type': 'ticker-input', 'index': ALL}, 'value'),
            State({'type': 'value-input', 'index': ALL}, 'value'),
        )
        def update_holdings(n_clicks, currency, tickers, values):
            holdings = []
            for i in range(n_clicks):
                ticker_val = tickers[i] if i < len(tickers) else ''
                value_val = values[i] if i < len(values) else ''
                holdings.append(
                    html.Div([
                        dcc.Input(id={'type': 'ticker-input', 'index': i}, type='text', placeholder='Ticker',
                                  value=ticker_val),
                        dcc.Input(id={'type': 'value-input', 'index': i}, type='number',
                                  placeholder=f'Value (in {currency})', step='any', value=value_val)
                    ])
                )
            return holdings


        @self.callback(
            Output('create-portfolio', 'n_clicks'),
            Output('portfolio-distrib', 'children'),
            Input('create-portfolio', 'n_clicks'),
        )
        def create_portfolio(create_portfolio_n_click):
            if create_portfolio_n_click:
                self.portfolio = Portfolio(self.risk, self.cash_sgd, self.holdings, self.currency, self.allow_short, static=self.static)
                self.opti = Opti(self.portfolio)
                return 0, html.Div([
                    self.opti.plot_optimum(),
                    self.opti.plot_in_sample(),
                    self.opti.plot_weighted_perf()
                ])
            return 0, dash.no_update


        @self.callback(
            Output('create-backtest', 'n_clicks'),
            Output('backtest-graphs', 'children'),
            Input('create-backtest', 'n_clicks'),
        )
        def create_backtest(create_backtest_n_click):
            if create_backtest_n_click:
                self.backtest = Backtest(self.opti)
                return 0, html.Div([
                    self.backtest.plot_weights(),
                    self.backtest.plot_backtest(),
                    self.backtest.plot_perf_attrib()
                ])
            return 0, dash.no_update


        @self.callback(
            Output('rebalance-button', 'n_clicks'),
            Output('rebalance-div', 'children'),
            Input('rebalance-button', 'n_clicks'),
        )
        def rebalance(rebalance_n_click):
            if rebalance_n_click:
                return 0, html.Div([

                ])
            return 0, dash.no_update



