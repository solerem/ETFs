"""
Dash application to build, optimize, backtest, and rebalance an ETF portfolio.

This module wires together the core components:

* :class:`portfolio.Portfolio` — data preparation, universe pruning, objective.
* :class:`opti.Opti` — portfolio optimization and performance/diagnostic plots.
* :class:`rebalancer.Rebalancer` — converts optimal weights to a rebalance plan.
* :class:`backtest.Backtest` — rolling walk-forward re-optimization backtest.
* :class:`exposure.Exposure` — exposure breakdown charts.

The :class:`Dashboard` class builds a small UI with:
- Risk, currency, and shorting controls,
- Cash and current holdings inputs,
- Buttons to create an optimal portfolio, show exposures, rebalance, and run a backtest,
- A "crypto Sharpe" helper that shows tangency-portfolio weights for a small crypto universe.

Notes
-----
* The app uses multiple Dash callbacks. Each callback resets its triggering
  button's ``n_clicks`` to ``0`` after use to allow retriggering.
* Images are returned as Dash-ready components via :meth:`opti.Opti.save_fig_as_dash_img`.
"""

import dash
import pandas as pd
from dash import html, dcc, Input, Output, State
from dash.dependencies import ALL
from dash import dash_table
from data import Data
from backtest import Backtest
from rebalancer import Rebalancer
from portfolio import Portfolio
from exposure import Exposure
from opti import Opti


class Dashboard(dash.Dash):
    """
    Minimal Dash UI for ETF portfolio workflows.

    Parameters
    ----------
    static : bool, optional
        If ``True``, downstream components load cached CSVs instead of
        downloading fresh data (default: ``False``).

    Attributes
    ----------
    static : bool
        Passed to :class:`portfolio.Portfolio` / :class:`data.Data`.
    layout_functions : list[callable]
        Functions that return UI chunks used to compose the main layout.
    main_div : list | None
        Flat list of components used as children for the top-level ``html.Div``.
    risk : int | None
        Risk level from the UI.
    currency : str | None
        Selected base currency from the UI.
    allow_short : list[str] | None
        Checklist values; empty list means no shorting (long-only).
    cash_sgd : float | None
        Cash input (denominated in the selected currency).
    holdings : dict[str, float] | None
        Mapping of user-provided holdings (ticker -> value in base currency).
    portfolio : Portfolio | None
        Portfolio object created after clicking "Create Portfolio".
    opti : Opti | None
        Optimizer object tied to :attr:`portfolio`.
    backtest : Backtest | None
        Backtest object created after clicking "Launch Backtest".
    rebalancer : Rebalancer | None
        Rebalancer object created after clicking "Rebalance".
    exposure : Exposure | None
        Exposure object created after clicking "Display exposure".
    """

    def __init__(self, static=False):
        """
        Construct the Dash app, build the layout, and register callbacks.

        :param static: Use cached data instead of fetching if ``True``.
        :type static: bool
        :returns: ``None``.
        :rtype: None
        """
        super().__init__()
        self.static = static

        self.layout_functions = [
            Dashboard.text_title, Dashboard.radio_risk, Dashboard.radio_currency, Dashboard.radio_short,
            Dashboard.input_cash, Dashboard.button_holdings, Dashboard.button_create_portfolio,
            Dashboard.button_rebalance, Dashboard.button_display_exposure, Dashboard.button_create_backtest,
            Dashboard.button_crypto
        ]

        self.main_div = None
        self.risk, self.currency, self.allow_short, self.cash_sgd, self.holdings = None, None, None, None, None
        self.portfolio, self.opti, self.backtest, self.rebalancer, self.exposure = None, None, None, None, None
        self.get_layout()
        self.callbacks()

    def get_layout(self):
        """
        Compose the static layout sections from :attr:`layout_functions`.

        :returns: ``None`` (sets :attr:`layout`).
        :rtype: None
        """
        self.main_div = []
        for func in self.layout_functions:
            self.main_div.extend(func())

        self.layout = html.Div(self.main_div)

    @staticmethod
    def text_title():
        """
        Create the app title.

        :returns: Title component list.
        :rtype: list[dash.html.H1]
        """
        return [html.H1('ETF Portfolio Optimization')]

    @staticmethod
    def radio_risk():
        """
        Numeric input for risk level.

        :returns: A label and numeric input for risk.
        :rtype: list[dash.development.base_component.Component]
        """
        return [
            html.H4('Select risk level:'),
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
        """
        Dropdown for base currency selection.

        :returns: H4 label and currency dropdown.
        :rtype: list[dash.development.base_component.Component]
        """
        return [html.H4('Select currency:'),
                dcc.Dropdown(
                    id='radio-currency',
                    options=[{'label': x, 'value': x} for x in Data.possible_currencies],
                    value='USD',
                    clearable=False,
                    style={'width': '100px'}
                )]

    @staticmethod
    def radio_short():
        """
        Checklist to allow/disallow shorting.

        :returns: Checklist component; empty value implies long-only.
        :rtype: list[dash.development.base_component.Component]
        """
        return [
            dcc.Checklist(
                id='switch-short',
                options=[{'label': 'Allow Short', 'value': 'short'}],
                value=[],  # empty list means unchecked
                inputStyle={'margin-right': '5px'},
            )
        ]

    @staticmethod
    def input_cash():
        """
        Cash input whose label reflects the selected currency.

        :returns: Label and numeric input for cash.
        :rtype: list[dash.development.base_component.Component]
        """
        return [html.H4(id='cash-label'),
                dcc.Input(id='cash', type='number', value=100, step='any')]

    @staticmethod
    def button_holdings():
        """
        Holdings input section with an "Add Holding" button.

        :returns: Header, button, and container div.
        :rtype: list[dash.development.base_component.Component]
        """
        return [
            html.H4('Current Holdings:'),
            html.Button('Add Holding', id='button-holdings', n_clicks=0),
            html.Div(id='holdings-container', children=[])
        ]

    @staticmethod
    def button_create_portfolio():
        """
        Section to create and display an optimal portfolio.

        :returns: Header, button, and a loading wrapper for result graphs.
        :rtype: list[dash.development.base_component.Component]
        """
        return [html.H4('Optimal Portfolio:'),
                html.Button('Create Portfolio', id='create-portfolio', n_clicks=0),
                dcc.Loading(
                    id='loading-portfolio',
                    type='default',
                    children=html.Div(id='portfolio-distrib'))]

    @staticmethod
    def button_rebalance():
        """
        Section to build and display a rebalance table.

        :returns: Header, button, and result container.
        :rtype: list[dash.development.base_component.Component]
        """
        return [html.H4('Portfolio Rebalancing:'),
                html.Button('Rebalance', id='rebalance-button', n_clicks=0),
                html.Div(id='rebalance-div')]

    @staticmethod
    def button_display_exposure():
        """
        Section to render exposure breakdown charts.

        :returns: Header, button, and graph container.
        :rtype: list[dash.development.base_component.Component]
        """
        return [html.H4('Exposure:'),
                html.Button('Display exposure', id='display-exposure', n_clicks=0),
                html.Div(id='exposure-graphs')]

    @staticmethod
    def button_create_backtest():
        """
        Section to run and display a rolling backtest.

        :returns: Header, button, and loading wrapper for backtest graphs.
        :rtype: list[dash.development.base_component.Component]
        """
        return [html.H4('Backtest:'),
                html.Button('Launch Backtest', id='create-backtest', n_clicks=0),
                dcc.Loading(
                    id='loading-backtest',
                    type='default',
                    children=html.Div(id='backtest-graphs')
                )]

    @staticmethod
    def button_crypto():
        """
        Section to show crypto tangency-portfolio weights.

        :returns: Header, button, and table container.
        :rtype: list[dash.development.base_component.Component]
        """
        return [html.H4('Cryptos (beta):'),
                html.Button('Get crypto sharpe', id='crypto-sharpe', n_clicks=0),
                html.Div(id='crypto-opti')]

    def callbacks(self):
        """
        Register all Dash callbacks (inputs, buttons, and renderers).

        Each nested function has its own docstring describing inputs/outputs.

        :returns: ``None``.
        :rtype: None
        """

        @self.callback(
            Input('risk-input', 'value'),
            Input('radio-currency', 'value'),
            Input('switch-short', 'value'),
            Input('cash', 'value'),
            Input({'type': 'ticker-input', 'index': ALL}, 'value'),
            Input({'type': 'value-input', 'index': ALL}, 'value'),
        )
        def input_callbacks(risk, currency, allow_short, cash_sgd, holdings_tickers, holdings_values):
            """
            Capture form inputs and cache them on the instance.

            :param risk: Risk level numeric value.
            :type risk: int | float | None
            :param currency: Selected base currency (e.g., ``'USD'``).
            :type currency: str | None
            :param allow_short: Checklist values; empty list means long-only.
            :type allow_short: list[str]
            :param cash_sgd: Cash amount in selected currency.
            :type cash_sgd: float | int | None
            :param holdings_tickers: List of ticker strings from dynamic fields.
            :type holdings_tickers: list[str] | None
            :param holdings_values: List of numeric values aligned to tickers.
            :type holdings_values: list[float] | None
            :returns: ``None`` (state is stored on ``self``).
            :rtype: None
            """
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
            """
            Update the "Input Cash" label with the current currency.

            :param selected_currency: Newly selected currency code.
            :type selected_currency: str
            :returns: Label text.
            :rtype: str
            """
            return f'Input Cash (in {selected_currency})'

        @self.callback(
            Output('holdings-container', 'children'),
            Input('button-holdings', 'n_clicks'),
            Input('radio-currency', 'value'),
            State({'type': 'ticker-input', 'index': ALL}, 'value'),
            State({'type': 'value-input', 'index': ALL}, 'value'),
        )
        def update_holdings(n_clicks, currency, tickers, values):
            """
            Render the dynamic list of holding rows.

            :param n_clicks: Count of "Add Holding" button clicks.
            :type n_clicks: int
            :param currency: Base currency for the value placeholder.
            :type currency: str
            :param tickers: Existing ticker field values (by index).
            :type tickers: list[str] | None
            :param values: Existing value field entries (by index).
            :type values: list[float] | None
            :returns: List of holding row components.
            :rtype: list[dash.development.base_component.Component]
            """
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
            """
            Build an optimal portfolio and render its graphs.

            Resets the button's ``n_clicks`` to 0 after rendering so the user
            can click again.

            :param create_portfolio_n_click: Click counter for the button.
            :type create_portfolio_n_click: int
            :returns: Tuple of (button reset, graphs container or no update).
            :rtype: tuple[int, dash.development.base_component.Component]
            """
            if create_portfolio_n_click:
                self.portfolio = Portfolio(self.risk, self.cash_sgd, self.holdings, self.currency, self.allow_short,
                                           static=self.static)
                self.opti = Opti(self.portfolio)
                return 0, html.Div([
                    self.opti.plot_in_sample(),
                    self.opti.plot_optimum(),
                    self.opti.plot_weighted_perf(),
                    self.opti.plot_drawdown()
                ])
            return 0, dash.no_update

        @self.callback(
            Output('create-backtest', 'n_clicks'),
            Output('backtest-graphs', 'children'),
            Input('create-backtest', 'n_clicks'),
            prevent_initial_call=True
        )
        def create_backtest(create_backtest_n_click):
            """
            Run a rolling backtest and render its graphs.

            :param create_backtest_n_click: Click counter for the backtest button.
            :type create_backtest_n_click: int
            :returns: Tuple of (button reset, graphs container or no update).
            :rtype: tuple[int, dash.development.base_component.Component]
            """
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
            Input('radio-currency', 'value')
        )
        def rebalance(rebalance_n_click, selected_currency):
            """
            Compute a rebalance plan and show it as a table.

            :param rebalance_n_click: Click counter for the rebalance button.
            :type rebalance_n_click: int
            :param selected_currency: Currency code to annotate the table column.
            :type selected_currency: str
            :returns: Tuple of (button reset, table container or no update).
            :rtype: tuple[int, dash.development.base_component.Component]
            """
            if rebalance_n_click:
                self.rebalancer = Rebalancer(self.opti)
                df = self.rebalancer.rebalance_df
                df.rename(columns={'Buy/Sell': f'Buy/Sell ({selected_currency})'}, inplace=True)

                return 0, html.Div([
                    dash_table.DataTable(
                        data=df.to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in df.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        page_size=10
                    )
                ], style={'width': '40%'})
            return 0, dash.no_update

        @self.callback(
            Output('display-exposure', 'n_clicks'),
            Output('exposure-graphs', 'children'),
            Input('display-exposure', 'n_clicks'),
        )
        def display_exposure(display_exposure_n_click):
            """
            Render exposure breakdown charts for the optimized portfolio.

            :param display_exposure_n_click: Click counter for the exposure button.
            :type display_exposure_n_click: int
            :returns: Tuple of (button reset, graphs container or no update).
            :rtype: tuple[int, dash.development.base_component.Component]
            """
            if display_exposure_n_click:
                self.exposure = Exposure(self.opti)
                return 0, html.Div([
                    self.exposure.plot_currency(),
                    self.exposure.plot_category(),
                    self.exposure.plot_sector(),
                    self.exposure.plot_type(),
                    self.exposure.plot_geo()
                ])
            return 0, dash.no_update

        @self.callback(
            Output('crypto-sharpe', 'n_clicks'),
            Output('crypto-opti', 'children'),
            Input('crypto-sharpe', 'n_clicks'),
        )
        def crypto_sharpe(crypto_sharpe_n_click):
            """
            Display crypto tangency-portfolio weights as a table.

            :param crypto_sharpe_n_click: Click counter for the crypto button.
            :type crypto_sharpe_n_click: int
            :returns: Tuple of (button reset, table container or no update).
            :rtype: tuple[int, dash.development.base_component.Component]
            """
            if crypto_sharpe_n_click:
                df = (pd.Series(self.portfolio.crypto_opti, name='Weight').rename_axis('Ticker').reset_index())
                df = df[df['Weight'] != 0]
                df['Weight'] = [f'{x}%' for x in df['Weight']]

                return 0, html.Div([
                    dash_table.DataTable(
                        data=df.to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in df.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        page_size=10
                    )
                ], style={'width': '15%'})
            return 0, dash.no_update
