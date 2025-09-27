"""
Interactive Dash application for ETF portfolio optimization.

This module defines :class:`Dashboard`, a Dash app that lets a user:

* Choose a risk level, base currency, and current holdings.
* Construct an optimal portfolio using :class:`portfolio.Portfolio` and
  :class:`opti.Opti`.
* View in-sample diagnostics (allocation pie, equity curve, drawdown, attribution).
* Generate a rolling walk-forward backtest via :class:`backtest.Backtest`.
* Produce a rebalance table with buy/sell amounts via :class:`rebalancer.Rebalancer`.
* Visualize exposure breakdowns via :class:`exposure.Exposure`.
* Inspect a simple crypto Sharpe-optimized allocation (beta).

The UI uses Bootstrap (via ``dash-bootstrap-components``) and arranges charts
in a responsive 2-up grid on larger screens (stacking on small screens).
"""

import dash
import pandas as pd
from dash import html, dcc, Input, Output, State
from dash.dependencies import ALL
from dash import dash_table

import dash_bootstrap_components as dbc

from data import Data
from backtest import Backtest
from rebalancer import Rebalancer
from portfolio import Portfolio
from exposure import Exposure
from opti import Opti


class Dashboard(dash.Dash):
    """
    Dash application shell for ETF portfolio workflows.

    Parameters
    ----------
    static : bool, optional
        If ``True``, child components that support caching will read pre-downloaded
        CSVs instead of hitting the network (passed along to :class:`data.Data`
        via :class:`portfolio.Portfolio`). Default is ``False``.

    Attributes
    ----------
    static : bool
        Whether to operate in cached/static mode for the underlying data loaders.
    risk : int | None
        Selected risk level from the slider (0–10).
    currency : str | None
        Selected base currency (one of :data:`data.Data.possible_currencies`).
    allow_short : bool
        Whether shorting is allowed (exposed to :class:`portfolio.Portfolio`).
    cash_sgd : float | None
        Cash value input (interpreted in the selected base currency).
    holdings : dict[str, float] | None
        Current holdings mapping entered by the user.
    portfolio : Portfolio | None
        Last constructed portfolio.
    opti : Opti | None
        Optimizer wrapper for the last constructed portfolio.
    backtest : Backtest | None
        Backtest wrapper computed from :attr:`opti`.
    rebalancer : Rebalancer | None
        Rebalancing helper built from :attr:`opti`.
    exposure : Exposure | None
        Exposure visualization helper built from :attr:`opti`.
    """

    def __init__(self, static=False):
        """
        Construct the Dash app, inject CSS, build layout, and register callbacks.

        :param static: If ``True``, downstream loaders use cached CSVs.
        :type static: bool
        :returns: ``None``.
        :rtype: None
        """
        super().__init__(external_stylesheets=[dbc.themes.LUX, dbc.icons.BOOTSTRAP])
        self.index_string = self.index_string.replace(
            "</head>",
            """
            <style>
              @media (min-width: 992px){ .sticky-card{ position: sticky; top: 1rem; } }
              .chart-frame{ max-width:100%; max-height:60vh; overflow:auto; }
              .chart-frame img, .chart-frame svg, .chart-frame canvas{
                display:block; max-width:100%; height:auto !important;
              }
              .chart-frame .dash-graph{ height:420px !important; }

              /* NEW: 2-up grid for charts */
              .grid-2{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:1rem; align-items:start; }
              @media (max-width: 991.98px){ .grid-2{ grid-template-columns: 1fr; } } /* stack on small screens */
            </style>
            </head>
            """
        )

        self.static = static

        self.main_div = None
        self.risk, self.currency, self.allow_short, self.cash_sgd, self.holdings, self.rates = None, None, None, None, None, None
        self.portfolio, self.opti, self.backtest, self.rebalancer, self.exposure = None, None, None, None, None
        self.allow_short = False
        self.mode = 'etf'  # 'etf' or 'crypto'

        self.get_layout()
        self.callbacks()


    def get_layout(self):
        """
        Define and assign the top-level Dash layout.

        The layout includes:
        * A Bootstrap navbar,
        * A left sidebar with controls (risk, currency, holdings),
        * A right content area with tabs for portfolio, rebalance, exposure, backtest, and crypto.

        :returns: ``None`` (sets :attr:`layout`).
        :rtype: None
        """
        self.layout = html.Div([
            dcc.Store(id='init-store'),

            dbc.Navbar(
                dbc.Container([
                    dbc.NavbarBrand("ETF Portfolio Optimization", className="fw-semibold"),
                    dbc.NavItem(dbc.Badge("beta", color="secondary", className="ms-2"))
                ]),
                color="primary",
                dark=True,
                className="mb-4 shadow-sm"
            ),

            dbc.Container([
                dbc.Row([
                    dbc.Col(self._sidebar_controls(), md=4, lg=3, className="mb-4"),

                    dbc.Col(self._content_area(), md=8, lg=9)
                ], className="g-4")
            ], fluid=True),

            html.Footer(
                dbc.Container(
                    html.Small("Built with Dash + Bootstrap Components", className="text-muted"),
                    className="py-4"
                ),
                className="border-top mt-5"
            )
        ])

    def _sidebar_controls(self):
        """
        Build the left-hand sidebar with inputs and an "Add Holding" control.

        :returns: A Bootstrap card with form inputs.
        :rtype: dash.html.Div
        """
        return dbc.Card([
            dbc.CardHeader(html.Div([
                html.Span(className="bi bi-sliders me-2"),
                html.Span("Controls", className="fw-semibold")
            ])),
            dbc.CardBody([

                # --- NEW: Mode toggle (ETF / Crypto) ---
                dbc.Label("Mode", html_for="mode-toggle", className="fw-semibold"),
                dbc.RadioItems(
                    id='mode-toggle',
                    options=[{'label': 'ETF', 'value': 'etf'},
                             {'label': 'Crypto', 'value': 'crypto'}],
                    value='etf',
                    inline=True,
                    inputClassName="me-1", labelClassName="me-3",
                ),
                html.Div(className="mb-3"),

                # Risk controls wrapped so we can show/hide for crypto mode
                html.Div(id='risk-section', children=[
                    dbc.Label("Risk level", html_for="risk-input", className="fw-semibold"),
                    dcc.Slider(id='risk-input', min=0, max=10, step=1, value=5,
                               marks={i: str(i) for i in range(0, 11)},
                               tooltip={"placement": "bottom", "always_visible": False}),
                    html.Div(className="mb-3"),
                ]),

                dbc.Label("Base currency", html_for="radio-currency", className="fw-semibold"),
                dcc.Dropdown(
                    id='radio-currency',
                    options=[{'label': x, 'value': x} for x in Data.possible_currencies],
                    value='USD',
                    clearable=False
                ),
                html.Div(className="mb-3"),

                dbc.Label(id='cash-label', className="fw-semibold"),
                dbc.Input(id='cash', type='number', value=100, step='any', placeholder="Enter cash amount"),
                html.Div(className="mb-4"),

                html.Div([
                    html.Div(className="d-flex align-items-center mb-2", children=[
                        html.Div(className="fw-semibold me-auto", children="Current Holdings"),
                        dbc.Button([html.I(className="bi bi-plus-lg me-1"), "Add Holding"],
                                   id='button-holdings', n_clicks=0, color="secondary", size="sm", outline=True)
                    ]),
                    html.Div(id='holdings-container', children=[], className="vstack gap-2")
                ]),

                # Savings Rates section wrapped so we can show/hide for crypto mode
                html.Div(id='rates-section', children=[
                    html.Div(className="d-flex align-items-center mb-2", children=[
                        html.Div(className="fw-semibold me-auto", children="Savings Rates"),
                        dbc.Button([html.I(className="bi bi-plus-lg me-1"), "Add Rate"],
                                   id='button-rates', n_clicks=0, color="secondary", size="sm", outline=True)
                    ]),
                    html.Div(id='rates-container', children=[], className="vstack gap-2")
                ])
            ])
        ], className="shadow-sm sticky-card")

    def _content_area(self):
        """
        Build the right-hand tabbed content area (workspace).

        Tabs:
        * Optimal Portfolio — constructs and visualizes the optimized weights.
        * Rebalance — displays a rebalance table. (hidden in Crypto mode)
        * Exposure — shows currency/class/sector/type/geo pie charts.
        * Backtest — runs a rolling re-optimization backtest.

        :returns: A Bootstrap card containing Dash tabs and placeholders.
        :rtype: dash.html.Div
        """
        return dbc.Card([
            dbc.CardHeader(
                html.Div([html.Span(className="bi bi-graph-up-arrow me-2"),
                          html.Span("Workspace", className="fw-semibold")])
            ),
            dbc.CardBody([
                dcc.Tabs(id="main-tabs", value="tab-portfolio", children=[
                    dcc.Tab(label="Optimal Portfolio", value="tab-portfolio", children=[
                        html.Div(className="d-flex align-items-center mb-3", children=[
                            dbc.Button([html.I(className="bi bi-magic me-2"), "Create Portfolio"],
                                       id='create-portfolio', n_clicks=0, color="primary"),
                            html.Span("  — computes optimal weights and charts", className="text-muted ms-2")
                        ]),
                        dbc.Spinner(html.Div(id='portfolio-distrib'), size="md")
                    ]),

                    dcc.Tab(label="Rebalance", value="tab-rebalance", children=[
                        html.Div(className="d-flex align-items-center mb-3", children=[
                            dbc.Button([html.I(className="bi bi-arrow-repeat me-2"), "Rebalance"],
                                       id='rebalance-button', n_clicks=0, color="primary")
                        ]),
                        html.Div(id='rebalance-div', className="table-wrap")
                    ]),

                    dcc.Tab(label="Exposure", value="tab-exposure", id='exposure-tab', children=[
                        html.Div(className="d-flex align-items-center mb-3", children=[
                            dbc.Button([html.I(className="bi bi-pie-chart me-2"), "Display exposure"],
                                       id='display-exposure', n_clicks=0, color="primary")
                        ]),
                        html.Div(id='exposure-graphs')
                    ]),

                    dcc.Tab(label="Backtest", value="tab-backtest", children=[
                        html.Div(className="d-flex align-items-center mb-3", children=[
                            dbc.Button([html.I(className="bi bi-play-fill me-2"), "Launch Backtest"],
                                       id='create-backtest', n_clicks=0, color="primary"),
                            html.Span("  — rolling walk-forward re-optimization", className="text-muted ms-2")
                        ]),
                        dbc.Spinner(html.Div(id='backtest-graphs'), size="md")
                    ]),
                ])
            ])
        ], className="shadow-sm")

    def callbacks(self):
        """
        Register all Dash callbacks for interactivity.

        This method defines several nested callback functions bound to UI events:
        input synchronization, dynamic labels and holdings rows, portfolio
        creation, backtesting, rebalancing, exposure plotting.

        :returns: ``None``.
        :rtype: None
        """

        @self.callback(
            Output('init-store', 'data'),
            Input('mode-toggle', 'value'),
            Input('risk-input', 'value'),
            Input('radio-currency', 'value'),
            Input('cash', 'value'),
            Input({'type': 'holdings-ticker-input', 'index': ALL}, 'value'),
            Input({'type': 'holdings-value-input', 'index': ALL}, 'value'),
            Input({'type': 'rates-ticker-input', 'index': ALL}, 'value'),
            Input({'type': 'rates-value-input', 'index': ALL}, 'value'),
        )
        def input_callbacks(mode, risk, currency, cash_sgd, holdings_tickers, holdings_values, rates_tickers,
                            rates_values):
            """
            Synchronize sidebar inputs into instance attributes.

            :param mode: Selected mode ('etf' or 'crypto').
            :type mode: str
            :param risk: Risk slider value.
            :type risk: int
            :param currency: Selected base currency.
            :type currency: str
            :param cash_sgd: Input cash in base currency units.
            :type cash_sgd: float
            :param holdings_tickers: List of tickers entered in dynamic rows.
            :type holdings_tickers: list[str] | None
            :param holdings_values: List of values entered in dynamic rows.
            :type holdings_values: list[float] | None
            :returns: Dummy value for the dcc.Store (unused).
            :rtype: int
            """
            self.mode = mode or 'etf'
            self.risk = risk
            self.currency = currency
            self.cash_sgd = cash_sgd
            self.holdings = {ticker: value for ticker, value in zip(holdings_tickers, holdings_values)}
            # Ignore savings rates in crypto mode
            self.rates = {} if self.mode == 'crypto' else {ticker: value for ticker, value in
                                                           zip(rates_tickers, rates_values)}
            return 0

        @self.callback(
            Output('cash-label', 'children'),
            Input('radio-currency', 'value')
        )
        def update_cash_label(selected_currency):
            """
            Update the cash input label with the selected currency.

            :param selected_currency: Current base currency from dropdown.
            :type selected_currency: str
            :returns: Updated label text.
            :rtype: str
            """
            return f'Input Cash (in {selected_currency})'

        @self.callback(
            Output('holdings-container', 'children'),
            Input('button-holdings', 'n_clicks'),
            State({'type': 'holdings-ticker-input', 'index': ALL}, 'value'),
            State({'type': 'holdings-value-input', 'index': ALL}, 'value'),
        )
        def update_holdings(n_clicks, tickers, values):
            """
            Grow the list of holdings input rows and preserve entered values.

            :param n_clicks: Number of times "Add Holding" has been clicked.
            :type n_clicks: int
            :param currency: Current base currency (for value field label).
            :type currency: str
            :param tickers: Existing ticker values from state.
            :type tickers: list[str] | None
            :param values: Existing numeric values from state.
            :type values: list[float] | None
            :returns: List of InputGroup rows for the holdings container.
            :rtype: list[dash.html.Div]
            """
            holdings = []
            tickers = tickers or []
            values = values or []
            for i in range(n_clicks):
                ticker_val = tickers[i] if i < len(tickers) else ''
                value_val = values[i] if i < len(values) else ''
                holdings.append(
                    dbc.InputGroup([
                        dbc.InputGroupText("Ticker"),
                        dbc.Input(id={'type': 'holdings-ticker-input', 'index': i}, type='text', placeholder='eg SPX',
                                  value=ticker_val),
                        dbc.InputGroupText(f"Value"),
                        dbc.Input(id={'type': 'holdings-value-input', 'index': i}, type='number',
                                  placeholder='0', step=1, value=value_val)
                    ])
                )
            return holdings

        @self.callback(
            Output('rates-container', 'children'),
            Input('button-rates', 'n_clicks'),
            State({'type': 'rates-ticker-input', 'index': ALL}, 'value'),
            State({'type': 'rates-value-input', 'index': ALL}, 'value'),
        )
        def update_rates(n_clicks, tickers, values):
            """
            Grow the list of holdings input rows and preserve entered values.

            :param n_clicks: Number of times "Add Holding" has been clicked.
            :type n_clicks: int
            :param currency: Current base currency (for value field label).
            :type currency: str
            :param tickers: Existing ticker values from state.
            :type tickers: list[str] | None
            :param values: Existing numeric values from state.
            :type values: list[float] | None
            :returns: List of InputGroup rows for the holdings container.
            :rtype: list[dash.html.Div]
            """
            holdings = []
            tickers = tickers or []
            values = values or []
            for i in range(n_clicks):
                ticker_val = tickers[i] if i < len(tickers) else ''
                value_val = values[i] if i < len(values) else ''
                holdings.append(
                    dbc.InputGroup([
                        dbc.InputGroupText("Ticker"),
                        dbc.Input(id={'type': 'rates-ticker-input', 'index': i}, type='text', placeholder='eg EUR',
                                  value=ticker_val),
                        dbc.InputGroupText(f"Value"),
                        dbc.Input(id={'type': 'rates-value-input', 'index': i}, type='number',
                                  placeholder='0.00', step=.01, value=value_val)
                    ])
                )
            return holdings

        @self.callback(
            Output('risk-section', 'style'),
            Output('rates-section', 'style'),
            Output('exposure-tab', 'style'),
            Input('mode-toggle', 'value')
        )
        def toggle_sections(mode):
            hide = {'display': 'none'}
            show = {}
            is_crypto = (mode == 'crypto')
            return (hide if is_crypto else show,
                    hide if is_crypto else show,
                    hide if is_crypto else show)

        @self.callback(
            Output('create-portfolio', 'n_clicks'),
            Output('portfolio-distrib', 'children'),
            Input('create-portfolio', 'n_clicks'),
        )
        def create_portfolio(create_portfolio_n_click):
            """
            Build the portfolio and show in-sample charts.

            Triggered by the "Create Portfolio" button. Constructs
            :class:`portfolio.Portfolio` and :class:`opti.Opti`, then renders
            four charts (equity curve, allocation pie, drawdown, attribution).

            :param create_portfolio_n_click: Number of button clicks.
            :type create_portfolio_n_click: int
            :returns: Tuple ``(reset_clicks, charts_container)``.
            :rtype: tuple[int, dash.html.Div]
            """
            if create_portfolio_n_click:

                self.portfolio = Portfolio(
                    self.risk,
                    self.cash_sgd,
                    self.holdings,
                    self.currency,
                    self.allow_short,
                    static=self.static,
                    rates=self.rates,
                    crypto=(self.mode == 'crypto')
                )
                self.opti = Opti(self.portfolio)
                return 0, html.Div([
                    html.Div(self.opti.plot_in_sample(), className="chart-frame"),
                    html.Div(self.opti.plot_optimum(), className="chart-frame"),
                    html.Div(self.opti.plot_drawdown(), className="chart-frame"),
                    html.Div(self.opti.plot_weighted_perf(), className="chart-frame"),
                    html.Div(self.opti.plot_info()),
                ], className="grid-2")
            return 0, dash.no_update

        @self.callback(
            Output('create-backtest', 'n_clicks'),
            Output('backtest-graphs', 'children'),
            Input('create-backtest', 'n_clicks'),
            prevent_initial_call=True
        )
        def create_backtest(create_backtest_n_click):
            """
            Launch a rolling walk-forward backtest and display charts.

            :param create_backtest_n_click: Number of button clicks.
            :type create_backtest_n_click: int
            :returns: Tuple ``(reset_clicks, charts_container)`` with weights,
                      equity curve, and attribution.
            :rtype: tuple[int, dash.html.Div | dash.dcc.Loading]
            """
            if create_backtest_n_click:
                self.backtest = Backtest(self.opti)
                return 0, html.Div([
                    html.Div(self.backtest.plot_backtest(), className="chart-frame"),
                    html.Div(self.backtest.plot_weights(), className="chart-frame"),
                    html.Div(self.backtest.plot_drawdown(), className="chart-frame"),
                    html.Div(self.backtest.plot_perf_attrib(), className="chart-frame"),
                    html.Div(self.backtest.plot_info()),
                ], className="grid-2")
            return 0, dash.no_update

        @self.callback(
            Output('rebalance-button', 'n_clicks'),
            Output('rebalance-div', 'children'),
            Input('rebalance-button', 'n_clicks'),
            Input('radio-currency', 'value')
        )
        def rebalance(rebalance_n_click, selected_currency):
            """
            Compute and render the rebalance table.

            :param rebalance_n_click: Number of "Rebalance" button clicks.
            :type rebalance_n_click: int
            :param selected_currency: Current base currency for column labeling.
            :type selected_currency: str
            :returns: Tuple ``(reset_clicks, table_card)``.
            :rtype: tuple[int, dash.html.Div]
            """
            if rebalance_n_click:
                self.rebalancer = Rebalancer(self.opti)
                df = self.rebalancer.rebalance_df.copy()
                df.rename(columns={'Buy/Sell': f'Buy/Sell ({selected_currency})'}, inplace=True)

                table = dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.columns],
                    page_size=10,
                    sort_action='native',
                    style_table={'overflowX': 'auto'},
                    style_as_list_view=True,
                    style_header={
                        'fontWeight': '600',
                        'border': 'none'
                    },
                    style_cell={
                        'padding': '10px',
                        'border': 'none'
                    },
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgba(0,0,0,0.02)'}
                    ]
                )

                return 0, dbc.Card(dbc.CardBody(table), className="shadow-sm")
            return 0, dash.no_update

        @self.callback(
            Output('display-exposure', 'n_clicks'),
            Output('exposure-graphs', 'children'),
            Input('display-exposure', 'n_clicks'),
        )
        def display_exposure(display_exposure_n_click):
            """
            Render currency/class/sector/type/geo exposure pie charts.

            :param display_exposure_n_click: Number of "Display exposure" clicks.
            :type display_exposure_n_click: int
            :returns: Tuple ``(reset_clicks, charts_container)`` of pie charts.
            :rtype: tuple[int, dash.html.Div]
            """
            if display_exposure_n_click:
                self.exposure = Exposure(self.opti)
                return 0, html.Div([
                    html.Div(self.exposure.plot_currency(), className="chart-frame"),
                    html.Div(self.exposure.plot_category(), className="chart-frame"),
                    html.Div(self.exposure.plot_sector(), className="chart-frame"),
                    html.Div(self.exposure.plot_type(), className="chart-frame"),
                    html.Div(self.exposure.plot_geo(), className="chart-frame"),
                ], className="grid-2")
            return 0, dash.no_update

