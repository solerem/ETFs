"""
Interactive Dash application for ETF or Crypto portfolio optimization.

This module defines :class:`Dashboard`, a Dash app that lets a user:

* Choose a mode (ETF or Crypto), risk level (ETF only), base currency, and current holdings.
* Construct an optimal portfolio using :class:`portfolio.Portfolio` and
  :class:`opti.Opti`.
* View in-sample diagnostics (allocation pie, equity curve, drawdown, attribution).
* Generate a rolling walk-forward backtest via :class:`backtest.Backtest`.
* Produce a rebalance table with buy/sell amounts via :class:`rebalancer.Rebalancer`.
* Visualize exposure breakdowns via :class:`exposure.Exposure`.
* Inspect a simple crypto Sharpe-optimized allocation (beta via ``Portfolio.crypto`` mode).

UI/UX
-----
* Built with Dash + Bootstrap (``dash-bootstrap-components``).
* Responsive “2-up” grid for charts on wide screens; stacks on smaller screens.
* In Crypto mode, the Risk and Savings Rates controls are hidden (Sharpe-style
  objective ignores the mean–variance risk weight and currency savings rates).

Data/Caching
------------
* ``static=True`` enables cached CSVs for loaders inside :class:`portfolio.Portfolio`
  and :class:`data.Data`.
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
    Dash application shell for ETF/Crypto portfolio workflows.

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
    mode : str
        Application mode: ``'etf'`` (default) or ``'crypto'``. Toggles controls and
        objective used by :class:`portfolio.Portfolio`.
    risk : int | None
        Selected risk level from the slider (0–10). Hidden/ignored in Crypto mode.
    currency : str | None
        Selected base currency (one of :data:`data.Data.possible_currencies`).
    allow_short : bool
        Whether shorting is allowed (exposed to :class:`portfolio.Portfolio`).
    cash_sgd : float | None
        Cash value input (interpreted in the selected base currency).
    holdings : dict[str, float] | None
        Current holdings mapping entered by the user.
    rates : dict[str, float] | None
        Optional mapping of currency pseudo-tickers to annualized savings rates (%)—
        ignored in Crypto mode.
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
    main_div : dash.html.Div | None
        Top-level container reference (not used externally).
    """

    def __init__(self):
        """
        Construct the Dash app, inject CSS, build layout, and register callbacks.

        The constructor:
        * Applies Bootstrap theme and lightweight custom CSS (grid, sticky sidebar).
        * Initializes state (mode, risk, currency, holdings, etc.).
        * Builds the sidebar + workspace layout.
        * Registers all interaction callbacks.

        Parameters
        ----------
        static : bool, optional
            If ``True``, downstream loaders use cached CSVs.

        Returns
        -------
        None
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
              .grid-2{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:1rem; align-items:start; }
              @media (max-width: 991.98px){ .grid-2{ grid-template-columns: 1fr; } }
            </style>
            </head>
            """
        )



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
        * A left sidebar with controls (mode, risk, currency, holdings, savings rates),
        * A right content area with tabs for portfolio, rebalance, exposure, and backtest.

        Returns
        -------
        None
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
        Build the left-hand sidebar with inputs for mode, risk, currency, holdings, and rates.

        Returns
        -------
        dash.html.Div
            A Bootstrap card with form inputs.
        """
        return dbc.Card([
            dbc.CardHeader(html.Div([
                html.Span(className="bi bi-sliders me-2"),
                html.Span("Controls", className="fw-semibold")
            ])),
            dbc.CardBody([

                # Mode toggle (ETF / Crypto)
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

                # Risk controls (hidden in Crypto mode)
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

                # Savings Rates (hidden in Crypto mode)
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

        Tabs
        ----
        * Optimal Portfolio — constructs and visualizes the optimized weights.
        * Rebalance — displays a rebalance table (hidden in Crypto mode by callback).
        * Exposure — shows currency/class/sector/type/geo pie charts.
        * Backtest — runs a rolling re-optimization backtest.

        Returns
        -------
        dash.html.Div
            A Bootstrap card containing Dash tabs and placeholders.
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

        Defines the following behaviors:
        * Input synchronization into instance state (mode, risk, currency, holdings, rates).
        * Dynamic labels and row builders for holdings and savings rates.
        * Mode-based show/hide for risk, savings rates, and exposure tab (crypto mode hides them).
        * Portfolio creation: constructs :class:`portfolio.Portfolio` and :class:`opti.Opti`,
          then renders four in-sample charts + a metrics table.
        * Backtest creation: runs :class:`backtest.Backtest` and renders equity, weights, drawdown,
          and attribution + metrics.
        * Rebalance: computes a rebalance table from :class:`rebalancer.Rebalancer`.
        * Exposure: renders five pie charts via :class:`exposure.Exposure`.

        Returns
        -------
        None
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

            Parameters
            ----------
            mode : str
                Selected mode ('etf' or 'crypto').
            risk : int
                Risk slider value (ignored in crypto mode).
            currency : str
                Selected base currency.
            cash_sgd : float
                Cash in base currency units.
            holdings_tickers : list[str] | None
                Tickers entered in dynamic rows.
            holdings_values : list[float] | None
                Values entered in dynamic rows.
            rates_tickers : list[str] | None
                Currency codes for savings rates (ETF mode only).
            rates_values : list[float] | None
                Annualized savings rates in percent (ETF mode only).

            Returns
            -------
            int
                Dummy value for the dcc.Store (unused).
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

            Parameters
            ----------
            selected_currency : str
                Current base currency from dropdown.

            Returns
            -------
            str
                Updated label text.
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

            Parameters
            ----------
            n_clicks : int
                Number of times "Add Holding" has been clicked.
            tickers : list[str] | None
                Existing ticker values from state.
            values : list[float] | None
                Existing numeric values from state.

            Returns
            -------
            list[dash.html.Div]
                Rows for the holdings container.
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
                        dbc.InputGroupText("Value"),
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
            Grow the list of savings-rate input rows and preserve values.

            Parameters
            ----------
            n_clicks : int
                Number of times "Add Rate" has been clicked.
            tickers : list[str] | None
                Existing currency codes.
            values : list[float] | None
                Existing rate values (percent).

            Returns
            -------
            list[dash.html.Div]
                Rows for the rates container.
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
                        dbc.InputGroupText("Value"),
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
            """
            Show/hide controls and tabs based on the selected mode.

            * Crypto mode hides the risk slider, savings rates section, and
              the Exposure tab (exposures are ETF-oriented).
            """
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
            four charts (equity curve, allocation pie, drawdown, attribution)
            and a small metrics table.

            Returns
            -------
            tuple[int, dash.html.Div]
                ``(reset_clicks, charts_container)``.
            """
            if create_portfolio_n_click:

                self.portfolio = Portfolio(
                    self.risk,
                    self.cash_sgd,
                    self.holdings,
                    self.currency,
                    self.allow_short,
                    static=True,
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

            Returns
            -------
            tuple[int, dash.html.Div]
                ``(reset_clicks, charts_container)`` with equity curve,
                weight stack, drawdown, attribution, and a metrics table.
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

            Parameters
            ----------
            rebalance_n_click : int
                Number of "Rebalance" button clicks.
            selected_currency : str
                Current base currency for column labeling.

            Returns
            -------
            tuple[int, dash.html.Div]
                ``(reset_clicks, table_card)`` with a DataTable of trades.
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

            Returns
            -------
            tuple[int, dash.html.Div]
                ``(reset_clicks, charts_container)`` with five pie charts.
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
