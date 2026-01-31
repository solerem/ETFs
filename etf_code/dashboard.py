import dash
import pandas as pd
import threading
from dash import html, dcc, Input, Output, State
from dash.dependencies import ALL
from dash import dash_table

import dash_bootstrap_components as dbc

# Shared state for backtest progress (written by worker thread, read by interval callback)
_backtest_state = {"running": False, "current": 0, "total": 1, "done": False, "result": None, "error": None}

from data import Data
from backtest import Backtest
from rebalancer import Rebalancer
from portfolio import Portfolio
from exposure import Exposure
from opti import Opti


class Dashboard(dash.Dash):
    def __init__(self):
        super().__init__(external_stylesheets=[dbc.themes.LUX, dbc.icons.BOOTSTRAP])
        self.index_string = self.index_string.replace(
            "</head>",
            """
            <style>
              body { background-color: #4a4a4a !important; }
              .dash-container { background-color: #4a4a4a !important; }
              #react-entry-point { background-color: #4a4a4a !important; }
              @media (min-width: 992px){ .sticky-card{ position: sticky; top: 1rem; } }
              .chart-frame{ max-width:100%; max-height:none; overflow:visible; }
              .chart-frame img, .chart-frame svg, .chart-frame canvas{
                display:block; max-width:100%; height:auto !important;
              }
              .chart-frame .dash-graph{ height:520px !important; }
              .grid-2{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:1rem; align-items:start; }
              @media (max-width: 991.98px){ .grid-2{ grid-template-columns: 1fr; } }
              .card { background-color: #e9ecef !important; }
              .card-body { background-color: #e9ecef !important; }
              .card-header { background-color: #e9ecef !important; }
              .rc-slider-track { background-color: #212529 !important; }
              .rc-slider-handle { border-color: #212529 !important; }
              .rc-slider-handle:active { border-color: #212529 !important; box-shadow: 0 0 0 5px rgba(33, 37, 41, 0.2) !important; }
              .rc-slider-dot-active { border-color: #212529 !important; }
              .position-toggle-btn{ 
                flex: 1; 
                font-size: 1rem; 
                font-weight: 500; 
                padding: 0.625rem 1rem;
                min-height: 44px;
                border-radius: 0;
                background-color: white;
                border: 1px solid #dee2e6;
                color: #212529;
              }
              .position-toggle-btn:first-child {
                border-top-left-radius: 0.375rem;
                border-bottom-left-radius: 0.375rem;
              }
              .position-toggle-btn:last-child {
                border-top-right-radius: 0.375rem;
                border-bottom-right-radius: 0.375rem;
              }
              .position-toggle-btn.active {
                z-index: 1;
                background-color: #212529 !important;
                border-color: #212529 !important;
                color: white !important;
              }
              .position-toggle-btn:not(.active) {
                background-color: white !important;
                border-color: #212529 !important;
                color: #212529 !important;
              }
              .position-toggle-btn:hover:not(.active) {
                background-color: #f8f9fa !important;
              }
            </style>
            </head>
            """
        )

        self.main_div = None
        self.risk, self.currency, self.cash_sgd, self.holdings, self.rates, self.max_assets = None, None, None, None, None, 10
        self.portfolio, self.opti, self.backtest, self.rebalancer, self.exposure = None, None, None, None, None
        self.long_only = False  # False for Long/Short, True for Long only

        self.get_layout()
        self.callbacks()

    def get_layout(self):
        self.layout = html.Div([
            dcc.Store(id='init-store'),

            dbc.Navbar(
                dbc.Container([
                    dbc.NavbarBrand("ETF Portfolio Optimization", className="fw-semibold"),
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

        ])

    def _sidebar_controls(self):
        return dbc.Card([
            dbc.CardBody([

                html.Div([
                    dbc.ButtonGroup([
                        dbc.Button(
                            "Long/Short",
                            id="btn-long-short",
                            n_clicks=0,
                            active=True,
                            color="secondary",
                            className="position-toggle-btn"
                        ),
                        dbc.Button(
                            "Long only",
                            id="btn-long-only",
                            n_clicks=0,
                            active=False,
                            color="secondary",
                            className="position-toggle-btn"
                        )
                    ], id="long-only-toggle", className="w-100 mb-3")
                ]),
                html.Div(className="mb-3"),

                # Risk controls
                dbc.Label("Risk level", html_for="risk-input", className="fw-semibold"),
                dcc.Slider(id='risk-input', min=0, max=10, step=1, value=5,
                           marks={i: str(i) for i in range(0, 11)},
                           tooltip={"placement": "bottom", "always_visible": False}),
                html.Div(className="mb-3"),

                dbc.Label("Max number of assets", html_for="max-assets-input", className="fw-semibold"),
                dbc.Input(id='max-assets-input', type='number', value=10, min=1, max=100, step=1,
                          placeholder="Max positions (e.g. 10)"),
                html.Div(className="mb-3"),

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

                # Savings Rates
                html.Div(className="d-flex align-items-center mb-2", children=[
                    html.Div(className="fw-semibold me-auto", children="Savings Rates"),
                    dbc.Button([html.I(className="bi bi-plus-lg me-1"), "Add Rate"],
                               id='button-rates', n_clicks=0, color="secondary", size="sm", outline=True)
                ]),
                html.Div(id='rates-container', children=[], className="vstack gap-2"),
                
                html.Div(className="mt-4"),
                dbc.Button([html.I(className="bi bi-magic me-2"), "Create Portfolio"],
                           id='create-portfolio', n_clicks=0, color="primary", className="w-100")
            ])
        ], className="shadow-sm sticky-card")

    def _content_area(self):
        return dbc.Card([
            dbc.CardBody([
                dcc.Tabs(id="main-tabs", value="tab-portfolio", children=[
                    dcc.Tab(label="Optimal Portfolio", value="tab-portfolio", children=[
                        dbc.Spinner(html.Div(id='portfolio-distrib'), size="md")
                    ]),

                    dcc.Tab(label="Rebalance", value="tab-rebalance", children=[
                        html.Div(id='rebalance-div', className="table-wrap")
                    ]),

                    dcc.Tab(label="Exposure", value="tab-exposure", children=[
                        html.Div(id='exposure-graphs')
                    ]),

                    dcc.Tab(label="Backtest", value="tab-backtest", children=[
                        html.Div(className="d-flex align-items-center mb-3", children=[
                            dbc.Button([html.I(className="bi bi-play-fill me-2"), "Launch Backtest"],
                                       id='create-backtest', n_clicks=0, color="primary"),
                        ]),
                        html.Div(id='backtest-progress-container', className="mb-3"),
                        html.Div(id='backtest-graphs'),
                        dcc.Interval(id='backtest-interval', interval=400, n_intervals=0)
                    ]),
                ])
            ])
        ], className="shadow-sm")

    def callbacks(self):

        @self.callback(
            Output('btn-long-short', 'active'),
            Output('btn-long-only', 'active'),
            Output('btn-long-short', 'outline'),
            Output('btn-long-only', 'outline'),
            Input('btn-long-short', 'n_clicks'),
            Input('btn-long-only', 'n_clicks'),
            prevent_initial_call=False
        )
        def update_toggle_buttons(btn_long_short_clicks, btn_long_only_clicks):
            ctx = dash.callback_context
            if not ctx.triggered:
                # Initial load - default to Long/Short
                return True, False, False, True
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            # Always return mutually exclusive states
            if trigger_id == 'btn-long-only':
                # Long only is active, Long/Short is not
                return False, True, True, False
            elif trigger_id == 'btn-long-short':
                # Long/Short is active, Long only is not
                return True, False, False, True
            
            # Default fallback
            return True, False, False, True

        @self.callback(
            Output('init-store', 'data'),
            Input('risk-input', 'value'),
            Input('max-assets-input', 'value'),
            Input('radio-currency', 'value'),
            Input('cash', 'value'),
            Input('btn-long-short', 'active'),
            Input('btn-long-only', 'active'),
            Input({'type': 'holdings-ticker-input', 'index': ALL}, 'value'),
            Input({'type': 'holdings-value-input', 'index': ALL}, 'value'),
            Input({'type': 'rates-ticker-input', 'index': ALL}, 'value'),
            Input({'type': 'rates-value-input', 'index': ALL}, 'value'),
        )
        def input_callbacks(risk, max_assets, currency, cash_sgd, btn_long_short_active, btn_long_only_active,
                            holdings_tickers, holdings_values, rates_tickers, rates_values):
            self.risk = risk
            self.max_assets = 10 if max_assets is None else max(1, int(max_assets))
            self.currency = currency
            self.cash_sgd = cash_sgd
            self.long_only = btn_long_only_active if btn_long_only_active else False
            self.holdings = {ticker: value for ticker, value in zip(holdings_tickers, holdings_values)}
            self.rates = {ticker: value for ticker, value in zip(rates_tickers, rates_values)}
            return 0

        @self.callback(
            Output('cash-label', 'children'),
            Input('radio-currency', 'value')
        )
        def update_cash_label(selected_currency):
            return f'Input Cash (in {selected_currency})'

        @self.callback(
            Output('holdings-container', 'children'),
            Input('button-holdings', 'n_clicks'),
            State({'type': 'holdings-ticker-input', 'index': ALL}, 'value'),
            State({'type': 'holdings-value-input', 'index': ALL}, 'value'),
        )
        def update_holdings(n_clicks, tickers, values):
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
            Output('create-portfolio', 'n_clicks'),
            Output('portfolio-distrib', 'children'),
            Output('rebalance-div', 'children'),
            Output('exposure-graphs', 'children'),
            Input('create-portfolio', 'n_clicks'),
        )
        def create_portfolio(create_portfolio_n_click):
            if create_portfolio_n_click:
                self.portfolio = Portfolio(
                    self.risk,
                    self.cash_sgd,
                    self.holdings,
                    self.currency,
                    static=True,
                    rates=self.rates,
                )
                self.opti = Opti(self.portfolio, long_only=self.long_only, max_assets=self.max_assets)

                portfolio_div = html.Div([
                    html.Div(self.opti.plot_in_sample(), className="chart-frame"),
                    html.Div(self.opti.plot_optimum(), className="chart-frame"),
                    html.Div(self.opti.plot_drawdown(), className="chart-frame"),
                    html.Div(self.opti.plot_weighted_perf(), className="chart-frame"),
                    html.Div(self.opti.plot_info()),
                ], className="grid-2")

                self.rebalancer = Rebalancer(self.opti)
                df = self.rebalancer.rebalance_df.copy()
                df.rename(columns={'Buy/Sell': f'Buy/Sell ({self.currency})'}, inplace=True)
                # Display ticker column with alternative names from alternatives.csv (etf_full_names unchanged)
                df['Ticker'] = df['Ticker'].map(Data.ticker_display_name)

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

                rebalance_div = dbc.Card(dbc.CardBody(table), className="shadow-sm")

                self.exposure = Exposure(self.opti)
                exposure_div = html.Div([
                    html.Div(self.exposure.plot_currency(), className="chart-frame"),
                    html.Div(self.exposure.plot_category(), className="chart-frame"),
                    html.Div(self.exposure.plot_sector(), className="chart-frame"),
                    html.Div(self.exposure.plot_geo(), className="chart-frame"),
                ], className="grid-2")

                return 0, portfolio_div, rebalance_div, exposure_div
            return 0, dash.no_update, dash.no_update, dash.no_update

        def _run_backtest(opti):
            global _backtest_state
            try:
                _backtest_state["running"] = True
                _backtest_state["done"] = False
                _backtest_state["current"] = 0
                _backtest_state["total"] = 1
                _backtest_state["result"] = None
                _backtest_state["error"] = None

                def progress(current, total):
                    _backtest_state["current"] = current
                    _backtest_state["total"] = total

                backtest = Backtest(opti, progress_callback=progress)
                _backtest_state["backtest"] = backtest
                result_div = html.Div([
                    html.Div(backtest.plot_backtest(), className="chart-frame"),
                    html.Div(backtest.plot_weights(), className="chart-frame"),
                    html.Div(backtest.plot_drawdown(), className="chart-frame"),
                    html.Div(backtest.plot_perf_attrib(), className="chart-frame"),
                    html.Div(backtest.plot_info()),
                ], className="grid-2")
                _backtest_state["result"] = result_div
                _backtest_state["done"] = True
            except Exception as e:
                _backtest_state["error"] = str(e)
                _backtest_state["done"] = True
            finally:
                _backtest_state["running"] = False

        @self.callback(
            Output('create-backtest', 'n_clicks'),
            Output('backtest-progress-container', 'children'),
            Output('backtest-graphs', 'children'),
            Input('create-backtest', 'n_clicks'),
            Input('backtest-interval', 'n_intervals'),
            prevent_initial_call=True
        )
        def create_backtest(create_backtest_n_click, _n_intervals):
            global _backtest_state
            if create_backtest_n_click and not _backtest_state["running"] and not _backtest_state["done"]:
                if self.opti is None:
                    return 0, [], html.Div("Create a portfolio first.", className="text-muted")
                threading.Thread(target=_run_backtest, args=(self.opti,), daemon=True).start()
                progress_bar = dbc.Progress(
                    value=0,
                    label="0%",
                    striped=True,
                    animated=True,
                    style={"height": "24px"}
                )
                return 0, progress_bar, dash.no_update

            if _backtest_state["running"]:
                current = _backtest_state["current"]
                total = max(1, _backtest_state["total"])
                pct = int(100 * current / total) if total else 0
                progress_bar = dbc.Progress(
                    value=pct,
                    label=f"{pct}%",
                    striped=True,
                    animated=True,
                    style={"height": "24px"}
                )
                return dash.no_update, progress_bar, dash.no_update

            if _backtest_state["done"]:
                _backtest_state["done"] = False
                if _backtest_state.get("error"):
                    err = _backtest_state["error"]
                    _backtest_state["error"] = None
                    return 0, [], html.Div(f"Error: {err}", className="text-danger")
                result = _backtest_state.get("result")
                if result is not None:
                    self.backtest = _backtest_state.get("backtest")
                return 0, [], result if result is not None else dash.no_update

            return dash.no_update, dash.no_update, dash.no_update
