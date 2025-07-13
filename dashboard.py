import dash
from dash import html, dcc, Input, Output



class Dashboard(dash.Dash):

    def __init__(self):

        super().__init__()
        self.layout_functions = [Dashboard.title, Dashboard.radio_risk_comp, Dashboard.radio_currency_comp, Dashboard.switch_short_comp]

        self.main_div = None
        self.risk, self.currency, self.allow_short = None, None, None
        self.get_layout()
        self.callbacks()


    def get_layout(self):

        self.main_div = []
        for func in self.layout_functions:
            self.main_div.extend(func())

        self.layout = html.Div(self.main_div)


    @staticmethod
    def title():
        return [html.H1("ETF Rebalancer")]


    @staticmethod
    def radio_risk_comp():
        return [html.H4("Select risk level:"),
        dcc.RadioItems(
            id='radio-risk',
            options=[
                {'label': 'Low', 'value': 1},
                {'label': 'Medium', 'value': 2},
                {'label': 'High', 'value': 3}
            ],
            value = 3
        )]


    @staticmethod
    def radio_currency_comp():
        return [html.H4("Select currency:"),
                dcc.RadioItems(
                    id='radio-currency',
                    options=[{'label': x, 'value': x} for x in ['SGD', 'EUR', 'USD']],
                    value='USD'
                )]


    @staticmethod
    def switch_short_comp():
        return [html.H4("Allow Short:"),
            dcc.RadioItems(
                id='switch-short',
                options=[
                    {'label': 'Yes', 'value': True},
                    {'label': 'No', 'value': False}
                ],
                value=False,
            )
        ]


    def callbacks(self):

        @self.callback(
            Input('radio-risk', 'value'),
            Input('radio-currency', 'value'),
            Input('switch-short', 'value')
        )
        def input_callbacks(risk, currency, allow_short):
            self.risk = risk
            self.currency = currency
            self.allow_short = allow_short


Dashboard().run()










