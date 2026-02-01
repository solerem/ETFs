from opti import Opti
from data import Data
import plotly.graph_objects as go

from charts import dash_graph


class Exposure:
    def __init__(self, opti):
        self.opti = opti
        self.optimum = self.opti.optimum
        self.exposure_df = self.opti.portfolio.data.exposure

    def plot_pie_chart(self, dico, title):
        dico = {key: abs(dico[key]) for key in dico if dico[key] != 0}
        sorted_dico = dict(sorted(dico.items(), key=lambda item: item[1], reverse=True))
        fig = go.Figure(data=[go.Pie(labels=list(sorted_dico.keys()),
                                     values=list(sorted_dico.values()))])
        fig.update_traces(
            textinfo='label+percent',
            texttemplate='%{label}: %{percent:.1%}',
            hovertemplate='%{label}: %{percent:.1%}<extra></extra>'
        )
        fig.update_layout(
            title=title,
            legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5)
        )
        return dash_graph(fig)

    def plot_currency(self):
        etf_currency = self.opti.portfolio.data.etf_currency

        currency_dict = {curr: 0 for curr in Data.possible_currencies+Data.helper_currencies}
        for ticker in self.optimum:
            if ticker in Data.possible_currencies:
                currency_dict[ticker] += self.optimum[ticker]
            else:
                currency_dict[etf_currency[ticker]] += self.optimum[ticker]

        return self.plot_pie_chart(currency_dict, 'Currency')

    def plot_other_exposure(self, name):
        category_df = self.exposure_df[name].dropna()

        category_dict = {cat: 0 for cat in category_df.unique()}
        for ticker in self.optimum:
            if ticker in category_df:
                category_dict[category_df[ticker]] += self.optimum[ticker]

        total = sum(category_dict.values())
        if total == 0:
            return None

        # Normalize to proportions before plotting
        for cat in category_dict:
            category_dict[cat] /= total

        return self.plot_pie_chart(category_dict, name)

    def plot_category(self):
        return self.plot_other_exposure('Asset Class')

    def plot_sector(self):
        return self.plot_other_exposure('Stock Sector')

    def plot_geo(self):
        return self.plot_other_exposure('Geography')
