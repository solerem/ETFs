from opti import Opti
from data import Data
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import io
import base64
from dash import html


class Exposure:

    def __init__(self, opti):

        self.opti = opti
        self.optimum = self.opti.optimum
        self.exposure_df = self.opti.portfolio.data.exposure


    def plot_pie_chart(self, dico, title):

        dico = {key: dico[key] for key in dico if dico[key] != 0}
        sorted_dico = dict(sorted(dico.items(), key=lambda item: item[1], reverse=True))

        fig, ax = plt.subplots()
        ax.pie(
            sorted_dico.values(),
            labels=sorted_dico.keys(),
            autopct=lambda pct: f'{int(round(pct))}%'
        )
        ax.set_title(title)


        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        encoded = base64.b64encode(buf.read()).decode('utf-8')
        img_src = f"data:image/png;base64,{encoded}"

        return html.Img(src=img_src, style={"maxWidth": "100%", "height": "auto"})


    def plot_currency(self):

        etf_currency = self.opti.portfolio.data.etf_currency

        currency_dict = {curr: 0 for curr in Data.possible_currencies}
        for ticker in self.optimum:
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

        for cat in category_dict:
            category_dict[cat] /= total

        return self.plot_pie_chart(category_dict, name)


    def plot_category(self):

        return self.plot_other_exposure('Asset Class')


    def plot_sector(self):

        return self.plot_other_exposure('Stock Sector')


    def plot_type(self):

        return self.plot_other_exposure('Bond Type')


    def plot_geo(self):

        return self.plot_other_exposure('Geography')

