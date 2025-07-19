from opti import Opti
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import io
import base64
from dash import html

'''currency/geographic/sector/asset class'''

class Exposure:

    def __init__(self, opti):

        self.opti = opti
        self.optimum = self.opti.optimum


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

        currency_dict = {curr: 0 for curr in ['SGD', 'EUR', 'USD']}
        for ticker in self.optimum:
            currency_dict[etf_currency[ticker]] += self.optimum[ticker]

        return self.plot_pie_chart(currency_dict, 'Currency exposure')


