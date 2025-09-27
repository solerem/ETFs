"""
Exposure breakdown plots for optimized portfolios.

This module defines :class:`Exposure`, which produces pie charts showing the
portfolio's composition across:

* Trading currencies (via ETF native currency and FX pseudo-tickers),
* Asset class,
* Equity sector,
* Bond type,
* Geography.

Data sources
------------
* Exposures are read from ``opti.portfolio.data.exposure`` (a pandas DataFrame
  indexed by ticker with categorical columns such as ``Asset Class``,
  ``Stock Sector``, ``Bond Type``, and ``Geography``).
* Optimized weights are read from :class:`~opti.Opti` (``opti.optimum``).
* Ticker trading currencies are read from
  ``opti.portfolio.data.etf_currency``; FX pseudo-tickers are taken from
  :data:`data.Data.possible_currencies`.

All figures are returned as Dash-ready ``html.Img`` elements using
:meth:`opti.Opti.save_fig_as_dash_img`. A non-interactive Matplotlib backend
(``Agg``) is used so this module can run headless.
"""

from opti import Opti
from data import Data
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Exposure:
    """
    Build exposure pie charts from an optimized portfolio.

    Parameters
    ----------
    opti : Opti
        Optimizer instance providing:
        - ``optimum``: mapping ``{ticker: weight}`` of optimized weights.
        - ``portfolio.data.exposure``: pandas DataFrame with categorical
          exposure columns (e.g., ``Asset Class``, ``Stock Sector``,
          ``Bond Type``, ``Geography``).
        - ``portfolio.data.etf_currency``: mapping ticker → native trading currency.
        - ``portfolio.currency`` and :data:`data.Data.possible_currencies`
          for FX pseudo-tickers.

    Attributes
    ----------
    opti : Opti
        Reference to the optimizer.
    optimum : dict[str, float]
        Optimized weight mapping used to aggregate exposures.
    exposure_df : pandas.DataFrame
        Table of categorical exposures (indexed by ticker; columns per category).
    """

    def __init__(self, opti):
        """
        Initialize the exposure helper and capture required references.

        Parameters
        ----------
        opti : Opti
            Optimizer instance with optimal weights and exposure data.

        Returns
        -------
        None
        """
        self.opti = opti
        self.optimum = self.opti.optimum
        self.exposure_df = self.opti.portfolio.data.exposure

    def plot_pie_chart(self, dico, title):
        """
        Render a pie chart from a category-to-weight dictionary.

        Zero-weight categories are removed. The figure is converted to a Dash
        ``html.Img`` via :meth:`opti.Opti.save_fig_as_dash_img`.

        Parameters
        ----------
        dico : dict[str, float]
            Mapping from category label to (non-normalized) weight.
        title : str
            Chart title.

        Returns
        -------
        dash.html.Img
            Dash image component for embedding in a layout.
        """
        dico = {key: dico[key] for key in dico if dico[key] != 0}
        sorted_dico = dict(sorted(dico.items(), key=lambda item: item[1], reverse=True))

        fig, ax = plt.subplots()
        ax.pie(
            sorted_dico.values(),
            labels=sorted_dico.keys(),
            autopct=lambda pct: f'{int(round(pct))}%'
        )
        ax.set_title(title)
        return Opti.save_fig_as_dash_img(fig, output_path=None)

    def plot_currency(self):
        """
        Plot exposure by trading currency (including FX pseudo-tickers).

        Logic
        -----
        * If a key in ``optimum`` is itself a currency code in
          :data:`data.Data.possible_currencies`, treat it directly as currency
          exposure (FX pseudo-ticker).
        * Otherwise, look up the ETF's native trading currency in
          ``portfolio.data.etf_currency`` and attribute the weight accordingly.

        Notes
        -----
        Weights are used as-is (no extra normalization) since the optimized
        weights already sum to 1 under the L1 budget.

        Returns
        -------
        dash.html.Img
            Dash image component for the currency pie chart.
        """
        etf_currency = self.opti.portfolio.data.etf_currency

        currency_dict = {curr: 0 for curr in Data.possible_currencies+Data.helper_currencies}
        for ticker in self.optimum:
            if ticker in Data.possible_currencies:
                currency_dict[ticker] += self.optimum[ticker]
            else:
                currency_dict[etf_currency[ticker]] += self.optimum[ticker]

        return self.plot_pie_chart(currency_dict, 'Currency')

    def plot_other_exposure(self, name):
        """
        Generic pie chart for a categorical exposure column.

        The method aggregates optimized weights by the category in
        ``exposure_df[name]`` (e.g., ``'Asset Class'``, ``'Stock Sector'``,
        ``'Bond Type'``, ``'Geography'``). If the total weight for that
        category set is zero, returns ``None``.

        Parameters
        ----------
        name : str
            Column name in :attr:`exposure_df` to aggregate by.

        Returns
        -------
        dash.html.Img | None
            Dash image component for the category pie chart, or ``None`` if
            the optimized portfolio has no exposure mapped for that column.
        """
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
        """
        Plot exposure by high-level asset class.

        Returns
        -------
        dash.html.Img | None
            Dash image component, or ``None`` if there is no exposure.
        """
        return self.plot_other_exposure('Asset Class')

    def plot_sector(self):
        """
        Plot exposure by equity sector.

        Returns
        -------
        dash.html.Img | None
            Dash image component, or ``None`` if there is no exposure.
        """
        return self.plot_other_exposure('Stock Sector')

    def plot_type(self):
        """
        Plot exposure by bond type.

        Returns
        -------
        dash.html.Img | None
            Dash image component, or ``None`` if there is no exposure.
        """
        return self.plot_other_exposure('Bond Type')

    def plot_geo(self):
        """
        Plot exposure by geography.

        Returns
        -------
        dash.html.Img | None
            Dash image component, or ``None`` if there is no exposure.
        """
        return self.plot_other_exposure('Geography')
