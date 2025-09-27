"""
Rebalancing utilities to translate optimal weights into actionable trades.

This module exposes :class:`Rebalancer`, which takes an optimized portfolio
(:class:`~opti.Opti`) and computes:

* Target base-currency amounts per ticker from optimal weights and total
  liquidity.
* Dollar (or base-currency) buy/sell differences versus current holdings.
* A tidy pandas DataFrame summarizing the rebalance plan with human-readable
  before/after allocations.
* Optional substitution hints by appending ``(<BEST alternative>)`` to tickers
  found in the ``alternatives.csv`` map loaded by :class:`data.Data`.

Workflow
--------
1. Construct :class:`Rebalancer` with an :class:`~opti.Opti` instance.
2. It extracts the original holdings, computes target amounts and differences,
   resolves long names, and assembles :attr:`rebalance_df`.

Notes
-----
* Liquidity is taken from ``opti.portfolio.liquidity`` and assumed to be in the
  same base currency used by the data/optimization pipeline.
* Differences are rounded to integer currency units for readability and may need
  further lot-size or notional rounding downstream.
"""

import pandas as pd


class Rebalancer:
    """
    Build a rebalance plan from an optimized portfolio.

    Parameters
    ----------
    opti : Opti
        An optimizer instance exposing:
        * ``optimum_all`` — mapping ``{ticker: weight}`` over the current universe.
        * ``portfolio.holdings`` — current position values (same currency as liquidity).
        * ``portfolio.liquidity`` — cash + current holdings total value.
        * ``portfolio.data.etf_full_names`` — pandas Series mapping ticker -> long name.
        * ``portfolio.data.alternatives`` — dict mapping ticker -> preferred substitute
          label (optional; used to annotate the ``Ticker`` column).

    Attributes
    ----------
    opti : Opti
        Reference to the optimizer/portfolio wrapper.
    goal : dict[str, float] | None
        Target currency amounts per ticker (weight × liquidity).
    difference : dict[str, float] | None
        Rounded currency deltas to trade (positive = buy, negative = sell).
        Only non-zero entries are kept.
    rebalance_df : pandas.DataFrame | None
        Summary table with columns:
        ``['Ticker', 'ETF', 'Buy/Sell', 'Before', 'After']``.
        ``Ticker`` may include a ``(<BEST>)`` suffix when an alternative exists.
    full_names : dict[str, str] | None
        Mapping from ticker to long display name for tickers in :attr:`difference`.
    original : dict[str, str | float] | None
        Baseline allocation per ticker. For tickers present in current holdings,
        values are formatted percentage strings like ``'12%'``; for new tickers
        not currently held, the value is ``0``.
    """

    def __init__(self, opti):
        """
        Initialize and compute the rebalance plan artifacts.

        Calls, in order: :meth:`get_original`, :meth:`get_difference`,
        :meth:`get_full_names`, :meth:`get_df`.

        Parameters
        ----------
        opti : Opti
            Optimizer instance with optimal weights and portfolio data.

        Returns
        -------
        None
        """
        self.opti = opti
        self.goal, self.difference, self.rebalance_df, self.full_names, self.original = None, None, None, None, None

        self.get_original()
        self.get_difference()
        self.get_full_names()
        self.get_df()

    def get_original(self):
        """
        Build the baseline (current) allocation dictionary.

        Converts current holdings ``{ticker: value}`` into percentage strings
        relative to the total, e.g., ``'8%'``. For tickers that appear in the
        optimized universe but are not currently held, inserts ``0``.

        Side Effects
        ------------
        Sets :attr:`original`.

        Returns
        -------
        None
        """
        self.original = self.opti.portfolio.holdings.copy()
        total = sum(self.original.values())

        for ticker in self.original:
            self.original[ticker] /= total
            self.original[ticker] = str(round(100 * self.original[ticker])) + '%'

        for ticker in self.opti.optimum_all:
            if ticker not in self.original:
                self.original[ticker] = 0

    def get_difference(self):
        """
        Compute target amounts and buy/sell differences versus current holdings.

        * Target amount per ticker: ``goal[t] = weight[t] * liquidity``.
        * Difference per ticker: ``goal[t] - current_value[t]`` (rounded to int).
        * Removes zero (after rounding) entries.

        Side Effects
        ------------
        Sets :attr:`goal` and :attr:`difference`.

        Returns
        -------
        None
        """
        self.goal = {ticker: self.opti.optimum_all[ticker] * self.opti.portfolio.liquidity
                     for ticker in self.opti.optimum_all}

        self.difference = self.goal.copy()

        for ticker in self.opti.portfolio.holdings:
            if ticker in self.difference:
                self.difference[ticker] -= self.opti.portfolio.holdings[ticker]
            else:
                self.difference[ticker] = -self.opti.portfolio.holdings[ticker]

        for ticker in self.difference:
            self.difference[ticker] = round(self.difference[ticker])

        # Drop zeros after rounding
        self.difference = {ticker: self.difference[ticker] for ticker in self.difference
                           if self.difference[ticker]}

    def get_full_names(self):
        """
        Resolve long ETF names for tickers that require trades.

        Side Effects
        ------------
        Sets :attr:`full_names` using ``portfolio.data.etf_full_names`` for the
        tickers present in :attr:`difference`.

        Returns
        -------
        None
        """
        self.full_names = {ticker: self.opti.portfolio.data.etf_full_names.loc[ticker]
                           for ticker in self.difference}

    def get_df(self):
        """
        Assemble the rebalance summary DataFrame.

        The resulting table includes:
        * ``Ticker`` — symbol (possibly suffixed with the best alternative),
        * ``ETF`` — long name,
        * ``Buy/Sell`` — currency amount to trade (rounded int),
        * ``Before`` — current allocation as a percentage string or ``0``,
        * ``After`` — target allocation as a percentage string.

        Rows with an empty target (NaN) **and** ``Before == 0`` are dropped.
        The table is sorted by ``Buy/Sell`` descending.
        If a ticker is found in the alternatives map, it is displayed as
        ``<Ticker> (<BEST>)``.

        Side Effects
        ------------
        Sets :attr:`rebalance_df`.

        Returns
        -------
        None
        """
        goal = self.goal.copy()
        for ticker in goal:
            goal[ticker] = str(round(100 * goal[ticker] / self.opti.portfolio.liquidity)) + '%'
        goal = {ticker: goal[ticker] for ticker in goal if (goal[ticker] != '0%')}

        self.rebalance_df = pd.DataFrame({
            'ETF': self.full_names,
            'Buy/Sell': self.difference,
            'Before': self.original,
            'After': goal
        }).reset_index().sort_values(by='Buy/Sell', ascending=False)

        self.rebalance_df = self.rebalance_df[(~self.rebalance_df['After'].isna()) | (self.rebalance_df['Before'] != 0)]
        self.rebalance_df.rename(columns={'index': 'Ticker'}, inplace=True)

        # Annotate with preferred substitutes when available
        alt = self.opti.portfolio.data.alternatives
        for i, row in self.rebalance_df.iterrows():
            if row['Ticker'] in alt:
                self.rebalance_df.loc[i, 'Ticker'] = f'{row["Ticker"]} ({alt[row["Ticker"]]})'
