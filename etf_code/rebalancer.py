import pandas as pd


class Rebalancer:
    def __init__(self, opti):
        self.opti = opti
        self.goal, self.difference, self.rebalance_df, self.full_names, self.original = None, None, None, None, None
        self.new_cash_component = {}  # New cash allocation per ticker
        self.rebalancing_component = {}  # Rebalancing adjustment per ticker

        self.get_original()
        self.get_difference()
        self.get_full_names()
        self.get_df()

    def get_original(self):
        self.original = self.opti.portfolio.holdings.copy()
        total = sum(abs(v) for v in self.original.values())

        for ticker in self.original:
            self.original[ticker] /= total
            self.original[ticker] = str(round(100 * self.original[ticker])) + '%'

        for ticker in self.opti.optimum_all:
            if ticker not in self.original:
                self.original[ticker] = 0

    def get_difference(self):
        self.goal = {ticker: self.opti.optimum_all[ticker] * self.opti.portfolio.liquidity
                     for ticker in self.opti.optimum_all}

        # Calculate current portfolio value (sum of absolute holdings for liquidity calculation)
        current_total = sum(abs(v) for v in self.opti.portfolio.holdings.values())
        # Old liquidity = cash + current_total (before any new cash additions)
        # If liquidity increased, that's new cash to allocate
        new_cash_amount = max(0, self.opti.portfolio.liquidity - current_total)
        
        self.difference = {}
        self.new_cash_component = {}
        self.rebalancing_component = {}
        
        for ticker in self.opti.optimum_all:
            goal_value = self.goal[ticker]
            current_holding = self.opti.portfolio.holdings.get(ticker, 0)
            target_weight = self.opti.optimum_all[ticker]
            
            if current_total > 0 and ticker in self.opti.portfolio.holdings:
                # Existing position: separate into new cash allocation and rebalancing
                # New cash component = allocation of new cash based on target weights
                new_cash_comp = target_weight * new_cash_amount
                self.new_cash_component[ticker] = new_cash_comp
                
                # Scaled current = position after allocating new cash proportionally
                scaled_current = current_holding + new_cash_comp
                
                # Rebalancing component = adjustment needed to get from scaled_current to goal
                rebalancing_comp = goal_value - scaled_current
                self.rebalancing_component[ticker] = rebalancing_comp
                
                # For short positions, flip only the rebalancing component in the final difference
                # The total represents the actual trade to execute
                if current_holding < 0:
                    # Flip rebalancing component: take opposite trade for shorts
                    # Total = new_cash_component - rebalancing_component
                    total_difference = new_cash_comp - rebalancing_comp
                else:
                    # Long position: no flip needed
                    total_difference = new_cash_comp + rebalancing_comp
            else:
                # New position: all difference is new cash allocation, no flip
                total_difference = goal_value - current_holding
                self.new_cash_component[ticker] = total_difference
                self.rebalancing_component[ticker] = 0
            
            self.difference[ticker] = total_difference

        for ticker in self.difference:
            self.difference[ticker] = round(self.difference[ticker])

        # Drop zeros after rounding
        self.difference = {ticker: self.difference[ticker] for ticker in self.difference
                           if self.difference[ticker]}

    def get_full_names(self):
        self.full_names = {ticker: self.opti.portfolio.data.etf_full_names.loc[ticker]
                           for ticker in self.difference}

    def get_df(self):
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
