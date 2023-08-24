import pandas as pd

from stok.portfolio import Portfolio

from .strategies import BaseStrategy, StrategyBuyActionModel, StrategySellActionModel

SHORT_TERM_WINDOW = 5
LONG_TERM_WINDOW = 15


class MovingAverageStrategy(BaseStrategy):
    def __init__(
        self,
        short_term_window=5,
        long_term_window=15,
        fund_allocation_proportion=0.5,
        initial_holding_days=15,
    ):
        self.short_term_window = short_term_window
        self.long_term_window = long_term_window
        self.fund_allocation_proportion = fund_allocation_proportion
        self.initial_holding_days = initial_holding_days

    def execute(
        self,
        context: pd.DataFrame,
        portfolio: Portfolio,
        available_funds: float,
        stock_price: float,
    ) -> StrategyBuyActionModel | StrategySellActionModel | None:
        if len(context) < self.initial_holding_days:
            # hold until we have significant amount of data
            return None
        short_term_avg = context.Close.rolling(window=self.short_term_window).mean()
        long_term_avg = context.Close.rolling(window=self.long_term_window).mean()
        buy_signal = (
            short_term_avg.iloc[-1] > long_term_avg.iloc[-1]
            and short_term_avg.iloc[-2] <= long_term_avg.iloc[-2]
        )
        sell_signal = (
            short_term_avg.iloc[-1] < long_term_avg.iloc[-1]
            and short_term_avg.iloc[-2] >= long_term_avg.iloc[-2]
        )
        if buy_signal:
            # allocate 50% of available funds
            funds_to_invest = available_funds * self.fund_allocation_proportion
            quantity_to_buy = funds_to_invest // stock_price
            return StrategyBuyActionModel(
                symbol="GOOG",
                quantity=quantity_to_buy,
            )
        if sell_signal:
            # sell all - if we have any
            if not portfolio._portfolio.empty:
                return StrategySellActionModel(
                    symbol="GOOG",
                    quantity=portfolio._portfolio.at["GOOG", "quantity"],
                )
        return None
