import random

import pandas as pd

from stok.portfolio import Portfolio

from .strategies import BaseStrategy, StrategyBuyActionModel, StrategySellActionModel

ACTIONS = ["buy", "sell", "hold"]


class RandomStrategy(BaseStrategy):
    def execute(
        self,
        context: pd.DataFrame,
        portfolio: Portfolio,
        available_funds: float,
        stock_price: float,
    ) -> StrategyBuyActionModel | StrategySellActionModel | None:
        # have to buy if we have nothing
        if len(portfolio._portfolio) == 0:
            return StrategyBuyActionModel(
                quantity=random.randint(1, 4),
                symbol="GOOG",
            )
        action = ACTIONS[random.randint(0, 2)]
        if action == "buy":
            return StrategyBuyActionModel(
                quantity=random.randint(1, 4),
                symbol="GOOG",
            )
        elif action == "sell":
            # just don't sell more than what we have
            rand_index = random.randint(0, len(portfolio._portfolio) - 1)
            portfolio_entry = portfolio._portfolio.iloc[rand_index]
            return StrategySellActionModel(
                symbol="GOOG",
                quantity=random.randint(1, portfolio_entry.quantity),
            )
