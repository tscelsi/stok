"""
The Simulator class is responsible for simulating the stock market.


Usage:

```python
    # run a simulation for 200 days
    s = Simulator()
    s.run()
    s.save_results()
```
"""
import datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict

from stok.portfolio import Portfolio, PortfolioEntryModel

from ..strategies.random_strategy import RandomStrategy
from ..strategies.strategies import StrategyBuyActionModel, StrategySellActionModel


class EmulatorError(Exception):
    pass


class StateModel(BaseModel):
    """State of my portfolio for a single stock"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    portfolio: Portfolio
    money_available: float


class Emulator:
    def __init__(
        self,
        initial_state: StateModel = StateModel(
            portfolio=Portfolio(), money_available=100
        ),
    ) -> None:
        self.strategy = RandomStrategy()
        self._state = initial_state
        # TODO: implement historical tracking for analytics
        self._history = []

    def run(self):
        history = pd.read_csv("goog.csv", index_col="Date", parse_dates=True)
        for date, day_stats in history.iterrows():
            ctx = history.loc[:date]
            action = self.strategy.execute(ctx, day_stats.Close, self._state.portfolio)
            if action is not None:
                self._update_state(
                    action,
                    day_stats.Close,
                    date,
                )

    def _update_state(
        self,
        action: StrategyBuyActionModel | StrategySellActionModel,
        price: float,
        date: datetime.date,
    ) -> None:
        """Update the state of the emulator based on the action taken by
        the playing strategy"""
        if action.action_type == "buy":
            self._buy_stock(action, price)
        elif action.action_type == "sell":
            self._sell_stock(action, price)
        # update history with new state

    def _enough_money(self, price_of_buy_action: float) -> bool:
        """Do we have enough action to buy this stock"""
        return self._state.money_available >= price_of_buy_action

    def _buy_stock(self, action: StrategyBuyActionModel, unit_price: float) -> None:
        """Buy a stock by updating money available and portfolio counts"""
        price_of_buy_action = action.quantity * unit_price
        if not self._enough_money(price_of_buy_action):
            raise EmulatorError("ur broke homie")
        self._state.money_available -= price_of_buy_action
        buy_entry = PortfolioEntryModel(
            symbol=action.symbol,
            quantity=action.quantity,
        )
        self._state.portfolio._add(buy_entry)

    def _sell_stock(self, action: StrategySellActionModel, unit_price: float) -> None:
        """Sell a stock by updating money available and portfolio counts"""
        revenue_of_sell_action = action.quantity * unit_price
        self._state.money_available += revenue_of_sell_action
        self._state.portfolio._remove(action)
