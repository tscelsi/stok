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
from pathlib import Path

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
    available_funds: float


class Emulator:
    def __init__(
        self,
        initial_state: StateModel = StateModel(
            portfolio=Portfolio(), available_funds=100
        ),
        strategy=RandomStrategy(),
    ) -> None:
        self.strategy = strategy
        self._state = initial_state
        # init history tracking dataframes
        self._portfolio_history = pd.DataFrame()
        self._wallet_history = pd.DataFrame()
        self._action_history = pd.DataFrame()

    def run(self, days: pd.DataFrame):
        for date, day_stats in days.iterrows():
            ctx = days.loc[:date]
            action = self.strategy.execute(
                ctx, self._state.portfolio, self._state.available_funds, day_stats.Close
            )
            if action is not None:
                self._update_state(
                    action,
                    day_stats.Close,
                )
            # update history
            self._snapshot(date, action, day_stats.Close)

    def save(
        self,
        portfolio_history_file_path: Path = "portfolio_history.csv",
        wallet_history_file_path: Path = "wallet_history.csv",
        action_history_file_path: Path = "action_history.csv",
    ):
        self._portfolio_history.to_csv(portfolio_history_file_path)
        self._wallet_history.to_csv(wallet_history_file_path)
        self._action_history.to_csv(action_history_file_path)

    def _snapshot(
        self,
        date: datetime.date,
        action: StrategyBuyActionModel | StrategySellActionModel | None,
        unit_price: float,
    ):
        """Save the state of the emulator after the days actions"""
        self._update_portfolio_history(date, unit_price)
        self._update_action_history(date, action)
        self._update_wallet_history(date)

    def _update_portfolio_history(self, date: datetime.date, unit_price: float):
        """Update the history of the emulator with the current value of each stock
        in the portfolio"""
        new_rows = []
        for symbol, row in self._state.portfolio._portfolio.iterrows():
            new_rows.append(
                {
                    "symbol": symbol,
                    "quantity": row.quantity,
                    "unit_price": unit_price,
                    "holding_value": unit_price * row.quantity,
                }
            )
        self._portfolio_history = pd.concat(
            [self._portfolio_history, pd.DataFrame(new_rows, index=[date])]
        )

    def _update_wallet_history(self, date: datetime.date):
        self._wallet_history = pd.concat(
            [
                self._wallet_history,
                pd.DataFrame(
                    {"available_funds": [self._state.available_funds]}, index=[date]
                ),
            ]
        )

    def _update_action_history(
        self,
        date: datetime.date,
        action: StrategyBuyActionModel | StrategySellActionModel | None,
    ):
        if action is None:
            return
        self._action_history = pd.concat(
            [
                self._action_history,
                pd.DataFrame(
                    {
                        "symbol": [action.symbol],
                        "quantity": [action.quantity],
                        "action_type": [action.action_type],
                    },
                    index=[date],
                ),
            ]
        )

    def _update_state(
        self,
        action: StrategyBuyActionModel | StrategySellActionModel,
        price: float,
    ) -> None:
        """Update the state of the emulator based on the action taken by
        the playing strategy"""
        if action.action_type == "buy":
            self._buy_stock(action, price)
        elif action.action_type == "sell":
            self._sell_stock(action, price)

    def _enough_money(self, price_of_buy_action: float) -> bool:
        """Do we have enough action to buy this stock"""
        return self._state.available_funds >= price_of_buy_action

    def _buy_stock(self, action: StrategyBuyActionModel, unit_price: float) -> None:
        """Buy a stock by updating money available and portfolio counts"""
        price_of_buy_action = action.quantity * unit_price
        if not self._enough_money(price_of_buy_action):
            raise EmulatorError("ur broke homie")
        self._state.available_funds -= price_of_buy_action
        buy_entry = PortfolioEntryModel(
            symbol=action.symbol,
            quantity=action.quantity,
        )
        self._state.portfolio._add(buy_entry)

    def _sell_stock(self, action: StrategySellActionModel, unit_price: float) -> None:
        """Sell a stock by updating money available and portfolio counts"""
        revenue_of_sell_action = action.quantity * unit_price
        self._state.available_funds += revenue_of_sell_action
        self._state.portfolio._remove(action)

    def _portfolio_value(self, unit_price: float):
        """Calculate the value of the portfolio at the current unit price.
        Assumes only a single stock in the portfolio."""
        return (
            self._state.available_funds
            + self._state.portfolio._portfolio.iloc[0].quantity * unit_price
        )
