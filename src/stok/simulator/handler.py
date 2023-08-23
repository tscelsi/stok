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

from pydantic import BaseModel, ConfigDict

from stok.portfolio import Portfolio, PortfolioEntryModel

from ..models import SimulatorDayModel
from ..strategies.random_strategy import RandomStrategy
from ..strategies.strategies import StrategyBuyActionModel, StrategySellActionModel


class SimulatorError(Exception):
    pass


class StateModel(BaseModel):
    """State of my portfolio for a single stock"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    portfolio: Portfolio
    money_available: float


class Simulator:
    def __init__(
        self,
        initial_state: StateModel = StateModel(
            portfolio=Portfolio(), money_available=100
        ),
    ) -> None:
        self.strategy = RandomStrategy()
        self._state = initial_state

    def _update_state(
        self,
        action: StrategyBuyActionModel | StrategySellActionModel,
        day: SimulatorDayModel,
    ):
        if action.action_type == "buy":
            self._buy_stock(action, day)
        elif action.action_type == "sell":
            self._sell_stock(action, day)

    def _enough_money(self, price_of_buy_action: float) -> bool:
        """Do we have enough action to buy this stock"""
        return self._state.money_available >= price_of_buy_action

    def _buy_stock(
        self, action: StrategyBuyActionModel, day: SimulatorDayModel
    ) -> None:
        """Buy a stock based on e.o.d price"""
        price_of_buy_action = action.quantity * day.close
        if not self._enough_money(price_of_buy_action):
            raise SimulatorError("ur broke homie")
        self._state.money_available -= price_of_buy_action
        buy_entry = PortfolioEntryModel(
            symbol=action.symbol,
            buy_price=day.close,
            bought_date=day.date,
            quantity=action.quantity,
        )
        self._state.portfolio._add(buy_entry)

    def _sell_stock(
        self, action: StrategySellActionModel, day: SimulatorDayModel
    ) -> None:
        """Buy a stock based on e.o.d price"""
        revenue_of_sell_action = action.quantity * day.close
        self._state.money_available += revenue_of_sell_action
        self._state.portfolio._remove(action)
