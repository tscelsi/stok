from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from stok.portfolio import Portfolio

ActionType = Literal["buy", "sell", "hold"]


class StrategyBuyActionModel(BaseModel):
    quantity: int = Field(gt=0)
    symbol: str = Field(min_length=1, max_length=4)
    action_type: str = Field("buy")


class StrategySellActionModel(BaseModel):
    symbol: str = Field(min_length=1, max_length=4)
    quantity: int = Field(gt=0)
    action_type: str = Field("sell")


class BaseStrategy:
    def execute(
        self,
        context: pd.DataFrame,
        portfolio: Portfolio,
        available_funds: float,
        stock_price: float,
    ) -> StrategyBuyActionModel | StrategySellActionModel | None:
        raise NotImplementedError()
