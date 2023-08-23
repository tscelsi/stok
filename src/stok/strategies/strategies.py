from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from stok.models import ActionModel

ActionType = Literal["buy", "sell", "hold"]


class BaseStrategy:
    def execute(self, context: object) -> ActionModel:
        raise NotImplementedError()


class StrategyBuyActionModel(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    quantity: int = Field(gt=0)
    symbol: str = Field(min_length=1, max_length=4)
    action_type: str = Field("buy")


class StrategySellActionModel(BaseModel):
    id: UUID
    quantity: int = Field(gt=0)
    action_type: str = Field("sell")
