import random

import pandas as pd

from .strategies import BaseStrategy
from stok.models import ActionModel

ACTIONS = ["buy", "sell", "hold"]


class RandomStrategy(BaseStrategy):
    def execute(self, context: pd.DataFrame, current_price: float) -> ActionModel:
        action = ACTIONS[random.randint(0, 2)]
        return ActionModel(
            action=action,
            quantity=random.randint(0, 100),
            price=current_price if action == "buy" else 0.0,
            date=context.iloc[-1].index,
            stock="GOOG",
            confidence=-1,
        )
