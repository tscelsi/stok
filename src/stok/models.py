import datetime

from pydantic import BaseModel


class StockModel(BaseModel):
    """Stock that has been bought"""

    symbol: str
    buy_price: float
    bought_date: datetime.date
    quantity: int


class ActionModel(BaseModel):
    """An action taken by a strategy"""

    action: str
    quantity: int
    price: float
    date: datetime.date
    stock: str
    confidence: float


class SimulatorDayModel(BaseModel):
    date: datetime.date
    open: float
    high: float
    low: float
    close: float
    volume: int
    dividends: float
    stock_splits: float
