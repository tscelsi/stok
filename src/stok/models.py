import datetime

from pydantic import BaseModel


class SimulatorDayModel(BaseModel):
    date: datetime.date
    open: float
    high: float
    low: float
    close: float
    volume: int
    dividends: float
    stock_splits: float
