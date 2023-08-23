import datetime
from uuid import UUID, uuid4

import pandas as pd
from pydantic import BaseModel, Field

from stok.strategies.strategies import StrategySellActionModel


class PortfolioError(Exception):
    pass


class PortfolioEntryModel(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    symbol: str = Field(min_length=1, max_length=4)
    buy_price: float = Field(gt=0)
    bought_date: datetime.date
    quantity: int = Field(gt=0)


class Portfolio:
    def __init__(self, initial_portfolio: pd.DataFrame | None = None) -> None:
        if initial_portfolio is not None:
            self._validate_portfolio(initial_portfolio)
            self._portfolio = initial_portfolio
        else:
            self._portfolio = pd.DataFrame()
        # TODO: implement history
        self._history = []

    def _validate_portfolio(self, portfolio: pd.DataFrame) -> None:
        if not set(PortfolioEntryModel.__fields__.keys()).issubset(
            set(portfolio.columns).union({"id"})
        ):
            raise PortfolioError(
                "Portfolio must contain the following columns: "
                + ", ".join(PortfolioEntryModel.__fields__.keys())
            )

    def _add(self, buy_action: PortfolioEntryModel):
        """Add a stock to the portfolio by entering it into the dataframe. Assumes the
        purchase is valid."""
        if self._portfolio.empty:
            self._portfolio = pd.DataFrame(
                [buy_action.model_dump(exclude={"id"})], index=[buy_action.id]
            )
        else:
            self._portfolio = pd.concat(
                [
                    self._portfolio,
                    pd.DataFrame(
                        [buy_action.model_dump(exclude={"id"})],
                        index=[buy_action.id],
                    ),
                ]
            )

    def _remove(self, sell_action: StrategySellActionModel) -> None:
        """Remove a stock by from the portfolio if there is no remaining
        quantity. Assumes the removal is valid (in terms of price etc.)."""
        try:
            stock = self._portfolio.loc[sell_action.id]
        except KeyError:
            raise PortfolioError(
                f"Cannot sell stock {sell_action.id} because it does not exist"
            )
        if stock.quantity < sell_action.quantity:
            raise PortfolioError(
                f"Cannot sell {sell_action.quantity} from {sell_action.id} because only"
                + f" {stock.quantity} are available"
            )
        if stock.quantity == sell_action.quantity:
            self._portfolio = self._portfolio.drop(sell_action.id)
            return
        self._portfolio.at[sell_action.id, "quantity"] -= sell_action.quantity
