import pandas as pd
from pydantic import BaseModel, Field

from stok.strategies.strategies import StrategySellActionModel


class PortfolioError(Exception):
    pass


class PortfolioEntryModel(BaseModel):
    symbol: str = Field(min_length=1, max_length=4)
    quantity: int = Field(gt=0)


class Portfolio:
    def __init__(self, initial_portfolio: pd.DataFrame | None = None) -> None:
        if initial_portfolio is not None:
            self._validate_portfolio(initial_portfolio)
            self._portfolio = initial_portfolio
        else:
            self._portfolio = pd.DataFrame()

    def _validate_portfolio(self, portfolio: pd.DataFrame) -> None:
        if not set(PortfolioEntryModel.__fields__.keys()).issubset(
            set(portfolio.columns).union({"symbol"})
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
                [buy_action.model_dump(exclude={"symbol"})], index=[buy_action.symbol]
            )
            return
        try:
            self._portfolio.at[buy_action.symbol, "quantity"] += buy_action.quantity
        except KeyError:
            self._portfolio = pd.concat(
                [
                    self._portfolio,
                    pd.DataFrame(
                        [buy_action.model_dump(exclude={"symbol"})],
                        index=[buy_action.symbol],
                    ),
                ]
            )

    def _remove(self, sell_action: StrategySellActionModel) -> None:
        """Remove a stock by from the portfolio if there is no remaining
        quantity. Assumes the removal is valid (in terms of price etc.)."""
        try:
            stock = self._portfolio.loc[sell_action.symbol]
        except KeyError:
            raise PortfolioError(
                f"Cannot sell stock {sell_action.symbol} because it does not exist"
                + " in portfolio"
            )
        if stock.quantity < sell_action.quantity:
            raise PortfolioError(
                f"Cannot sell {sell_action.quantity} from {sell_action.symbol} because"
                + f" only {stock.quantity} are available"
            )
        if stock.quantity == sell_action.quantity:
            self._portfolio = self._portfolio.drop(sell_action.symbol)
            return
        self._portfolio.at[sell_action.symbol, "quantity"] -= sell_action.quantity
