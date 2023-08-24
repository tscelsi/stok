import pytest
from polyfactory.factories.pydantic_factory import ModelFactory

from stok.portfolio import Portfolio, PortfolioEntryModel, PortfolioError
from stok.strategies import StrategySellActionModel


@pytest.fixture
def portfolio():
    yield Portfolio()


def test_portfolio_init() -> None:
    Portfolio()


def test_portfolio_buy(portfolio: Portfolio):
    action = PortfolioEntryModel(
        symbol="GOOG",
        quantity=1,
    )
    portfolio._add(action)
    assert portfolio._portfolio.loc["GOOG"].to_dict() == {
        "quantity": 1,
    }


def test_portfolio_two_buy_actions(
    portfolio: Portfolio, buy_action_factory: ModelFactory
):
    actions: list[PortfolioEntryModel] = buy_action_factory.batch(2)
    first_buy = actions[0]
    second_buy = actions[1]
    first_buy.symbol = "GOOG"
    second_buy.symbol = "AAPL"
    portfolio._add(first_buy)
    portfolio._add(second_buy)
    assert len(portfolio._portfolio) == 2
    assert portfolio._portfolio.loc[first_buy.symbol].to_dict() == {
        "quantity": first_buy.quantity,
    }
    assert portfolio._portfolio.loc[second_buy.symbol].to_dict() == {
        "quantity": second_buy.quantity,
    }


def test_portfolio_buy_same_stock(
    portfolio: Portfolio, buy_action_factory: ModelFactory
):
    # Arrange - two buy actions of 2 GOOG
    actions: list[PortfolioEntryModel] = buy_action_factory.batch(
        2, symbol="GOOG", quantity=2
    )
    first_buy = actions[0]
    second_buy = actions[1]
    # Act
    portfolio._add(first_buy)
    portfolio._add(second_buy)
    assert len(portfolio._portfolio) == 1
    assert portfolio._portfolio.at[first_buy.symbol, "quantity"] == 4


def test_portfolio_sell_all(portfolio: Portfolio, buy_action_factory: ModelFactory):
    # Arrange
    buy_action = buy_action_factory.build(quantity=1, symbol="GOOG")
    portfolio._add(buy_action)
    sell_action = StrategySellActionModel(symbol="GOOG", quantity=1, action_type="sell")
    # Act
    portfolio._remove(sell_action)
    assert portfolio._portfolio.empty


def test_portfolio_sell_some(portfolio: Portfolio, buy_action_factory: ModelFactory):
    # Arrange
    buy_action = buy_action_factory.build(quantity=5, symbol="GOOG")
    portfolio._add(buy_action)
    sell_action = StrategySellActionModel(symbol="GOOG", quantity=1, action_type="sell")
    # Act
    portfolio._remove(sell_action)
    assert not portfolio._portfolio.empty
    assert portfolio._portfolio.loc[buy_action.symbol].quantity == 4


def test_portfolio_sell_too_much(
    portfolio: Portfolio, buy_action_factory: ModelFactory
):
    buy_action = buy_action_factory.build(quantity=5, symbol="GOOG")
    portfolio._add(buy_action)
    with pytest.raises(PortfolioError) as e:
        sell_action = StrategySellActionModel(
            symbol="GOOG", quantity=10, action_type="sell"
        )
        portfolio._remove(sell_action)
    assert e.value.args[0] == (
        f"Cannot sell 10 from {buy_action.symbol} because only 5 are available"
    )


def test_portfolio_sell_cant_find_stock(
    portfolio: Portfolio, buy_action_factory: ModelFactory
):
    # Arrange
    buy_action = buy_action_factory.build(quantity=5, symbol="GOOG")
    portfolio._add(buy_action)
    with pytest.raises(PortfolioError) as e:
        sell_action = StrategySellActionModel(
            symbol="TSLA",
            quantity=10,
            action="sell",
        )
        portfolio._remove(sell_action)
    assert e.value.args[0] == (
        f"Cannot sell stock {sell_action.symbol} because it does not exist in portfolio"
    )
