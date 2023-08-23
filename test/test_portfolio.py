import datetime
from uuid import UUID, uuid4

import pytest
from polyfactory.factories.pydantic_factory import ModelFactory

from stok.portfolio import Portfolio, PortfolioEntryModel, PortfolioError
from stok.strategies import StrategySellActionModel


@pytest.fixture
def portfolio():
    yield Portfolio()


@pytest.fixture
def uuid():
    yield uuid4()


@pytest.fixture
def buy_action_factory(uuid: UUID):
    class BuyActionFactory(ModelFactory):
        __model__ = PortfolioEntryModel

    yield BuyActionFactory()


def test_portfolio_init() -> None:
    Portfolio()


def test_portfolio_buy(portfolio: Portfolio, uuid: UUID):
    action = PortfolioEntryModel(
        id=uuid,
        symbol="GOOG",
        buy_price=100.0,
        bought_date="2021-01-01",
        quantity=1,
    )
    portfolio._add(action)
    assert portfolio._portfolio.loc[uuid].to_dict() == {
        "symbol": "GOOG",
        "buy_price": 100.0,
        "bought_date": datetime.date.fromisoformat("2021-01-01"),
        "quantity": 1,
    }


def test_portfolio_two_buy_actions(
    portfolio: Portfolio, buy_action_factory: ModelFactory
):
    actions: list[PortfolioEntryModel] = buy_action_factory.batch(2)
    first_buy = actions[0]
    second_buy = actions[1]
    portfolio._add(first_buy)
    portfolio._add(second_buy)
    assert len(portfolio._portfolio) == 2
    assert portfolio._portfolio.loc[first_buy.id].to_dict() == {
        "symbol": first_buy.symbol,
        "buy_price": first_buy.buy_price,
        "bought_date": first_buy.bought_date,
        "quantity": first_buy.quantity,
    }
    assert portfolio._portfolio.loc[second_buy.id].to_dict() == {
        "symbol": second_buy.symbol,
        "buy_price": second_buy.buy_price,
        "bought_date": second_buy.bought_date,
        "quantity": second_buy.quantity,
    }


def test_portfolio_sell_all(portfolio: Portfolio, buy_action_factory: ModelFactory):
    # Arrange
    buy_action = buy_action_factory.build(quantity=1)
    portfolio._add(buy_action)
    sell_action = StrategySellActionModel(
        id=buy_action.id, quantity=1, action_type="sell"
    )
    # Act
    portfolio._remove(sell_action)
    assert portfolio._portfolio.empty


def test_portfolio_sell_some(portfolio: Portfolio, buy_action_factory: ModelFactory):
    # Arrange
    buy_action = buy_action_factory.build(quantity=5)
    portfolio._add(buy_action)
    sell_action = StrategySellActionModel(
        id=buy_action.id, quantity=1, action_type="sell"
    )
    # Act
    portfolio._remove(sell_action)
    assert not portfolio._portfolio.empty
    assert portfolio._portfolio.loc[buy_action.id].quantity == 4


def test_portfolio_remove_too_much(
    portfolio: Portfolio, buy_action_factory: ModelFactory
):
    buy_action = buy_action_factory.build(quantity=5)
    portfolio._add(buy_action)
    with pytest.raises(PortfolioError) as e:
        sell_action = StrategySellActionModel(
            id=buy_action.id, quantity=10, action_type="sell"
        )
        portfolio._remove(sell_action)
    assert e.value.args[0] == (
        f"Cannot sell 10 from {buy_action.id} because only 5 are available"
    )


def test_portfolio_remove_cant_find_id(
    portfolio: Portfolio, buy_action_factory: ModelFactory
):
    # Arrange
    buy_action = buy_action_factory.build(quantity=5)
    portfolio._add(buy_action)
    with pytest.raises(PortfolioError) as e:
        sell_action = StrategySellActionModel(
            id="ac3f4eed-d1c8-411c-919a-aea502c1e27b",
            quantity=10,
            action="sell",
        )
        portfolio._remove(sell_action)
    assert e.value.args[0] == (
        f"Cannot sell stock {sell_action.id} because it does not exist"
    )
