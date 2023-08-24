from uuid import uuid4

import pandas as pd
import pytest
from polyfactory.factories.pydantic_factory import ModelFactory

from stok.emulator import Emulator, EmulatorError, StateModel
from stok.portfolio import Portfolio, PortfolioError
from stok.strategies import StrategyBuyActionModel, StrategySellActionModel


@pytest.fixture
def uuid():
    yield uuid4()


@pytest.fixture
def buy_action():
    return StrategyBuyActionModel(
        quantity=1,
        symbol="GOOG",
    )


@pytest.fixture
def initial_portfolio(uuid):
    yield Portfolio(
        initial_portfolio=pd.DataFrame(
            {
                "quantity": [1],
            },
            index=["GOOG"],
        )
    )


@pytest.fixture
def initial_state(initial_portfolio):
    return StateModel(portfolio=initial_portfolio, money_available=900.0)


@pytest.fixture
def emulator(initial_state):
    yield Emulator(initial_state=initial_state)


def test_simulator_init():
    Emulator()


def test_buy_stock_valid(emulator: Emulator, buy_action_factory: ModelFactory):
    buy_action = buy_action_factory.build(symbol="GOOG", quantity=1)
    emulator._buy_stock(buy_action, 200)
    assert emulator._state.money_available == 700
    assert len(emulator._state.portfolio._portfolio) == 1
    assert emulator._state.portfolio._portfolio.index.to_list() == ["GOOG"]
    assert emulator._state.portfolio._portfolio.quantity.to_list() == [2]


def test_buy_stock_raises_when_not_enough_money(
    emulator: Emulator, buy_action_factory: ModelFactory
):
    buy_action = buy_action_factory.build(symbol="GOOG", quantity=1000)
    with pytest.raises(EmulatorError) as e:
        emulator._buy_stock(buy_action, 100.0)
    assert e.value.args[0] == "ur broke homie"


def test_sell_stock_valid(emulator: Emulator):
    sell_action = StrategySellActionModel(action_type="sell", quantity=1, symbol="GOOG")
    emulator._sell_stock(sell_action, 200)
    # we gain 200 from selling
    assert emulator._state.money_available == 1100
    assert len(emulator._state.portfolio._portfolio) == 0


def test_sell_stock_non_existent_stock_raises(emulator: Emulator):
    sell_action = StrategySellActionModel(
        action_type="sell",
        symbol="AAPL",
        quantity=1,
    )
    with pytest.raises(PortfolioError) as e:
        emulator._sell_stock(sell_action, 200)
    assert (
        e.value.args[0]
        == f"Cannot sell stock {sell_action.symbol} because it does not exist"
        + " in portfolio"
    )


def test_run():
    """Simple simulation that starts with an empty portfolio and $100"""
    s = Emulator(initial_state=StateModel(portfolio=Portfolio(), money_available=10000))
    s.run()
    print("HI")
