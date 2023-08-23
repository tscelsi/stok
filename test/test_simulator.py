import datetime
from uuid import uuid4

import pandas as pd
import pytest

from stok.portfolio import Portfolio, PortfolioError
from stok.simulator import Simulator, SimulatorError, StateModel
from stok.strategies import StrategyBuyActionModel, StrategySellActionModel


@pytest.fixture
def uuid():
    yield uuid4()


@pytest.fixture
def buy_action():
    return StrategyBuyActionModel(
        action="buy",
        quantity=1,
        symbol="GOOG",
    )


@pytest.fixture
def initial_portfolio(uuid):
    yield Portfolio(
        initial_portfolio=pd.DataFrame(
            {
                "symbol": ["GOOG"],
                "buy_price": [100.0],
                "bought_date": [datetime.date.fromisoformat("2021-01-01")],
                "quantity": [1],
            },
            index=[uuid],
        )
    )


@pytest.fixture
def initial_state(initial_portfolio):
    return StateModel(portfolio=initial_portfolio, money_available=900.0)


@pytest.fixture
def simulator(initial_state):
    yield Simulator(initial_state=initial_state)


def test_simulator_init():
    Simulator()


def test_buy_stock_valid(simulator: Simulator, buy_action: StrategyBuyActionModel):
    day = type("Day", (), {"close": 200.0, "date": "2021-01-02"})
    simulator._buy_stock(buy_action, day)
    assert simulator._state.money_available == 700
    assert len(simulator._state.portfolio._portfolio) == 2
    assert simulator._state.portfolio._portfolio.symbol.to_list() == ["GOOG", "GOOG"]
    assert simulator._state.portfolio._portfolio.buy_price.to_list() == [100, 200]


def test_buy_stock_raises_when_not_enough_money(
    simulator: Simulator, buy_action: StrategyBuyActionModel
):
    buy_action.quantity = 1000
    day = type("Day", (), {"close": 100.0, "date": "2021-01-02"})
    with pytest.raises(SimulatorError) as e:
        simulator._buy_stock(buy_action, day)
    assert e.value.args[0] == "ur broke homie"


def test_sell_stock_valid(simulator: Simulator, uuid):
    sell_action = StrategySellActionModel(
        id=uuid,
        action_type="sell",
        quantity=1,
    )
    day = type("Day", (), {"close": 200.0, "date": "2021-01-02"})
    simulator._sell_stock(sell_action, day)
    # we gain 200 from selling
    assert simulator._state.money_available == 1100
    assert len(simulator._state.portfolio._portfolio) == 0


def test_sell_stock_non_existent_stock_raises(simulator: Simulator, uuid):
    sell_action = StrategySellActionModel(
        id=uuid4(),  # doesnt exist
        action_type="sell",
        quantity=1,
    )
    day = type("Day", (), {"close": 200.0, "date": "2021-01-02"})
    with pytest.raises(PortfolioError) as e:
        simulator._sell_stock(sell_action, day)
    assert (
        e.value.args[0]
        == f"Cannot sell stock {sell_action.id} because it does not exist"
    )


def test_run():
    """Simple simulation that starts with an empty portfolio and $100"""
    s = Simulator(
        initial_state=StateModel(portfolio=Portfolio(), money_available=10000)
    )
    s.run()
    print("HI")
