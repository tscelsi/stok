import pandas as pd
import pytest
from polyfactory.factories.pydantic_factory import ModelFactory

from paths import ROOT_DIR
from stok.emulator import Emulator, EmulatorError, StateModel
from stok.portfolio import Portfolio, PortfolioError
from stok.strategies import StrategyBuyActionModel, StrategySellActionModel


@pytest.fixture
def buy_action():
    return StrategyBuyActionModel(
        quantity=1,
        symbol="GOOG",
    )


@pytest.fixture
def initial_portfolio():
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
    return StateModel(portfolio=initial_portfolio, available_funds=900.0)


@pytest.fixture
def emulator(initial_state):
    yield Emulator(initial_state=initial_state)


def test_simulator_init():
    Emulator()


def test_buy_stock_valid(emulator: Emulator, buy_action_factory: ModelFactory):
    buy_action = buy_action_factory.build(symbol="GOOG", quantity=1)
    emulator._buy_stock(buy_action, 200)
    assert emulator._state.available_funds == 700
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
    assert emulator._state.available_funds == 1100
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


def test_portfolio_value(emulator: Emulator):
    # assuming a GOOG stock price of $100
    # we have one GOOG stock and 900 in our wallet.
    assert emulator._portfolio_value(100) == 1000
    assert emulator._portfolio_value(500) == 1400


def test_update_portfolio_history(emulator: Emulator):
    emulator._update_portfolio_history(pd.Timestamp("2020-01-01"), 100)
    assert len(emulator._portfolio_history) == 1
    assert emulator._portfolio_history.iloc[0].to_dict() == {
        "symbol": "GOOG",
        "quantity": 1,
        "unit_price": 100,
        "holding_value": 100,
    }
    assert emulator._portfolio_history.index.to_list() == [pd.Timestamp("2020-01-01")]


def test_update_action_history(emulator: Emulator, buy_action):
    emulator._update_action_history(pd.Timestamp("2020-01-01"), buy_action)
    assert len(emulator._action_history) == 1
    assert emulator._action_history.iloc[0].to_dict() == {
        "symbol": "GOOG",
        "quantity": 1,
        "action_type": "buy",
    }
    emulator._update_action_history(pd.Timestamp("2020-01-01"), buy_action)
    assert len(emulator._action_history) == 2
    assert emulator._action_history.quantity.to_list() == [1, 1]


def test_run():
    """Simple simulation that starts with an empty portfolio and $100"""
    s = Emulator(initial_state=StateModel(portfolio=Portfolio(), available_funds=10000))
    days = pd.read_csv(ROOT_DIR / "goog.csv", index_col="Date", parse_dates=True)
    s.run(days)
    print("HI")
