import numpy as np
import pandas as pd
import pytest

from paths import TEST_DIR
from stok.finrl_slim.env import StockTradingEnv


@pytest.fixture
def data_df():
    df = pd.read_csv(TEST_DIR / "finrl_slim" / "fixtures" / "goog.csv")
    yield df


@pytest.fixture
def multi_data_df():
    df = pd.read_csv(
        TEST_DIR / "finrl_slim" / "fixtures" / "goog_aapl.csv", index_col=0
    )
    yield df


@pytest.fixture()
def multi_env_kwargs():
    yield {
        "stock_dims": 2,
        "hmax": 100,
        "initial_amount": 10000,
        "num_stock_shares": [0, 0],
        "buy_cost_pct": [0.001, 0.001],
        "sell_cost_pct": [0.001, 0.001],
        "tech_indicator_list": [],
    }


@pytest.fixture()
def env_kwargs():
    yield {
        "stock_dims": 1,
        "hmax": 100,
        "initial_amount": 10000,
        "num_stock_shares": [0],
        "buy_cost_pct": [0.001],
        "sell_cost_pct": [0.001],
        "tech_indicator_list": [],
    }


@pytest.fixture
def setup_env(data_df, env_kwargs):
    yield StockTradingEnv(
        df=data_df,
        **env_kwargs,
    ), data_df


@pytest.fixture
def multi_setup_env(multi_data_df, multi_env_kwargs):
    yield StockTradingEnv(
        df=multi_data_df,
        **multi_env_kwargs,
    ), multi_data_df


def test_get_stock_price(setup_env):
    env: StockTradingEnv = setup_env[0]
    df: pd.DataFrame = setup_env[1]
    res = env.get_stock_price(0)
    # should be closing price of day 1
    assert res == df.iloc[0]["close"]


def test_multi_get_stock_price(multi_setup_env):
    env: StockTradingEnv = multi_setup_env[0]
    df: pd.DataFrame = multi_setup_env[1]
    assert env.get_stock_price(0) == df.iloc[0]["close"]
    assert env.get_stock_price(1) == df.iloc[1]["close"]
    with pytest.raises(IndexError):
        env.get_stock_price(2)


def test_get_stock_num(setup_env):
    env: StockTradingEnv = setup_env[0]
    assert env.get_stock_num(0) == 0
    with pytest.raises(IndexError):
        env.get_stock_num(1)


def test_multi_get_stock_num(multi_setup_env):
    env: StockTradingEnv = multi_setup_env[0]
    assert env.get_stock_num(0) == 0
    assert env.get_stock_num(1) == 0
    with pytest.raises(IndexError):
        env.get_stock_num(2)


def test_get_balance(setup_env):
    env: StockTradingEnv = setup_env[0]
    assert env._st_balance == 10000


def test_get_stock_prices(setup_env):
    env: StockTradingEnv = setup_env[0]
    df: pd.DataFrame = setup_env[1]
    assert env._st_stock_prices == [df.iloc[0]["close"]]


def test_multi_get_stock_prices(multi_setup_env):
    env: StockTradingEnv = multi_setup_env[0]
    df: pd.DataFrame = multi_setup_env[1]
    assert np.equal(env._st_stock_prices, df.iloc[:2]["close"].to_list()).all()


def test_get_stock_holdings(setup_env):
    env: StockTradingEnv = setup_env[0]
    assert env._st_stock_holdings == [0]


def test_multi_get_stock_holdings(multi_setup_env):
    env: StockTradingEnv = multi_setup_env[0]
    assert np.equal(env._st_stock_holdings, [0, 0]).all()


def test_env_init(setup_env):
    env: StockTradingEnv = setup_env[0]
    df: pd.DataFrame = setup_env[1]
    # balance, goog price, num shares
    assert env.state_dims == 3
    # initial amount + goog price * num shares
    assert env.asset_memory == [10000]
    # initial state is [balance, goog price, num shares]
    assert env.state == [10000, df.iloc[0]["close"], 0]


def test_multi_env_init(multi_setup_env):
    env: StockTradingEnv = multi_setup_env[0]
    df: pd.DataFrame = multi_setup_env[1]
    # balance, goog price, num shares
    assert env.state_dims == 5
    # initial amount + goog price * num shares
    assert env.asset_memory == [10000]
    # initial state is [balance, goog price, num shares]
    assert env.state == [10000, *df.iloc[:2]["close"].to_list(), 0, 0]


def test_sell_stock_succeeds(setup_env):
    env: StockTradingEnv = setup_env[0]
    df: pd.DataFrame = setup_env[1]
    # setup env with 10 shares of GOOG
    env.state = [10000, df.iloc[0]["close"], 10]
    assert env.turbulence_threshold is None
    num_shares_to_sell = env._sell_stock(0, 2)
    # we are selling two shares
    assert num_shares_to_sell == 2
    # expect 8 shares left
    stock_price = df.iloc[0]["close"]
    # take into account the sell cost percentage
    sell_price = 2 * stock_price * (1 - env.sell_cost_pct[0])
    assert env.state == [10000 + sell_price, stock_price, 8]


def test_sell_stock_more_than_amount_holding(setup_env):
    """When selling more than the amount the portfolio is holding,
    we should sell all the shares but not more."""
    env: StockTradingEnv = setup_env[0]
    df: pd.DataFrame = setup_env[1]
    # setup env with 10 shares of GOOG
    env.state = [10000, df.iloc[0]["close"], 5]
    assert env.turbulence_threshold is None
    num_shares_to_sell = env._sell_stock(0, 10)
    # we are selling two shares
    assert num_shares_to_sell == 5
    # expect 0 shares left
    stock_price = df.iloc[0]["close"]
    # take into account the sell cost percentage
    sell_price = 5 * stock_price * (1 - env.sell_cost_pct[0])
    assert env.state == [10000 + sell_price, stock_price, 0]
    assert env.trades == 1


def test_buy_stock_succeeds(setup_env):
    env: StockTradingEnv = setup_env[0]
    df: pd.DataFrame = setup_env[1]
    stock_price = df.iloc[0]["close"]
    # setup env with 10 shares of GOOG
    env.state = [10000, stock_price, 0]
    assert env.turbulence_threshold is None
    assert stock_price * 10 < 10000
    num_shares_to_buy = env._buy_stock(0, 10)
    # we are buying 10 shares
    assert num_shares_to_buy == 10
    # expect 10 shares left
    # take into account the buy cost percentage
    buy_price = 10 * stock_price * (1 + env.buy_cost_pct[0])
    assert env.state == [10000 - buy_price, stock_price, 10]
    assert env.trades == 1


def test_buy_stock_when_not_enough_money(setup_env):
    """The environment should only allow us to buy as many shares as we can afford."""
    env: StockTradingEnv = setup_env[0]
    df: pd.DataFrame = setup_env[1]
    stock_price = df.iloc[0]["close"]
    initial_balance = 10000
    # the buy amount here requests to buy one more than is possible
    # with the initial balance of 10000
    excess_buy_amount = (initial_balance // (stock_price * (1 + 0.001))) + 1
    env.state = [initial_balance, stock_price, 0]
    assert env.turbulence_threshold is None
    import math

    num_shares_to_buy = env._buy_stock(0, math.floor(excess_buy_amount))
    # the environment only allows us to buy as many shares as we can afford
    # which is one less than the excess buy amount calculated above
    assert num_shares_to_buy == excess_buy_amount - 1
    # take into account the buy cost percentage
    buy_price = (excess_buy_amount - 1) * stock_price * (1 + env.buy_cost_pct[0])
    assert env.state == [10000 - buy_price, stock_price, excess_buy_amount - 1]
    assert env.trades == 1


def test_step_buy_action(data_df: pd.DataFrame, env_kwargs: dict):
    env: StockTradingEnv = StockTradingEnv(
        df=data_df,
        **{
            **env_kwargs,
            "num_stock_shares": [0],
        },
    )
    df = data_df
    # action is a number between -1 and 1. This gets converted to a number of shares
    # to buy or sell based on the hmax value passed into the constructor.
    # since our hmax=100, an action of 0.4 gets converted into a buy request
    # for 40 shares. We wrap it in a list because the environment expects an action
    # for each stock in the portfolio.
    initial_portfolio_value = 10000
    actions = np.array([0.4])
    new_state, reward, terminated, truncated, info = env.step(actions)
    assert not terminated
    assert not truncated
    assert info == {}
    # assert all values updated correctly for buy
    day_1_stock_price = df.iloc[0]["close"]
    day_2_stock_price = df.iloc[1]["close"]
    new_balance = initial_portfolio_value - (
        40 * day_1_stock_price * (1 + env.buy_cost_pct[0])
    )
    assert new_state == [new_balance, day_2_stock_price, 40]
    assert env._st_stock_holdings == [40]
    assert env._st_balance == new_balance
    assert env.trades == 1
    # new stock price in state
    assert env._st_stock_prices == [df.iloc[1]["close"]]
    assert env.cost == 40 * day_1_stock_price * env.buy_cost_pct[0]
    # test analytics updates
    # the actions memory is the buy/sell amounts of each stock in full value not float
    assert env.actions_memory == [np.array([40])]
    # the asset memory is a history of the portfolio value
    new_portfolio_value = new_balance + (40 * df.iloc[1]["close"])
    assert env.get_current_portfolio_value() == new_portfolio_value
    assert env.asset_memory == [10000, new_portfolio_value]
    # The reward is just the increase in portfolio value
    assert reward == new_portfolio_value - initial_portfolio_value
    assert env.reward == new_portfolio_value - initial_portfolio_value
    assert env.rewards_memory == [new_portfolio_value - initial_portfolio_value]
    # state memory
    assert env.state_memory == [
        [new_balance, day_2_stock_price, 40],
    ]


def test_multi_step_buy_actions(multi_data_df: pd.DataFrame, multi_env_kwargs: dict):
    env: StockTradingEnv = StockTradingEnv(
        df=multi_data_df,
        **{
            **multi_env_kwargs,
            "num_stock_shares": [0, 0],
        },
    )
    df = multi_data_df
    # action is a number between -1 and 1. This gets converted to a number of shares
    # to buy or sell based on the hmax value passed into the constructor.
    # since our hmax=100, an action of 0.4 gets converted into a buy request
    # for 40 shares. We wrap it in a list because the environment expects an action
    # for each stock in the portfolio.
    initial_portfolio_value = 10000
    # buy 40 AAPL and 10 GOOG
    actions = np.array([0.4, 0.1])
    new_state, reward, terminated, truncated, info = env.step(actions)
    assert not terminated
    assert not truncated
    assert info == {}
    # assert all values updated correctly for buy
    day_1_aapl_stock_price = df.iloc[0]["close"]
    day_1_goog_stock_price = df.iloc[1]["close"]
    day_2_aapl_stock_price = df.iloc[2]["close"]
    day_2_goog_stock_price = df.iloc[3]["close"]
    new_balance = initial_portfolio_value - (
        40 * day_1_aapl_stock_price * (1 + env.buy_cost_pct[0])
        + 10 * day_1_goog_stock_price * (1 + env.buy_cost_pct[0])
    )
    assert np.isclose(
        new_state,
        [
            new_balance,
            day_2_aapl_stock_price,
            day_2_goog_stock_price,
            40,
            10,
        ],
    ).all()
    assert np.isclose(env._st_stock_holdings, [40, 10]).all()
    assert np.isclose(env._st_balance, new_balance)
    assert env.trades == 2
    # new stock price in state
    assert np.isclose(
        env._st_stock_prices, [df.iloc[2]["close"], df.iloc[3]["close"]]
    ).all()
    assert np.isclose(
        env.cost,
        40 * day_1_aapl_stock_price * env.buy_cost_pct[0]
        + 10 * day_1_goog_stock_price * env.buy_cost_pct[1],
    )
    # test analytics updates
    # the actions memory is the buy/sell amounts of each stock in full value not float
    assert np.isclose(env.actions_memory, [np.array([40, 10])]).all()
    # the asset memory is a history of the portfolio value
    new_portfolio_value = (
        new_balance + (40 * df.iloc[2]["close"]) + (10 * df.iloc[3]["close"])
    )
    assert np.isclose(env.get_current_portfolio_value(), new_portfolio_value)
    assert np.isclose(env.asset_memory, [10000, new_portfolio_value]).all()
    # The reward is just the increase in portfolio value
    assert np.isclose(reward, new_portfolio_value - initial_portfolio_value)
    assert np.isclose(env.reward, new_portfolio_value - initial_portfolio_value)
    assert np.isclose(
        env.rewards_memory, [new_portfolio_value - initial_portfolio_value]
    ).all()
    # state memory
    assert np.isclose(
        env.state_memory,
        [
            [new_balance, day_2_aapl_stock_price, day_2_goog_stock_price, 40, 10],
        ],
    ).all()


def test_step_sell_action(data_df, env_kwargs):
    env: StockTradingEnv = StockTradingEnv(
        df=data_df,
        **{
            **env_kwargs,
            "num_stock_shares": [100],
        },
    )
    df = data_df
    # action is a number between -1 and 1. This gets converted to a number of shares
    # to buy or sell based on the hmax value passed into the constructor.
    # since our hmax=100, an action of -0.4 gets converted into a sell request
    # for 40 shares. We wrap it in a list because the environment expects an action
    # for each stock in the portfolio.
    day_1_stock_price = df.iloc[0]["close"]
    day_2_stock_price = df.iloc[1]["close"]
    initial_portfolio_value = 10000 + (100 * day_1_stock_price)
    # start with 100 shares
    assert env.state == [10000, day_1_stock_price, 100]
    actions = np.array([-0.4])
    new_state, reward, terminated, truncated, info = env.step(actions)
    assert not terminated
    assert not truncated
    assert info == {}
    # assert all values updated correctly for buy
    new_balance = 10000 + (40 * day_1_stock_price * (1 - env.buy_cost_pct[0]))
    assert new_state == [new_balance, day_2_stock_price, 60]
    assert env._st_stock_holdings == [60]
    assert env._st_balance == new_balance
    assert env.trades == 1
    # new day 2 stock price in state
    assert env._st_stock_prices == [day_2_stock_price]
    assert env.cost == 40 * day_1_stock_price * (1 - env.buy_cost_pct[0])
    # test analytics updates
    # the actions memory is the buy/sell amounts of each stock in full value not float
    assert env.actions_memory == [np.array([-40])]
    # the asset memory is a history of the portfolio value
    new_portfolio_value = new_balance + (60 * day_2_stock_price)
    assert env.get_current_portfolio_value() == new_portfolio_value
    assert env.asset_memory == [initial_portfolio_value, new_portfolio_value]
    # The reward is just the increase in portfolio value
    assert reward == new_portfolio_value - initial_portfolio_value
    assert env.reward == new_portfolio_value - initial_portfolio_value
    assert env.rewards_memory == [new_portfolio_value - initial_portfolio_value]
    # state memory
    assert env.state_memory == [
        [new_balance, day_2_stock_price, 60],
    ]


def test_reset(setup_env: StockTradingEnv):
    env = setup_env[0]
    df = setup_env[1]
    init_state, observations = env.reset()
    day_1_stock_price = df.iloc[0]["close"]
    assert init_state == [10000, day_1_stock_price, 0]
    assert observations == {}
    assert env.state == [10000, day_1_stock_price, 0]
    assert env.asset_memory == [10000]
    assert env.state_memory == []
    assert env.actions_memory == []
    assert env.rewards_memory == []
    assert env.trades == 0
    assert env.cost == 0
    assert env.reward == 0
    assert not env.terminal
    assert env.turbulence == 0
    # episode has been incremented
    assert env.episode == 1


def test_multi_reset(multi_setup_env: StockTradingEnv):
    env = multi_setup_env[0]
    df = multi_setup_env[1]
    init_state, observations = env.reset()
    day_1_aapl_stock_price = df.iloc[0]["close"]
    day_1_goog_stock_price = df.iloc[1]["close"]
    assert init_state == [10000, day_1_aapl_stock_price, day_1_goog_stock_price, 0, 0]
    assert observations == {}
    assert env.state == [10000, day_1_aapl_stock_price, day_1_goog_stock_price, 0, 0]
    assert env.asset_memory == [10000]
    assert env.state_memory == []
    assert env.actions_memory == []
    assert env.rewards_memory == []
    assert env.trades == 0
    assert env.cost == 0
    assert env.reward == 0
    assert not env.terminal
    assert env.turbulence == 0
    # episode has been incremented
    assert env.episode == 1


def test_reset_with_initial_portfolio(env_kwargs: dict, data_df: pd.DataFrame):
    env = StockTradingEnv(
        df=data_df,
        **{
            **env_kwargs,
            "initial_amount": 20000,
            "num_stock_shares": [100],
        },
    )
    df = data_df
    init_state, observations = env.reset()
    day_1_stock_price = df.iloc[0]["close"]
    init_portfolio_value = 20000 + (100 * day_1_stock_price)
    assert init_state == [20000, day_1_stock_price, 100]
    assert observations == {}
    assert env.state == [20000, day_1_stock_price, 100]
    assert env.asset_memory == [init_portfolio_value]
    assert env.state_memory == []
    assert env.actions_memory == []
    assert env.rewards_memory == []
    assert env.trades == 0
    assert env.cost == 0
    assert env.reward == 0
    assert not env.terminal
    assert env.turbulence == 0
    # episode has been incremented
    assert env.episode == 1


def test_get_asset_memory_df(setup_env):
    env: StockTradingEnv = setup_env[0]
    df: pd.DataFrame = setup_env[1]
    actions = np.array([0.4])
    env.step(actions)
    portfolio_over_time_df = env.get_asset_memory_df()
    assert len(portfolio_over_time_df) == 2
    assert portfolio_over_time_df["account_value"].iloc[0] == 10000
    assert portfolio_over_time_df.date.to_list() == df.iloc[:2].date.to_list()


def test_get_action_memory_df(setup_env):
    env: StockTradingEnv = setup_env[0]
    df: pd.DataFrame = setup_env[1]
    actions = np.array([0.4])
    env.step(actions)
    actions_over_time_df = env.get_action_memory_df()
    # took a buy action on the first day
    assert len(actions_over_time_df) == 1
    assert actions_over_time_df["actions"].iloc[0] == [40]
    assert actions_over_time_df.date.to_list() == df.iloc[:1].date.to_list()


def test_get_state_memory_df(setup_env):
    """The memory tracks the state of the environment at close of each day.
    In the state, we have the balance, the stock price and how many stocks
    are being held.
    """
    env: StockTradingEnv = setup_env[0]
    df: pd.DataFrame = setup_env[1]
    actions = np.array([0.4])
    env.step(actions)
    states_over_time_df = env.get_state_memory_df()
    assert len(states_over_time_df) == 1
    assert states_over_time_df.date.to_list() == df.iloc[:1].date.to_list()
