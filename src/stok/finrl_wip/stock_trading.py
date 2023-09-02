from __future__ import annotations

import itertools
import sys

import pandas as pd
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from ..config import DATA_SAVE_DIR
from ..config import INDICATORS
from ..config import RESULTS_DIR
from ..config import TENSORBOARD_LOG_DIR
from ..config import TRAINED_MODEL_DIR
from ..config import DOW_30_TICKER
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from ..preprocessing.preprocessors import data_split
from ..preprocessing.preprocessors import FeatureEngineer
from ..preprocessing.yahoodownloader import YahooDownloader
from finrl.plot import backtest_stats
from finrl.plot import get_baseline
from finrl.plot import plot_return


def stock_trading(
    train_start_date: str,
    train_end_date: str,
    trade_start_date: str,
    trade_end_date: str,
    if_store_actions: bool = True,
    if_store_result: bool = True,
    ticker_list: list[str] = DOW_30_TICKER,
):
    sys.path.append("../FinRL")
    check_and_make_directories(
        [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
    )
    date_col = "date"
    tic_col = "tic"
    df = YahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=ticker_list
    ).fetch_data()
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)
    list_ticker = processed[tic_col].unique().tolist()
    list_date = list(
        pd.date_range(processed[date_col].min(), processed[date_col].max()).astype(str)
    )
    combination = list(itertools.product(list_date, list_ticker))

    init_train_trade_data = pd.DataFrame(
        combination, columns=[date_col, tic_col]
    ).merge(processed, on=[date_col, tic_col], how="left")
    init_train_trade_data = init_train_trade_data[
        init_train_trade_data[date_col].isin(processed[date_col])
    ]
    init_train_trade_data = init_train_trade_data.sort_values([date_col, tic_col])

    init_train_trade_data = init_train_trade_data.fillna(0)

    init_train_data = data_split(
        init_train_trade_data, train_start_date, train_end_date
    )
    init_trade_data = data_split(
        init_train_trade_data, trade_start_date, trade_end_date
    )

    stock_dimension = len(init_train_data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    initial_amount = 1000000
    env_kwargs = {
        "hmax": 100,
        "initial_amount": initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    e_train_gym = StockTradingEnv(df=init_train_data, **env_kwargs)

    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))

    agent = DRLAgent(env=env_train)
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    }
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
    # set up logger
    tmp_path = RESULTS_DIR + "/ppo"
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_ppo.set_logger(new_logger_ppo)
    trained_ppo = agent.train_model(
        model=model_ppo, tb_log_name="ppo", total_timesteps=50000
    )

    # trade
    e_trade_gym = StockTradingEnv(
        df=init_trade_data,
        turbulence_threshold=70,
        risk_indicator_col="vix",
        **env_kwargs,
    )
    # env_trade, obs_trade = e_trade_gym.get_sb_env()

    result_ppo, actions_ppo = DRLAgent.DRL_prediction(
        model=trained_ppo, environment=e_trade_gym
    )

    if isinstance(result_ppo, tuple):
        actions_ppo = result_ppo[1]
        result_ppo = result_ppo[0]

    # store actions
    if if_store_actions:
        actions_ppo.to_csv("actions_ppo.csv")

    # dji
    dji_ = get_baseline(ticker="^DJI", start=trade_start_date, end=trade_end_date)
    dji = pd.DataFrame()
    dji[date_col] = dji_[date_col]
    dji["DJI"] = dji_["close"]
    # select the rows between trade_start and trade_end (not included), since some values may not in this region
    dji = dji.loc[
        (dji[date_col] >= trade_start_date) & (dji[date_col] < trade_end_date)
    ]

    result = dji

    result_ppo.rename(columns={"account_value": "PPO"}, inplace=True)
    result = pd.merge(result, result_ppo, how="left")

    # remove the rows with nan
    result = result.dropna(axis=0, how="any")

    # calc the column name of strategies, including DJI
    col_strategies = []
    for col in result.columns:
        if col != date_col and col != "" and "Unnamed" not in col:
            col_strategies.append(col)

    # make sure that the first row of DJI is initial_amount
    col = "DJI"
    result[col] = result[col] / result[col].iloc[0] * initial_amount
    result = result.reset_index(drop=True)

    # stats
    for col in col_strategies:
        stats = backtest_stats(result, value_col_name=col)
        print("\nstats of " + col + ": \n", stats)

    # print and save result
    print("result: ", result)
    if if_store_result:
        result.to_csv("result.csv")

    # plot fig
    plot_return(
        result=result,
        column_as_x=date_col,
        if_need_calc_return=True,
        savefig_filename="stock_trading.png",
        xlabel="Date",
        ylabel="Return",
        if_transfer_date=True,
        num_days_xticks=20,
    )


if __name__ == "__main__":
    train_start_date = "2009-01-01"
    train_end_date = "2022-09-01"
    trade_start_date = "2022-09-01"
    trade_end_date = "2023-11-01"
    if_store_actions = True
    if_store_result = True

    stock_trading(
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        trade_start_date=trade_start_date,
        trade_end_date=trade_end_date,
        if_store_actions=if_store_actions,
        if_store_result=if_store_result,
    )