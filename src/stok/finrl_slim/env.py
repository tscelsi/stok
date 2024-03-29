from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv


class EnvBaseCallback(ABC):
    @abstractmethod
    def _on_episode_end(self, env: StockTradingEnv):
        raise NotImplementedError()

    @abstractmethod
    def _on_step(self, env: StockTradingEnv):
        raise NotImplementedError()


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dims: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        tech_indicator_list: list[str],
        reward_scaling: float = 1,
        turbulence_threshold: float | None = None,
        risk_indicator_col: str = "turbulence",
        day: int = 0,
        make_plots: bool = False,
        print_verbosity: int = 10,
        previous_state=None,
        type: Literal["train", "test"] = "train",
        **kwargs,
    ) -> None:
        """A stock trading environment built on Farama foundation's gymnasium

        Args:
            df (pd.DataFrame): The stock market data over time
            stock_dims (int): The number of stocks this env is trading
            hmax (int): Maximum number of stocks to buy/sell in one action
            initial_amount (int): Initial amount of money to trade with
            num_stock_shares (list[int]): Initial number of shares in portfolio
            buy_cost_pct (list[float]): Stockbroker fee for buying in percent (0 to 1)
            sell_cost_pct (list[float]): Stockbroker fee for selling in percent (0 to 1)
            reward_scaling (float): A scaling factor for the reward.
            tech_indicator_list (list[str]): List of technical indicators to use
                Each item in this list must be included in the df columns
            turbulence_threshold (float, optional): The turbulence indicator threshold
                over which to sell. If set, the risk_indicator_col must be set in df.
                    Defaults to None.
            risk_indicator_col (str, optional): Column in df to use as risk indicator
                column. Defaults to "turbulence".
            day (int, optional): Index of day to begin emulation. Defaults to 0.
            make_plots (bool, optional): Save a portfolio value over time plot to a
                file at the end of an episode. Defaults to False.
            print_verbosity (int, optional): After how many episodes to print a summary.
                Defaults to 10.
            previous_state (list, optional): The previous state of the environment can
                be thought of as seeding the state. Defaults to None.
            type (Literal[train | test]): The type of environment. Defaults to 'train'.

            DEPRECATED - initial (bool, optional):  Was used to distinguish between an
                env begun from scratch and an env passed in a previous state.
                Defaults to True.
        """
        if "initial" in kwargs:
            warnings.warn(
                "'initial' kwarg is deprecated, pass a 'previous_state' instead"
            )
        self.day = day
        self.type = type
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df = df.sort_values(["date", "tic"])
        self.df = df
        self.stock_dims = stock_dims
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = (
            previous_state[0] if previous_state is not None else initial_amount
        )
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list
        # balance + stock_dims * (prices + portfolio) + tech_dim * stock_dims
        self.state_dims = (
            1 + 2 * self.stock_dims + len(tech_indicator_list) * self.stock_dims
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.stock_dims,))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dims,)
        )
        self.turbulence_threshold = turbulence_threshold
        self.daily_data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.risk_indicator_col = risk_indicator_col
        self.previous_state = previous_state
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.env_episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dims])
            )
        ]  # the initial total asset is cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]

        self.callbacks = []
        self._seed()

    def get_stock_price(self, index):
        if index < self.stock_dims and index >= 0:
            return self.state[index + 1]
        raise IndexError("Index out of range")

    def get_stock_num(self, index):
        if index < self.stock_dims and index >= 0:
            return self.state[index + self.stock_dims + 1]
        raise IndexError("Index out of range")

    @property
    def _st_balance(self):
        return self.state[0]

    @property
    def _st_stock_prices(self) -> np.NDArray[float]:
        return np.array(self.state[1 : (self.stock_dims + 1)], dtype=float)

    @property
    def _st_stock_holdings(self) -> np.NDArray[int]:
        return np.array(
            self.state[(self.stock_dims + 1) : (self.stock_dims * 2 + 1)], dtype=int
        )

    def get_current_portfolio_value(self) -> float:
        return self._st_balance + sum(self._st_stock_prices * self._st_stock_holdings)

    def _sell_stock(self, index, action):
        curr_stock_price = self.get_stock_price(index)
        curr_stock_holding = self.get_stock_num(index)

        def _do_sell_normal():
            if curr_stock_price > 0:
                # check if the stock is able to sell
                # if we use price < 0 to denote a stock is unable to trade in that day,
                # the total asset calculation may be wrong if the price is unreasonable
                # - Sell only if the price is > 0
                #   i.e. (no missing data in this particular date)
                # - Perform sell action based on the sign of the action
                if curr_stock_holding > 0:
                    # Sell only if current asset is > 0
                    num_shares_to_sell = min(abs(action), curr_stock_holding)
                    sell_amount = (
                        curr_stock_price
                        * num_shares_to_sell
                        * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    self.state[0] += sell_amount
                    self.cost += sell_amount
                    # update holding
                    self.state[index + self.stock_dims + 1] -= num_shares_to_sell
                    self.trades += 1
                else:
                    num_shares_to_sell = 0
            else:
                num_shares_to_sell = 0

            return num_shares_to_sell

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if curr_stock_price > 0:
                    # Sell only if the price is > 0
                    # if turbulence goes over threshold, just sell all positions
                    if curr_stock_holding > 0:
                        # Sell only if current asset is > 0
                        # and sell everything
                        sell_amount = (
                            curr_stock_price
                            * curr_stock_holding
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dims + 1] = 0
                        self.cost += (
                            curr_stock_price
                            * curr_stock_holding
                            * self.sell_cost_pct[index]
                        )
                        num_shares_to_sell = curr_stock_holding
                        self.trades += 1
                    else:
                        num_shares_to_sell = 0
                else:
                    num_shares_to_sell = 0
            else:
                num_shares_to_sell = _do_sell_normal()
        else:
            num_shares_to_sell = _do_sell_normal()

        return num_shares_to_sell

    def _buy_stock(self, index, action):
        curr_stock_price = self.get_stock_price(index)

        def _do_buy():
            # if we're able to buy the stock
            # Buy only if the price is > 0 (no missing data in this particular date)
            if curr_stock_price > 0:
                # when buying stocks, we should consider the cost of trading when
                # computing the available amount
                available_amount = self.state[0] // (
                    curr_stock_price * (1 + self.buy_cost_pct[index])
                )

                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    curr_stock_price * buy_num_shares * (1 + self.buy_cost_pct[index])
                )
                # update balance
                self.state[0] -= buy_amount

                self.state[index + self.stock_dims + 1] += buy_num_shares

                self.cost += (
                    curr_stock_price * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
        return buy_num_shares

    def register_callbacks(self, callbacks: list[EnvBaseCallback]):
        self.callbacks.extend(callbacks)

    def _trigger_callbacks(self, method_name: str):
        for cb in self.callbacks:
            getattr(cb, method_name)(self)

    def step(self, actions: list[float]):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            self._trigger_callbacks("_on_episode_end")
            return self.state, self.reward, self.terminal, False, {}
        else:
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.astype(
                int
            )  # convert into integer because we can't buy fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dims)

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)

            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            self.daily_data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.daily_data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.daily_data[self.risk_indicator_col].values[0]
            self.state = self._update_state()
            self.asset_memory.append(self.get_current_portfolio_value())
            self.date_memory.append(self._get_date())
            reward = self.get_reward()
            scaled_reward = reward * self.reward_scaling
            self.rewards_memory.append(reward)
            self.reward = scaled_reward
            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step
        self._trigger_callbacks("_on_step")
        return self.state, self.reward, self.terminal, False, {}

    def get_sharpe(self) -> float | None:
        sharpe = None
        portfolio_ot = self.get_portfolio_memory_df()
        if portfolio_ot["daily_return"].std() != 0:
            sharpe = (
                (252**0.5)
                * portfolio_ot["daily_return"].mean()
                / portfolio_ot["daily_return"].std()
            )
        return sharpe

    def get_reward(self):
        """Override with your own reward function"""
        pre_action_portfolio_value = self.asset_memory[-2]
        post_action_portfolio_value = self.asset_memory[-1]
        return post_action_portfolio_value - pre_action_portfolio_value

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        # initiate state
        self.day = 0
        self.daily_data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.previous_state is None:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dims])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dims + 1)])
                * np.array(
                    self.previous_state[
                        (self.stock_dims + 1) : (self.stock_dims * 2 + 1)
                    ]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.env_episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.previous_state is None:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]
                    + self.daily_data.close.values.tolist()
                    + self.num_stock_shares
                    + sum(
                        (
                            self.daily_data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )  # append initial stocks_share to initial state, instead of all zero
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.daily_data.close]
                    + self.num_stock_shares
                    + sum(
                        ([self.daily_data[tech]] for tech in self.tech_indicator_list),
                        [],
                    )
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.daily_data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dims + 1) : (self.stock_dims * 2 + 1)
                    ]
                    + sum(
                        (
                            self.daily_data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.daily_data.close]
                    + self.previous_state[
                        (self.stock_dims + 1) : (self.stock_dims * 2 + 1)
                    ]
                    + sum(
                        ([self.daily_data[tech]] for tech in self.tech_indicator_list),
                        [],
                    )
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self._st_balance]
                + self.daily_data.close.values.tolist()
                + list(self._st_stock_holdings)
                + sum(
                    (
                        self.daily_data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.daily_data.close]
                + list(self._st_stock_holdings)
                + sum(
                    ([self.daily_data[tech]] for tech in self.tech_indicator_list), []
                )
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.daily_data.date.unique()[0]
        else:
            date = self.daily_data.date
        return date

    def get_rewards_memory_df(self):
        return pd.DataFrame(
            {
                "account_rewards": self.rewards_memory,
                "date": self.date_memory[:-1],
            }
        )

    def get_state_memory_df(self) -> pd.DataFrame:
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        return df_states

    def get_portfolio_memory_df(self) -> pd.DataFrame:
        """Returns a dataframe of the portfolio value over time as well as the daily
        return percentage."""
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        df_account_value["daily_return"] = df_account_value["account_value"].pct_change(
            1
        )
        return df_account_value

    def get_action_memory_df(self) -> pd.DataFrame:
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.daily_data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
