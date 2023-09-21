from pathlib import Path

import numpy as np
from mlflow import log_metric, log_table

from .env import EnvBaseCallback, StockTradingEnv


class DefaultEnvCallback(EnvBaseCallback):
    def _on_episode_end(self, env: StockTradingEnv):
        pass

    def _on_step(self, env: StockTradingEnv):
        pass


class MlFlowOnEpisodeEndCallback(DefaultEnvCallback):
    """Logs basic metrics to mlflow at the end of an episode"""

    def _on_episode_end(self, env: StockTradingEnv) -> None:
        rewards_memory = np.array(env.rewards_memory)
        portfolio_value_metric = (
            "portfolio_value" if env.type == "train" else "portfolio_value_eval"
        )
        total_reward_metric = (
            "total_reward" if env.type == "train" else "total_reward_eval"
        )
        mean_reward_metric = (
            "mean_reward" if env.type == "train" else "mean_reward_eval"
        )
        total_trades_metric = (
            "total_trades" if env.type == "train" else "total_trades_eval"
        )
        sharpe_metric = "sharpe" if env.type == "train" else "sharpe_eval"
        log_metric(
            portfolio_value_metric,
            env.get_current_portfolio_value(),
            step=env.env_episode,
        )
        log_metric(total_reward_metric, rewards_memory.sum(), step=env.env_episode)
        log_metric(mean_reward_metric, rewards_memory.mean(), step=env.env_episode)
        log_metric(total_trades_metric, env.trades, step=env.env_episode)
        log_metric(sharpe_metric, env.get_sharpe(), step=env.env_episode)
        log_table(
            env.get_portfolio_memory_df(), f"episode_{env.env_episode}_portfolio.json"
        )


class PrintEpisodeCallback(DefaultEnvCallback):
    """Callback for printing stock environment results at the end of an episode"""

    def __init__(
        self,
        print_verbosity: int = 10,
    ) -> None:
        super().__init__()
        self.print_verbosity = print_verbosity

    def _on_episode_end(self, env: StockTradingEnv) -> None:
        if env.env_episode % self.print_verbosity == 0:
            asset_memory = env.asset_memory
            rewards_memory = np.array(env.rewards_memory)
            sharpe = env.get_sharpe()
            print("=============== EPISODE STATS ==================")
            print(f"END DAY: {env.day}, EPISODE: {env.env_episode}")
            print(f"Start portfolio value: {asset_memory[0]:0.2f}")
            print("End portfolio value: " + f"{env.get_current_portfolio_value():0.2f}")
            print(f"Total reward: {rewards_memory.sum():0.2f}")
            print(f"Total cost: {env.cost:0.2f}")
            print(f"Total trades: {env.trades}")
            if sharpe is not None:
                print(f"Sharpe: {sharpe:0.3f}")
            else:
                print(f"Sharpe: {sharpe}")
            print("================================================")


class EpisodeSnapshotCallback(DefaultEnvCallback):
    """Saves episode information to files

    This includes the actions taken, the portfolio value over time, and the rewards
    over time.
    """

    def __init__(
        self,
        base_path: str | Path,
    ) -> None:
        super().__init__()
        self.base_path = Path(base_path)

    def _on_episode_end(self, env: StockTradingEnv) -> None:
        """A snapshot of the episode saved to a base directory"""
        save_dir = self.base_path / f"episode_{env.env_episode}"
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        df_actions = env.get_action_memory_df()
        df_actions.to_pickle(save_dir / "actions.pkl")
        portfolio_value_over_time = env.get_portfolio_memory_df()
        portfolio_value_over_time.to_pickle(save_dir / "portfolio_value.pkl")
        rewards_memory_df = env.get_rewards_memory_df()
        rewards_memory_df.to_pickle(save_dir / "account_rewards.pkl")
