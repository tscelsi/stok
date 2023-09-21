from pathlib import Path

import pandas as pd


class Analysis:
    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)

    def get_rewards_over_time(self, episode: int):
        """Retrieves the reward over time for an episode."""
        reward_path = (
            self.base_dir / "snapshots" / f"episode_{episode}" / "account_rewards.csv"
        )
        if not reward_path.exists():
            raise FileNotFoundError(f"Reward file not found: {reward_path}")
        df = pd.read_csv(reward_path, index_col=0)
        df["episode"] = episode
        return df

    def get_portfolio_values_over_time(self, episode: int):
        portfolio_path = (
            self.base_dir / "snapshots" / f"episode_{episode}" / "portfolio_value.csv"
        )
        if not portfolio_path.exists():
            raise FileNotFoundError(f"Portfolio file not found: {portfolio_path}")
        df = pd.read_csv(portfolio_path, index_col=0)
        df["episode"] = episode
        return df

    def get_actions_over_time(self, episode: int):
        actions_path = (
            self.base_dir / "snapshots" / f"episode_{episode}" / "actions.csv"
        )
        if not actions_path.exists():
            raise FileNotFoundError(f"Actions file not found: {actions_path}")
        df = pd.read_csv(actions_path, index_col=0)
        df["episode"] = episode
        return df

    def all_rewards_over_time(self):
        """Return a dataframe with all the rewards over time for all episodes."""
        snapshot_path = self.base_dir / "snapshots"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {snapshot_path}")
        rewards = []
        for episode_dir in snapshot_path.iterdir():
            # get the numeric episode number from the directory name
            episode = int(episode_dir.name.split("_")[1])
            rewards.append(self.get_rewards_over_time(episode))
        return pd.concat(rewards)

    def all_portfolio_values_over_time(self):
        """Return a dataframe with all the portfolio values over time
        for all episodes."""
        snapshot_path = self.base_dir / "snapshots"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {snapshot_path}")
        portfolios = []
        for episode_dir in snapshot_path.iterdir():
            # get the numeric episode number from the directory name
            episode = int(episode_dir.name.split("_")[1])
            portfolios.append(self.get_portfolio_value_over_time(episode))
        return pd.concat(portfolios)

    def all_actions_over_time(self):
        """Return a dataframe with all the actions over time for all episodes."""
        snapshot_path = self.base_dir / "snapshots"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {snapshot_path}")
        actions = []
        for episode_dir in snapshot_path.iterdir():
            # get the numeric episode number from the directory name
            episode = int(episode_dir.name.split("_")[1])
            actions.append(self.get_actions_over_time(episode))
        return pd.concat(actions)

    def plot_episode(self, episode: int):
        rewards = self.get_actions_over_time(episode)
        