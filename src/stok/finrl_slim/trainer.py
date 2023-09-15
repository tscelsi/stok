import pathlib
from typing import Any

import optuna
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecCheckNan

from ..stok_paths import ROOT_DIR
from .agent import DRLAgent
from .config import RESULTS_DIR
from .env import StockTradingEnv


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True


class Trainer:
    def __init__(
        self,
        train_env: StockTradingEnv,
        eval_env: StockTradingEnv | None,
        train_id: str,
        model_name: str = "ppo",
        save_freq: int = 5000,
    ):
        self.train_env = train_env
        self.eval_env = eval_env
        self.train_id = train_id
        self.model_name = model_name
        self.ticker_list = train_env.df.tic.unique().tolist()
        self.ticker_id = "_".join(self.ticker_list)
        # save options
        self._setup_train_dirs()
        self.save_freq = save_freq

    def train(
        self,
        trial: optuna.Trial | None = None,
        hyperparameters: dict[str, Any] | None = None,
        total_timesteps: int = 10000,
    ):
        if trial is not None:
            self._setup_trial_dirs(trial)
        vec_env, _ = self.train_env.get_sb_env()
        # catch nan wrapper
        vec_env = VecCheckNan(vec_env, raise_exception=True)
        agent = DRLAgent(env=self.train_env)
        model = agent.get_model(
            model_name=self.model_name,
            model_kwargs=hyperparameters,
            tensorboard_log=str(self.curr_tensorboard_log_dir),
        )
        new_logger = configure(str(self.root_save_dir / "log"), ["stdout", "csv"])
        model.set_logger(new_logger)
        callbacks = self._configure_callbacks(total_timesteps)
        tb_log_name = (
            "_".join([str(trial.number), self.train_id])
            if trial is not None
            else self.train_id
        )
        trained_model = agent.train_model(
            model=model,
            tb_log_name=tb_log_name,
            total_timesteps=total_timesteps,
            callback=callbacks,
        )
        return trained_model

    def _setup_train_dirs(self):
        self.root_save_dir = get_base_dir(self.model_name, self.ticker_id)
        self.root_chkpt_dir = self.root_save_dir / "checkpoints"
        self.root_eval_dir = self.root_save_dir / "eval"
        self.root_tensorboard_log_dir = self.root_save_dir / "tensorboard"
        self.curr_chkpt_dir = self.root_chkpt_dir
        self.curr_eval_dir = self.root_eval_dir
        self.curr_tensorboard_log_dir = self.root_tensorboard_log_dir
        # create
        self.root_save_dir.mkdir(exist_ok=True, parents=True)
        self.root_chkpt_dir.mkdir(exist_ok=True, parents=True)
        self.root_eval_dir.mkdir(exist_ok=True, parents=True)
        self.root_tensorboard_log_dir.mkdir(exist_ok=True, parents=True)

    def _setup_trial_dirs(self, trial: optuna.Trial):
        self.curr_chkpt_dir = self.root_chkpt_dir / f"trial_{str(trial.number)}"
        self.curr_eval_dir = self.root_eval_dir / f"trial_{str(trial.number)}"
        self.curr_tensorboard_log_dir = self.root_tensorboard_log_dir / str(
            trial.number
        )
        self.curr_chkpt_dir.mkdir(exist_ok=True, parents=True)
        self.curr_eval_dir.mkdir(exist_ok=True, parents=True)
        self.curr_tensorboard_log_dir.mkdir(exist_ok=True, parents=True)

    def _configure_callbacks(
        self, total_timesteps: int, trial: optuna.Trial | None = None
    ) -> CallbackList:
        name_prefix = f"trial_{trial.number}" if trial is not None else "chkpt"
        checkpoint_cb = CheckpointCallback(
            save_freq=self.save_freq,
            save_path=str(self.curr_chkpt_dir),
            name_prefix=name_prefix,
        )
        wrapped_env, _ = self.eval_env.get_sb_env()
        eval_cb = EvalCallback(
            wrapped_env,
            best_model_save_path=str(self.curr_eval_dir),
            log_path=str(self.curr_eval_dir),
            eval_freq=max(int(total_timesteps * 0.1), 1),
            deterministic=True,
            render=False,
        )
        tb_cb = TensorboardCallback()
        callbacks = CallbackList([checkpoint_cb, eval_cb, tb_cb])
        return callbacks


def get_base_dir(model_name: str, ticker_id: str) -> pathlib.Path:
    """Retrieve the base directory for a model and ticker combination.

    Within this directory there may be logs, saved study optimisation checkpoints,
    models and study files."""
    return ROOT_DIR / RESULTS_DIR / model_name / ticker_id
