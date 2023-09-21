import pathlib
from typing import Any

import mlflow
import optuna
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecCheckNan

from ..stok_paths import ROOT_DIR
from .agent import DRLAgent
from .callbacks import (
    EpisodeSnapshotCallback,
    MlFlowOnEpisodeEndCallback,
    PrintEpisodeCallback,
)
from .config import RESULTS_DIR
from .env import StockTradingEnv


class Trainer:
    def __init__(
        self,
        train_env: StockTradingEnv,
        eval_env: StockTradingEnv | None,
        train_id: str,
        model_name: str = "ppo",
        checkpoint_save_freq: int = 5000,
        print_verbosity: int = 10,
        use_sb_callbacks: bool = True,
        use_mlflow: bool = True,
    ):
        self.train_env = train_env
        self.eval_env = eval_env
        self.train_id = train_id
        self.model_name = model_name
        self.ticker_list = train_env.df.tic.unique().tolist()
        self.ticker_id = "_".join(self.ticker_list)
        self.print_verbosity = print_verbosity
        self.use_sb_callbacks = use_sb_callbacks
        self.use_mlflow = use_mlflow
        # save options
        self._setup_train_dirs()
        if use_mlflow:
            self._init_mlflow()
        self.checkpoint_save_freq = checkpoint_save_freq

    def _init_mlflow(self):
        print("setting MLFlow tracking uri...")
        self.mlflow_dir = self.root_save_dir / "mlruns"
        if not self.mlflow_dir.exists():
            self.mlflow_dir.mkdir(exist_ok=True, parents=True)
        mlflow.set_tracking_uri(self.mlflow_dir)
        mlflow.create_experiment(self.train_id)
        mlflow.set_experiment(self.train_id)

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
        agent = DRLAgent(env=vec_env)
        model = agent.get_model(
            model_name=self.model_name,
            model_kwargs=hyperparameters,
        )
        new_logger = configure(str(self.root_save_dir / "log"), ["stdout", "csv"])
        model.set_logger(new_logger)
        callbacks = (
            self._configure_sb_callbacks(total_timesteps)
            if self.use_sb_callbacks
            else []
        )
        self._configure_env_callbacks(trial)
        trained_model = agent.train_model(
            model=model,
            total_timesteps=total_timesteps,
            callback=callbacks,
        )
        trained_model.save(self.curr_eval_dir / "fully_trained_model")
        return trained_model

    def _setup_train_dirs(self):
        self.root_save_dir = (
            get_base_dir(self.model_name, self.ticker_id) / self.train_id
        )
        self.root_chkpt_dir = self.root_save_dir / "checkpoints"
        self.root_eval_dir = self.root_save_dir / "eval"
        self.curr_chkpt_dir = self.root_chkpt_dir
        self.curr_eval_dir = self.root_eval_dir
        # create
        self.root_save_dir.mkdir(exist_ok=True, parents=True)
        self.root_chkpt_dir.mkdir(exist_ok=True, parents=True)
        self.root_eval_dir.mkdir(exist_ok=True, parents=True)

    def _setup_trial_dirs(self, trial: optuna.Trial):
        self.curr_chkpt_dir = self.root_chkpt_dir / f"trial_{str(trial.number)}"
        self.curr_eval_dir = self.root_eval_dir / f"trial_{str(trial.number)}"
        self.curr_chkpt_dir.mkdir(exist_ok=True, parents=True)
        self.curr_eval_dir.mkdir(exist_ok=True, parents=True)

    def _configure_sb_callbacks(
        self, total_timesteps: int, trial: optuna.Trial | None = None
    ) -> CallbackList:
        """Initialise the stable baselines callbacks for training."""
        name_prefix = f"trial_{trial.number}" if trial is not None else "chkpt"
        callbacks = [CheckpointCallback(
            save_freq=self.checkpoint_save_freq,
            save_path=str(self.curr_chkpt_dir),
            name_prefix=name_prefix,
        )]
        if self.eval_env:
            wrapped_env, _ = self.eval_env.get_sb_env()
            eval_cb = EvalCallback(
                wrapped_env,
                best_model_save_path=str(self.curr_eval_dir),
                log_path=str(self.curr_eval_dir),
                eval_freq=max(int(total_timesteps * 0.1), 1),
                deterministic=True,
                render=False,
                n_eval_episodes=1,
            )
            callbacks = [eval_cb] + callbacks
        print(f"Registering {len(callbacks)} training callbacks...")
        return CallbackList(callbacks)

    def _configure_env_callbacks(self, trial: optuna.Trial | None) -> None:
        if trial is not None:
            save_dir = self.root_save_dir / "snapshots" / f"trial_{trial.number}"
        else:
            save_dir = self.root_save_dir / "snapshots"
        callbacks = [
            PrintEpisodeCallback(),
            EpisodeSnapshotCallback(save_dir),
        ]
        if self.use_mlflow:
            callbacks.append(MlFlowOnEpisodeEndCallback())
        self.train_env.register_callbacks(callbacks)
        if self.eval_env is not None:
            self.eval_env.register_callbacks([MlFlowOnEpisodeEndCallback()])


def get_base_dir(model_name: str, ticker_id: str) -> pathlib.Path:
    """Retrieve the base directory for a model and ticker combination.

    Within this directory there may be logs, saved study optimisation checkpoints,
    models and study files."""
    return ROOT_DIR / RESULTS_DIR / model_name / ticker_id
