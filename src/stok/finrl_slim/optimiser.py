import pathlib
from typing import Any

import joblib
import mlflow
import optuna
import pandas as pd

from ..stok_paths import ROOT_DIR
from .agent import MODELS, DRLAgent
from .config import RESULTS_DIR
from .env import StockTradingEnv
from .trainer import Trainer

# def evaluate_models(
#     test_data: pd.DataFrame,
#     model_name: str = "ppo",
#     indicators: list[str] = INDICATORS,
#     initial_amount: float = 10000,
# ):
#     """Evaluate the performance of a model on each ticker in a test dataset.
#     For this function to work, the test_data must have a column called "tic" and
#     there must exist a model for each ticker and model_name in the RESULTS_DIR."""
#     result_dfs = []
#     action_dfs = []
#     for ticker in test_data.tic.unique().tolist():
#         test_data_subset = test_data[test_data.tic == ticker]
#         portfolio_ot, actions_ot = test_once(
#             test_data_subset,
#             model_name=model_name,
#             indicators=indicators,
#         )
#         portfolio_ot.set_index("date", inplace=True)
#         portfolio_ot.index = pd.to_datetime(portfolio_ot.index, format="%Y-%m-%d")
#         portfolio_ot.index = portfolio_ot.index.date
#         hold_strategy_result = get_hold_strategy_value_ot(
#             ticker, test_data_subset, initial_amount
#         )
#         result_df = pd.concat([portfolio_ot, hold_strategy_result], axis=1)
#         result_df["tic"] = ticker
#         result_df = result_df.reset_index()
#         result_dfs.append(result_df)
#         action_dfs.append(actions_ot)
#     return pd.concat(result_dfs), action_dfs


def test_once(
    test_env: StockTradingEnv,
    model_path: str,
    model_name: str = "ppo",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test a model on a test dataset.

    Args:
        test_data (pd.DataFrame): The test dataset. Should contain close prices and
            technical indicators for a single ticker over time.
        trial (optuna.Trial, optional): Used in the case of hyperparameter
            optimisation to track the trial number. Defaults to None.
        model_name (str, optional): The name of the model architecture to use.
            Defaults to "ppo".
        indicators (list[str], optional): The technical indicators to use.
            Defaults to INDICATORS.

    Raises:
        ValueError: If an invalid model_name is passed.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The portfolio value over time and the
            actions taken over time.
    """
    print("loading trained model...")
    model = MODELS.get(model_name, None)
    if model is None:
        raise NotImplementedError("model_name is not supported")
    model = model.load(model_path)
    print("loaded model!")
    print("making predictions on test data...")
    portfolio_value_ot_df, actions_ot_df = DRLAgent.DRL_prediction(
        model=model, environment=test_env
    )
    print("finish making predictions!")
    return portfolio_value_ot_df, actions_ot_df


class SharpeObjective:
    """Defines an objective function for hyperparameter optimisation where
    the sharpe ratio is maximised.
    """

    def __init__(
        self,
        train_env: StockTradingEnv,
        eval_env: StockTradingEnv,
        trainer: Trainer,
        model_name: str = "ppo",
        total_timesteps: int = 10000,
    ):
        self.train_env = train_env
        self.eval_env = eval_env
        self.model_name = model_name
        self.trainer = trainer
        self.total_timesteps = total_timesteps

    def __call__(self, trial: optuna.Trial) -> Any:
        params = sample_hyperparams(trial, self.model_name)
        if self.trainer.use_mlflow:
            mlflow.start_run(run_name=str(trial.number))
            mlflow.log_params(params)
        self.trainer.train(
            trial=trial,
            hyperparameters=params,
            total_timesteps=self.total_timesteps,
        )
        eval_model_path = self.trainer.curr_eval_dir / "best_model"
        # fall back on the fully trained model if we're not saving evaluation
        # checkpoint models
        if not eval_model_path.exists():
            eval_model_path = self.trainer.curr_eval_dir / "fully_trained_model"
        # ot = over time
        portfolio_value_ot, _ = test_once(
            self.eval_env, eval_model_path, self.model_name
        )
        sharpe = self.calculate_sharpe(portfolio_value_ot)
        mlflow.log_metric("sharpe_opt", sharpe)
        if self.trainer.use_mlflow:
            mlflow.end_run()
        return sharpe

    def calculate_sharpe(self, df: pd.DataFrame):
        df["daily_return"] = df["account_value"].pct_change(1)
        if df["daily_return"].std() != 0:
            sharpe = (252**0.5) * df["daily_return"].mean() / df["daily_return"].std()
            return sharpe
        else:
            return 0


class SharpeOptimiser(Trainer):
    def __init__(
        self,
        *args,
        study_name: str = "hyperparam_optimisation",
        **kwargs,
    ):
        self.study_name = study_name
        super().__init__(*args, **kwargs)

    def _setup_train_dirs(self):
        super()._setup_train_dirs()
        self.study_dir = self.root_save_dir / "study"
        self.study_dir.mkdir(exist_ok=True, parents=True)

    def _load_or_create_study(self) -> optuna.study.Study:
        potential_study_save_path = self.study_dir / "study.pkl"
        if potential_study_save_path.exists():
            print("loading study from checkpoint...")
            study = joblib.load(potential_study_save_path)
            print("loaded study from checkpoint!")
            return study
        print("creating new study...")
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",  # maximise sharpe ratio
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(),
        )
        return study

    def optimise(
        self,
        total_timesteps: int = 10000,
        n_trials: int = 30,
    ) -> dict[str, Any]:
        """Optimise hyperparameters for a model using optuna."""
        study = self._load_or_create_study()
        checkpoint_save_callback = TrialCheckpointCallback(self.study_dir)
        callbacks = [checkpoint_save_callback]
        objective_fn = SharpeObjective(
            train_env=self.train_env,
            eval_env=self.eval_env,
            trainer=self,
            model_name=self.model_name,
            total_timesteps=total_timesteps,
        )
        print(f"beginning optimisation of {self.model_name}_{self.ticker_id}...")
        study.optimize(
            objective_fn,
            n_trials=n_trials,
            catch=(ValueError,),
            gc_after_trial=True,
            callbacks=callbacks,
        )
        print("done! dumping...")
        joblib.dump(study, self.study_dir / "finished_study.pkl")
        print("finished optimisation!")
        return {
            "study": study,
            "train_env": self.train_env,
            "test_env": self.eval_env,
        }


def sample_ppo_params(trial: optuna.Trial):
    """
    {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 64,
    }
    """
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-6, 0.1)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

    return {
        "ent_coef": ent_coef,
        "n_steps": n_steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }


def sample_timesteps(trial: optuna.Trial):
    return trial.suggest_categorical("total_timesteps", [10000, 20000, 50000, 100000])


def sample_hyperparams(trial: optuna.Trial, model_name: str):
    if model_name == "ppo":
        return sample_ppo_params(trial)
    else:
        raise ValueError("Invalid model_name")


class TrialCheckpointCallback:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        print(f"updating study checkpoint {trial.number}...")
        study_save_path = self.save_dir / "study.pkl"
        trial_num_save_path = self.save_dir / "trial_chkpt.txt"
        # save study checkpoint pickle
        joblib.dump(study, study_save_path)
        # save trial number
        with open(trial_num_save_path, "w") as f:
            f.write(str(trial.number))


def get_base_dir(model_name: str, ticker_id: str) -> pathlib.Path:
    """Retrieve the base directory for a model and ticker combination.

    Within this directory there may be logs, saved study optimisation checkpoints,
    models and study files."""
    return ROOT_DIR / RESULTS_DIR / model_name / ticker_id
