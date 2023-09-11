import pathlib
from typing import Any

import joblib
import optuna
import pandas as pd
from stable_baselines3.common.logger import configure

from paths import ROOT_DIR
from stok.finrl_slim.agent import MODELS, DRLAgent
from stok.finrl_slim.analysis import get_hold_strategy_value_ot
from stok.finrl_slim.config import INDICATORS, RESULTS_DIR
from stok.finrl_slim.env import StockTradingEnv
from stok.finrl_slim.preprocessing.preprocessors import Preprocessor


def evaluate_models(
    test_data: pd.DataFrame,
    model_name: str = "ppo",
    indicators: list[str] = INDICATORS,
    initial_amount: float = 10000,
):
    """Evaluate the performance of a model on each ticker in a test dataset.
    For this function to work, the test_data must have a column called "tic" and
    there must exist a model for each ticker and model_name in the RESULTS_DIR."""
    result_dfs = []
    action_dfs = []
    for ticker in test_data.tic.unique().tolist():
        test_data_subset = test_data[test_data.tic == ticker]
        portfolio_ot, actions_ot = test_once(
            test_data_subset,
            model_name=model_name,
            indicators=indicators,
        )
        portfolio_ot.set_index("date", inplace=True)
        portfolio_ot.index = pd.to_datetime(portfolio_ot.index, format="%Y-%m-%d")
        portfolio_ot.index = portfolio_ot.index.date
        hold_strategy_result = get_hold_strategy_value_ot(
            ticker, test_data_subset, initial_amount
        )
        result_df = pd.concat([portfolio_ot, hold_strategy_result], axis=1)
        result_df["tic"] = ticker
        result_df = result_df.reset_index()
        result_dfs.append(result_df)
        action_dfs.append(actions_ot)
    return pd.concat(result_dfs), action_dfs


def train_once(
    train_env: StockTradingEnv,
    trial: optuna.Trial = None,
    model_name: str = "ppo",
    hyperparameters: dict[str, Any] | None = None,
    total_timesteps: int = 10000,
) -> None:
    """Train a model on a training dataset.

    Args:
        train_data (pd.DataFrame): The train dataset. Should contain close prices and
            technical indicators for a single ticker over time.
        trial (optuna.Trial, optional): Used in the case of hyperparameter
            optimisation to track the trial number. Defaults to None.
        model_name (str, optional): The name of the model architecture to use.
            Defaults to "ppo".
        indicators (list[str], optional): The technical indicators to use.
            Defaults to INDICATORS.
        hyperparameters (dict[str, Any], optional): The hyperparameters to pass to the
            underlying model. Defaults to None.
    """
    ticker_id = "_".join(train_env.df.tic.unique())
    model_save_dir = get_base_dir(model_name, ticker_id)
    model_save_dir.mkdir(exist_ok=True, parents=True)
    model_save_name = (
        f"{model_name}_{ticker_id}_{trial.number}_model"
        if trial
        else f"{ticker_id}_{model_name}_model"
    )
    print(f"training {model_save_name} once...")
    agent = DRLAgent(env=train_env)
    model = agent.get_model(model_name=model_name, model_kwargs=hyperparameters)
    new_logger = configure(str(model_save_dir / "log"), ["stdout", "csv"])
    model.set_logger(new_logger)
    trained_ppo = agent.train_model(
        model=model, tb_log_name=model_name, total_timesteps=total_timesteps
    )
    print(f"done! saving {model_save_name} model...")
    trained_ppo.save(model_save_dir / model_save_name)
    print(f"finished training {model_save_name} model!")


def test_once(
    test_env: StockTradingEnv,
    trial: optuna.Trial = None,
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
    ticker_id = "_".join(test_env.df.tic.unique())
    model_save_dir = get_base_dir(model_name, ticker_id)
    model_save_name = (
        f"{model_name}_{ticker_id}_{trial.number}_model"
        if trial
        else f"{ticker_id}_{model_name}_model"
    )
    print(f"loading trained {model_save_name} model...")
    model = MODELS.get(model_name, None)
    if model is None:
        raise NotImplementedError("model_name is not supported")
    model = model.load(model_save_dir / model_save_name)
    print(f"loaded {model_save_name} model!")
    print("making predictions on test data...")
    portfolio_value_ot_df, actions_ot_df = DRLAgent.DRL_prediction(
        model=model, environment=test_env
    )
    print(f"finish making predictions for {model_save_name}!")
    return portfolio_value_ot_df, actions_ot_df


def calculate_sharpe(df):
    df["daily_return"] = df["account_value"].pct_change(1)
    if df["daily_return"].std() != 0:
        sharpe = (252**0.5) * df["daily_return"].mean() / df["daily_return"].std()
        return sharpe
    else:
        return 0


class Objective:
    """Defines an objective function for hyperparameter optimisation."""

    def __init__(
        self,
        train_env: StockTradingEnv,
        test_env: StockTradingEnv,
        model_name: str = "ppo",
        total_timesteps: int = 10000,
    ):
        self.train_env = train_env
        self.test_env = test_env
        self.model_name = model_name
        self.total_timesteps = total_timesteps

    def __call__(self, trial: optuna.Trial) -> Any:
        params = sample_params(trial, self.model_name)
        train_once(
            self.train_env,
            trial,
            self.model_name,
            hyperparameters=params,
            total_timesteps=self.total_timesteps,
        )
        # ot = over time
        portfolio_value_ot, _ = test_once(self.test_env, trial, self.model_name)
        sharpe = calculate_sharpe(portfolio_value_ot)
        return sharpe


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


def sample_params(trial: optuna.Trial, model_name: str):
    if model_name == "ppo":
        return sample_ppo_params(trial)
    else:
        raise ValueError("Invalid model_name")


class TrialCheckpointCallback:
    def __init__(self, ticker_id: str, model_name: str):
        self.ticker_id = ticker_id
        self.model_name = model_name
        self.chkpt_dir = get_optimisation_checkpoint_dir(model_name, ticker_id)
        self.chkpt_dir.mkdir(exist_ok=True, parents=True)

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        print(f"updating study checkpoint {trial.number}...")
        study_save_path = (
            self.chkpt_dir / f"{self.model_name}_{self.ticker_id}_chkpt.pkl"
        )
        trial_num_save_path = (
            self.chkpt_dir / f"{self.model_name}_{self.ticker_id}_trial_num.txt"
        )
        # save study pickle
        joblib.dump(study, study_save_path)
        # save trial number
        with open(trial_num_save_path, "w") as f:
            f.write(str(trial.number))


def get_base_dir(model_name: str, ticker_id: str) -> pathlib.Path:
    """Retrieve the base directory for a model and ticker combination.

    Within this directory there may be logs, saved study optimisation checkpoints,
    models and study files."""
    return ROOT_DIR / RESULTS_DIR / model_name / ticker_id


def get_optimisation_checkpoint_dir(model_name: str, ticker_id: str) -> pathlib.Path:
    """Retrieve the directory where optimisation checkpoints are saved."""
    return get_base_dir(model_name, ticker_id) / "optimisation_checkpoints"


def get_study(study_name: str, ticker_id: str, model_name: str) -> optuna.study.Study:
    potential_study_chkpt = (
        get_optimisation_checkpoint_dir(model_name, ticker_id)
        / f"{model_name}_{ticker_id}_chkpt.pkl"
    )
    if potential_study_chkpt.exists():
        print("loading study from checkpoint...")
        study = joblib.load(potential_study_chkpt)
        print("loaded study from checkpoint!")
        return study
    print("creating new study...")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # maximise sharpe ratio
        sampler=sampler,
        pruner=optuna.pruners.HyperbandPruner(),
    )
    return study


def optimise(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    indicators: list[str],
    model_name: str,
    study_name: str = "hyperparam_optimisation",
    n_trials: int = 30,
    train_total_timesteps: int = 10000,
    buy_cost_pct: list[float] = [0.001],
    sell_cost_pct: list[float] = [0.001],
    num_stock_shares: list[int] = [0],
    hmax: int = 100,
):
    """Optimise hyperparameters for a model using optuna."""
    ticker_id = "_".join(train_data.tic.unique())
    stock_dims = len(train_data.tic.unique())
    assert (
        len(buy_cost_pct) == len(sell_cost_pct) == len(num_stock_shares) == stock_dims
    ), f"buy_cost_pct, sell_cost_pct and num_stock_shares must have the same length as the number of stocks ({stock_dims})"  # noqa
    optimisation_save_dir = ROOT_DIR / RESULTS_DIR / model_name / ticker_id
    study = get_study(study_name, ticker_id, model_name)
    checkpoint_save_callback = TrialCheckpointCallback(ticker_id, model_name)
    train_env = StockTradingEnv(
        df=train_data,
        stock_dims=stock_dims,
        hmax=hmax,
        initial_amount=10000,
        num_stock_shares=num_stock_shares,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        tech_indicator_list=indicators,
    )
    test_env = StockTradingEnv(
        df=test_data,
        stock_dims=stock_dims,
        hmax=hmax,
        initial_amount=10000,
        num_stock_shares=num_stock_shares,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        turbulence_threshold=70,
        risk_indicator_col="vix",
        tech_indicator_list=indicators,
    )
    objective_fn = Objective(
        train_env,
        test_env,
        model_name=model_name,
        total_timesteps=train_total_timesteps,
    )
    print(f"beginning optimisation of {model_name}_{ticker_id}...")
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        catch=(ValueError,),
        gc_after_trial=True,
        callbacks=[checkpoint_save_callback],
    )
    print("done! dumping...")
    joblib.dump(study, optimisation_save_dir / f"{model_name}_{ticker_id}_study.pkl")
    print("finished optimisation!")
    return {
        "study": study,
        "train": {
            "env": train_env,
            "data": train_data,
        },
        "test": {
            "env": test_env,
            "data": test_data,
        },
    }


def main(
    train_start_date: str,
    train_end_date: str,
    test_start_date: str,
    test_end_date: str,
    ticker_list: list[str],
    total_timesteps: int = 10000,
    buy_cost_pct: list[float] = [0.001],
    sell_cost_pct: list[float] = [0.001],
    num_stock_shares: list[int] = [0],
    hmax: int = 100,
) -> optuna.Study:
    """The main training and optimisation loop.
    Calling this function will result in a study file
    being created for the ticker list. From this study
    file, the optimal hyperparameters can be loaded and
    used to load an optimal model. The optimisation currently
    attempts to maximise the sharpe ratio.
    """
    p = Preprocessor(
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        ticker_list=ticker_list,
    )
    train_data, test_data = p.get_train_test(
        use_technical_indicator=True,
        use_turbulence=True,
        use_vix=True,
        tech_indicator_list=INDICATORS,
    )
    result_object = optimise(
        train_data=train_data,
        test_data=test_data,
        indicators=INDICATORS,
        model_name="ppo",
        study_name="hyperparam_optimisation",
        n_trials=30,
        train_total_timesteps=total_timesteps,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        num_stock_shares=num_stock_shares,
        hmax=hmax,
    )
    return result_object
