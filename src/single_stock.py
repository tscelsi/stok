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
    train_data: pd.DataFrame,
    trial: optuna.Trial = None,
    model_name: str = "ppo",
    indicators: list[str] = INDICATORS,
    hyperparameters: dict[str, Any] | None = None,
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
    ticker_symbol = train_data.tic.unique().tolist()[0]
    model_save_dir = ROOT_DIR / RESULTS_DIR / model_name / ticker_symbol
    model_save_dir.mkdir(exist_ok=True, parents=True)
    model_save_name = (
        f"{model_name}_{ticker_symbol}_{trial.number}_model"
        if trial
        else f"{ticker_symbol}_{model_name}_model"
    )
    print(f"training {model_save_name} once...")
    env_train = StockTradingEnv(
        df=train_data,
        stock_dims=1,
        hmax=100,
        initial_amount=10000,
        num_stock_shares=[0],
        buy_cost_pct=[0.001],
        sell_cost_pct=[0.001],
        tech_indicator_list=indicators,
    )
    agent = DRLAgent(env=env_train)
    model = agent.get_model(model_name=model_name, model_kwargs=hyperparameters)
    new_logger = configure(str(model_save_dir / "log"), ["stdout", "csv"])
    model.set_logger(new_logger)
    trained_ppo = agent.train_model(
        model=model, tb_log_name=model_name, total_timesteps=10000
    )
    print(f"done! saving {model_save_name} model...")
    trained_ppo.save(model_save_dir / f"{model_save_name}_model")
    print(f"finished training {model_save_name} model!")


def test_once(
    test_data: pd.DataFrame,
    trial: optuna.Trial = None,
    model_name: str = "ppo",
    indicators: list[str] = INDICATORS,
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
    env_test = StockTradingEnv(
        df=test_data,
        stock_dims=1,
        hmax=100,
        initial_amount=10000,
        num_stock_shares=[0],
        buy_cost_pct=[0.001],
        sell_cost_pct=[0.001],
        turbulence_threshold=70,
        risk_indicator_col="vix",
        tech_indicator_list=indicators,
    )
    ticker_symbol = test_data.tic.unique().tolist()[0]
    model_save_dir = ROOT_DIR / RESULTS_DIR / model_name / ticker_symbol
    model_save_name = (
        f"{model_name}_{ticker_symbol}_{trial.number}_model"
        if trial
        else f"{ticker_symbol}_{model_name}_model"
    )
    print(f"loading trained {model_save_name} model...")
    model = MODELS.get(model_name, None)
    if model is None:
        raise NotImplementedError("model_name is not supported")
    model = model.load(model_save_dir / f"{model_save_name}_model", env=env_test)
    print(f"loaded {model_save_name} model!")
    print("making predictions on test data...")
    portfolio_value_ot_df, actions_ot_df = DRLAgent.DRL_prediction(
        model=model, environment=env_test
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
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        indicators: list[str] = INDICATORS,
        model_name: str = "ppo",
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.indicators = indicators
        self.model_name = model_name

    def __call__(self, trial: optuna.Trial) -> Any:
        params = sample_params(trial, self.model_name)
        train_once(
            self.train_data,
            trial,
            self.model_name,
            self.indicators,
            hyperparameters=params,
        )
        # ot = over time
        portfolio_value_ot, _ = test_once(
            self.test_data, trial, self.model_name, self.indicators
        )
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


def optimise(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    indicators: list[str],
    model_name: str,
    study_name: str = "hyperparam_optimisation",
    n_trials: int = 30,
):
    """Optimise hyperparameters for a model using optuna.

    This function assumes that the train and test data only contains
    the information for one ticker."""
    ticker_symbol = train_data.tic.unique().tolist()[0]
    opt_save_dir = ROOT_DIR / RESULTS_DIR / model_name / ticker_symbol
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # maximise sharpe ratio
        sampler=sampler,
        pruner=optuna.pruners.HyperbandPruner(),
    )
    objective_fn = Objective(train_data, test_data, indicators, model_name)
    print(f"beginning optimisation of {model_name}_{ticker_symbol}...")
    study.optimize(objective_fn, n_trials=n_trials, catch=(ValueError,))
    print("done! dumping...")
    joblib.dump(study, opt_save_dir / f"{model_name}_{ticker_symbol}_study.pkl")
    print("finished optimisation!")


def main() -> None:
    """The main training and optimisation loop.
    Calling this function will result in a study file
    being created for the ticker GOOG. From this study
    file, the optimal hyperparameters can be loaded and
    used to load an optimal model for GOOG.
    """
    p = Preprocessor(
        train_start_date="2009-01-01",
        train_end_date="2019-01-01",
        test_start_date="2019-01-01",
        test_end_date="2021-01-01",
        ticker_list=["GOOG"],
    )
    train_data, test_data = p.get_train_test(
        use_technical_indicator=True,
        use_turbulence=True,
        use_vix=True,
        tech_indicator_list=INDICATORS,
    )
    optimise(
        train_data=train_data,
        test_data=test_data,
        indicators=INDICATORS,
        model_name="ppo",
        study_name="hyperparam_optimisation",
        n_trials=30,
    )
