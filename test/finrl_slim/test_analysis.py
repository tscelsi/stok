import pandas as pd
import pytest

from paths import TEST_DIR


@pytest.fixture
def load_goog():
    path = TEST_DIR / "finrl_slim" / "fixtures" / "goog.csv"
    df = pd.read_csv(path)
    yield df


def test_get_hold_strategy_value_ot(load_goog):
    from stok.finrl_slim.analysis import get_hold_strategy_value_ot
    res = get_hold_strategy_value_ot("GOOG", load_goog, 1000)
    assert isinstance(res, pd.Series)


def test_evaluate_models(load_goog):
    from single_stock import evaluate_models
    result_dfs = evaluate_models(load_goog, "ppo", [], turbulence_threshold=None)
    print("HI")
