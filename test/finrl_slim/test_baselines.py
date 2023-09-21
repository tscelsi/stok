import pandas as pd
import pytest

from src.paths import TEST_DIR


@pytest.fixture
def load_goog():
    path = TEST_DIR / "finrl_slim" / "fixtures" / "goog.csv"
    df = pd.read_csv(path)
    yield df


@pytest.fixture
def load_multi_tic():
    path = TEST_DIR / "finrl_slim" / "fixtures" / "goog_aapl.csv"
    df = pd.read_csv(path)
    yield df


@pytest.fixture
def dummy_baseline():
    path = TEST_DIR / "finrl_slim" / "fixtures" / "dummy_baseline.csv"
    df = pd.read_csv(path)
    yield df


def test_baseline_dji(monkeypatch, dummy_baseline):
    import datetime

    from src.stok.finrl_slim.baselines import baseline_dji

    monkeypatch.setattr(
        "src.stok.finrl_slim.analysis._get_baseline",
        lambda *args, **kwargs: dummy_baseline,
    )
    # because patched don't worry about dates
    dji = baseline_dji("", "")
    assert isinstance(dji, pd.DataFrame)
    assert dji.iloc[0].close == 1000000
    assert isinstance(dji.index[0], datetime.date)


def test_baseline_mvo(monkeypatch, load_multi_tic):
    from src.stok.finrl_slim.baselines import baseline_mvo
    from src.stok.finrl_slim.preprocessing.preprocessors import Preprocessor

    # patched so dont worry about dates
    p = Preprocessor(
        ticker_list=["GOOG", "AAPL"],
        train_start_date="",
        train_end_date="",
        test_start_date="",
        test_end_date="",
    )

    def mock_get_train_test(*args, **kwargs):
        return load_multi_tic.iloc[:152], load_multi_tic.iloc[152:]

    monkeypatch.setattr(
        "src.stok.finrl_slim.preprocessing.preprocessors.Preprocessor.get_train_test",
        mock_get_train_test,
    )
    # assert runs
    baseline_mvo(p)
