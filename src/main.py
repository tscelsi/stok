import pandas as pd

from paths import ROOT_DIR
from stok.emulator import Emulator, StateModel
from stok.portfolio import Portfolio
from stok.strategies.moving_average import MovingAverageStrategy

if __name__ == "__main__":
    initial_state = StateModel(portfolio=Portfolio(), available_funds=1000)
    days = pd.read_csv(ROOT_DIR / "goog.csv", index_col="Date", parse_dates=True)
    e = Emulator(initial_state=initial_state, strategy=MovingAverageStrategy())
    e.run(days)
