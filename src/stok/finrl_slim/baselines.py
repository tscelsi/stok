from copy import deepcopy

import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier

from .preprocessing.preprocessors import Preprocessor
from .preprocessing.yahoodownloader import YahooDownloader


def get_daily_return(df: pd.DataFrame, value_col_name: str = "account_value"):
    """Create a daily return series using the account's portfolio value.

    Args:
        df (pd.DataFrame): The dataframe containing the account's portfolio value
            over time.
        value_col_name (str, optional): The column containins the portfolio values.
            Defaults to "account_value".

    Returns:
        pd.Series: The daily return series in percentage change.
    """
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)


def _get_baseline(ticker: str, start: str, end: str) -> pd.DataFrame:
    return YahooDownloader(
        start_date=start, end_date=end, ticker_list=[ticker]
    ).fetch_data()


def get_baseline(ticker: str, test_data: pd.DataFrame) -> pd.DataFrame:
    baseline_df = test_data[test_data.tic == ticker]
    return baseline_df.copy()


def baseline_dji(start_date: str, end_date: str) -> pd.DataFrame:
    df_dji = _get_baseline("^DJI", start_date, end_date)
    df_dji = df_dji[["date", "close"]]
    fst_day = df_dji["close"][0]
    dji = pd.merge(
        df_dji["date"],
        df_dji["close"].div(fst_day).mul(1000000),
        how="outer",
        left_index=True,
        right_index=True,
    ).set_index("date")
    dji.index = pd.to_datetime(dji.index, format="%Y-%m-%d")
    dji.index = dji.index.date
    dji = dji.rename(columns={"close": "^DJI"})
    return dji


def baseline_mvo(p: Preprocessor, initial_amount: int = 1000000) -> pd.DataFrame:
    """Retrieves the Mean Variance Optimisation (MVO) result.

    Returns:
        pd.DataFrame: The MVO result. With a column named "Mean Var".
    """

    def process_df_for_mvo(df):
        stock_dimension = len(df.tic.unique())
        df = df.sort_values(["date", "tic"], ignore_index=True)[
            ["date", "tic", "close"]
        ]
        fst = df
        fst = fst.iloc[0:stock_dimension, :]
        tic = fst["tic"].tolist()

        mvo = pd.DataFrame()

        for k in range(len(tic)):
            mvo[tic[k]] = 0

        for i in range(df.shape[0] // stock_dimension):
            n = df
            n = n.iloc[i * stock_dimension : (i + 1) * stock_dimension, :]
            date = n["date"][i * stock_dimension]
            mvo.loc[date] = n["close"].tolist()

        return mvo

    def StockReturnsComputing(StockPrice, Rows, Columns):
        StockReturn = np.zeros([Rows - 1, Columns])
        for j in range(Columns):  # j: Assets
            for i in range(Rows - 1):  # i: Daily Prices
                StockReturn[i, j] = (
                    (StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]
                ) * 100

        return StockReturn

    train, trade = p.get_train_test(
        use_technical_indicator=False,
        use_vix=False,
        use_turbulence=False,
    )
    stock_dimension = len(train.tic.unique())
    StockData = process_df_for_mvo(train)
    TradeData = process_df_for_mvo(trade)
    arStockPrices = np.asarray(StockData)
    [rows, cols] = arStockPrices.shape
    arReturns = StockReturnsComputing(arStockPrices, rows, cols)

    # compute mean returns and variance covariance matrix of returns
    meanReturns = np.mean(arReturns, axis=0)
    covReturns = np.cov(arReturns, rowvar=False)

    ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
    ef_mean.max_sharpe()
    cleaned_weights_mean = ef_mean.clean_weights()
    mvo_weights = np.array(
        [initial_amount * cleaned_weights_mean[i] for i in range(stock_dimension)]
    )
    LastPrice = np.array([1 / p for p in StockData.tail(1).to_numpy()[0]])
    Initial_Portfolio = np.multiply(mvo_weights, LastPrice)
    Portfolio_Assets = TradeData @ Initial_Portfolio
    MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])
    MVO_result.index = pd.to_datetime(MVO_result.index, format="%Y-%m-%d")
    return MVO_result
