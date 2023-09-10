from copy import deepcopy

import pandas as pd

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


def get_dow_jones_index_baseline(start: str, end: str) -> pd.DataFrame:
    """Retrieve the Dow Jones Industrial Average index as a baseline.

    Args:
        start (str): When the baseline should start.
        end (str): When the baseline should end.

    Returns:
        _type_: The baseline dataframe
    """
    TICKER = "^DJI"
    return _get_baseline(TICKER, start, end)


def _get_baseline(ticker: str, start: str, end: str) -> pd.DataFrame:
    return YahooDownloader(
        start_date=start, end_date=end, ticker_list=[ticker]
    ).fetch_data()


def get_baseline(ticker: str, test_data: pd.DataFrame) -> pd.DataFrame:
    baseline_df = test_data[test_data.tic == ticker]
    return baseline_df


def get_hold_strategy_value_ot(
    ticker: str, test_data: pd.DataFrame, initial_balance: float
) -> pd.Series:
    values = []
    baseline_df = get_baseline(ticker, test_data)
    baseline_df = get_daily_return(baseline_df, "close").fillna(0)
    baseline_df.index = baseline_df.index.date
    curr_value = initial_balance
    for pct_change in baseline_df:
        curr_value = curr_value * (1 + pct_change)
        values.append(curr_value)
    return pd.Series(values, index=baseline_df.index, name="holding_value")
