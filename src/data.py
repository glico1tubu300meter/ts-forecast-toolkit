"""
データ取得・窓データ生成ユーティリティ。

USD/JPY等の金融時系列に対する「過去lookback日分の対数リターン -> N期先の
累積対数リターンを直接予測する」というタスク形式(Direct forecasting)を
共通化している。
"""
from pathlib import Path

import numpy as np
import pandas as pd


def fetch_price_series(ticker: str, start: str, end: str) -> pd.Series:
    """yfinanceで日次終値を取得し、日付インデックスのSeriesとして返す"""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.rename("close")
    close.index.name = "date"
    return close.astype(float)


DEFAULT_ASSETS = {
    "usdjpy": "JPY=X",
    "eurusd": "EURUSD=X",
    "sp500": "^GSPC",
    "nikkei225": "^N225",
    "btcusd": "BTC-USD",
}


def make_direct_windows(prices: pd.Series, horizon: int, lookback: int):
    """過去lookback日の対数リターン -> horizon日先までの累積対数リターン、の窓データを作る。

    起点日tについて:
      入力  = log_ret[t-lookback : t]
      目標  = log(price[t+horizon-1] / price[t-1])  (起点日の前日を基準にした累積対数リターン)

    horizon=1のときは「翌日の対数リターンをそのまま予測する」タスクと等価。

    Returns
    -------
    X : np.ndarray, shape (n_samples, lookback)
    y : np.ndarray, shape (n_samples,)
    origin_dates : pd.DatetimeIndex
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    ret_vals = log_ret.values.astype(np.float32)
    ret_dates = log_ret.index
    price_vals = prices.values.astype(np.float32)
    price_dates = prices.index

    X, y, origin_dates = [], [], []
    for i in range(lookback, len(ret_vals)):
        origin_date = ret_dates[i]
        pos = price_dates.get_loc(origin_date)
        if pos - 1 < 0 or pos + horizon - 1 >= len(price_vals):
            continue
        X.append(ret_vals[i - lookback:i])
        target = np.log(price_vals[pos + horizon - 1] / price_vals[pos - 1])
        y.append(target)
        origin_dates.append(origin_date)
    return np.array(X), np.array(y, dtype=np.float32), pd.DatetimeIndex(origin_dates)


def train_test_split_by_tail(origin_dates: pd.DatetimeIndex, n_test: int):
    """末尾n_test件をテスト期間とするマスクを返す(walk-forward評価の簡易版)"""
    test_dates = origin_dates[-n_test:]
    test_mask = origin_dates.isin(test_dates)
    train_mask = ~test_mask
    return train_mask, test_mask


def price_targets_for_test(prices: pd.Series, origin_dates: pd.DatetimeIndex, horizon: int):
    """テスト期間の各起点日について、基準価格(起点日前日)と実際の目標価格(horizon日後)を返す"""
    price_at_origin_minus1 = np.array([
        prices.iloc[prices.index.get_loc(d) - 1] for d in origin_dates
    ], dtype=np.float64)
    actual_price = np.array([
        prices.iloc[prices.index.get_loc(d) + horizon - 1] for d in origin_dates
    ], dtype=np.float64)
    return price_at_origin_minus1, actual_price


def load_gdelt_tone(csv_path: Path) -> pd.Series:
    """15_fetch_gdelt_tone.py(newsディレクトリ参照)で取得した日次トーンCSVを読み込む"""
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
    return df["tone"]
