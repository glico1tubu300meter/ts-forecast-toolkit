"""
GDELTニューストーンを特徴量に加えたDLinear vs 価格のみのDLinearを比較する。

事前に news/fetch_gdelt_tone.py で日次トーンCSVを取得しておくこと。
GDELTは2017年以降しかカバーしないため、--startは2017-01-01以降を指定する。

使用例:
    python run_with_news.py --ticker JPY=X --news-csv out/gdelt_tone_daily.csv \\
        --start 2017-01-01 --horizon 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import fetch_price_series, load_gdelt_tone
from src.models import DLinear, DLinearWithNews
from src.train import naive_metrics, price_metrics


def make_windows_with_news(prices, tone, horizon, lookback):
    log_ret = np.log(prices / prices.shift(1)).dropna()
    ret_vals = log_ret.values.astype(np.float32)
    ret_dates = log_ret.index
    price_vals = prices.values.astype(np.float32)
    price_dates = prices.index
    tone_vals = tone.reindex(price_dates).ffill().bfill().values.astype(np.float32)

    X_ret, X_tone, y, origin_dates = [], [], [], []
    for i in range(lookback, len(ret_vals)):
        origin_date = ret_dates[i]
        pos = price_dates.get_loc(origin_date)
        if pos - 1 < 0 or pos + horizon - 1 >= len(price_vals) or pos - lookback < 0:
            continue
        X_ret.append(ret_vals[i - lookback:i])
        X_tone.append(tone_vals[pos - lookback:pos])
        target = np.log(price_vals[pos + horizon - 1] / price_vals[pos - 1])
        y.append(target)
        origin_dates.append(origin_date)
    return (np.array(X_ret), np.array(X_tone), np.array(y, dtype=np.float32), pd.DatetimeIndex(origin_dates))


def train_eval(model, X_ret_train, X_tone_train, y_train, X_ret_test, X_tone_test, use_tone, device, n_epochs=60, batch_size=64):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    n_train = X_ret_train.shape[0]
    t0 = time.perf_counter()
    for _ in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb_tone = X_tone_train[idx] if use_tone else None
            optimizer.zero_grad()
            loss = loss_fn(model(X_ret_train[idx], xb_tone), y_train[idx])
            loss.backward()
            optimizer.step()
    train_time = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        pred = model(X_ret_test, X_tone_test if use_tone else None).cpu().numpy()
    return pred, train_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="JPY=X")
    parser.add_argument("--news-csv", required=True)
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--n-test", type=int, default=252)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    prices = fetch_price_series(args.ticker, args.start, end)
    tone = load_gdelt_tone(Path(args.news_csv))

    X_ret_all, X_tone_all, y_all, origin_all = make_windows_with_news(prices, tone, args.horizon, args.lookback)
    n_test = min(args.n_test, max(50, len(origin_all) // 5))
    test_dates = origin_all[-n_test:]
    train_mask = ~origin_all.isin(test_dates)
    test_mask = origin_all.isin(test_dates)
    print(f"n_train={train_mask.sum()}  n_test={test_mask.sum()}")

    mu_y, sigma_y = y_all[train_mask].mean(), y_all[train_mask].std()
    mu_t, sigma_t = X_tone_all[train_mask].mean(), X_tone_all[train_mask].std()

    X_ret_train = torch.tensor(X_ret_all[train_mask]).to(device)
    X_tone_train = torch.tensor((X_tone_all[train_mask] - mu_t) / sigma_t).to(device)
    y_train = torch.tensor((y_all[train_mask] - mu_y) / sigma_y).to(device)
    X_ret_test = torch.tensor(X_ret_all[test_mask]).to(device)
    X_tone_test = torch.tensor((X_tone_all[test_mask] - mu_t) / sigma_t).to(device)

    price_base = np.array([prices.iloc[prices.index.get_loc(d) - 1] for d in origin_all[test_mask]])
    actual_price = np.array([prices.iloc[prices.index.get_loc(d) + args.horizon - 1] for d in origin_all[test_mask]])

    results = {"naive": naive_metrics(price_base, actual_price)}
    print(f"naive         RMSE={results['naive']['rmse']:.4f}  MAPE={results['naive']['mape_pct']:.3f}%")

    for name, model, use_tone in [
        ("dlinear", DLinear(args.lookback), False),
        ("dlinear_news", DLinearWithNews(args.lookback), True),
    ]:
        pred_norm, train_time = train_eval(model, X_ret_train, X_tone_train, y_train,
                                            X_ret_test, X_tone_test, use_tone, device)
        pred_price = price_base * np.exp(pred_norm * sigma_y + mu_y)
        metrics = price_metrics(actual_price, pred_price)
        results[name] = {**metrics, "train_sec": train_time}
        print(f"{name:14s} RMSE={metrics['rmse']:.4f}  MAPE={metrics['mape_pct']:.3f}%  train={train_time:.2f}s")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
