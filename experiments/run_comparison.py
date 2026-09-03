"""
汎用の時系列予測手法比較ランナー。

指定した資産(yfinanceティッカー)・ホライズン・手法群で
Naive + DLinear/TSMixer/PatchTST/TimeMixer を学習・評価する。

使用例:
    python run_comparison.py --ticker JPY=X --horizon 20 --lookback 30
    python run_comparison.py --ticker BTC-USD --horizon 1 --models dlinear,patchtst

元になった実験: I:\\マイドライブ\\Claude2\\output\\2609\\02-014-DLinear-TSMixer-PatchTST比較
(08_multihorizon.py, 11_multi_asset_horizon20.py 等を汎用化したもの)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import fetch_price_series, make_direct_windows, train_test_split_by_tail, price_targets_for_test
from src.models import MODEL_REGISTRY
from src.train import train_and_predict, naive_metrics, price_metrics


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="JPY=X", help="yfinanceティッカー(例: JPY=X, BTC-USD, ^GSPC)")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--horizon", type=int, default=20, help="予測ホライズン(営業日)")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--n-test", type=int, default=252)
    parser.add_argument("--n-epochs", type=int, default=60)
    parser.add_argument("--models", default="dlinear,tsmixer,patchtst,timemixer",
                         help="カンマ区切り。MODEL_REGISTRYのキーから選択")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="結果JSONの出力先(省略時は標準出力のみ)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    import pandas as pd
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    prices = fetch_price_series(args.ticker, args.start, end)
    print(f"{args.ticker}: {len(prices)}日  {prices.index[0].date()} - {prices.index[-1].date()}")

    X_all, y_all, origin_all = make_direct_windows(prices, args.horizon, args.lookback)
    train_mask, test_mask = train_test_split_by_tail(origin_all, args.n_test)
    print(f"n_train={train_mask.sum()}  n_test={test_mask.sum()}")

    mu, sigma = y_all[train_mask].mean(), y_all[train_mask].std()
    X_train = torch.tensor(X_all[train_mask]).to(device)
    y_train = torch.tensor((y_all[train_mask] - mu) / sigma).to(device)
    X_test = torch.tensor(X_all[test_mask]).to(device)

    price_base, actual_price = price_targets_for_test(prices, origin_all[test_mask], args.horizon)

    results = {"naive": naive_metrics(price_base, actual_price)}
    print(f"naive        RMSE={results['naive']['rmse']:.4f}  MAPE={results['naive']['mape_pct']:.3f}%")

    for name in args.models.split(","):
        name = name.strip()
        if name not in MODEL_REGISTRY:
            print(f"skip unknown model: {name}")
            continue
        model = MODEL_REGISTRY[name](args.lookback)
        pred_norm, timing = train_and_predict(model, X_train, y_train, X_test, device, n_epochs=args.n_epochs)
        pred_ret = pred_norm * sigma + mu
        pred_price = price_base * np.exp(pred_ret)
        metrics = price_metrics(actual_price, pred_price)
        results[name] = {**metrics, **timing}
        print(f"{name:10s} RMSE={metrics['rmse']:.4f}  MAPE={metrics['mape_pct']:.3f}%  "
              f"train={timing['train_sec']:.2f}s  params={timing['n_params']}")

    results["_meta"] = {
        "ticker": args.ticker, "horizon": args.horizon, "lookback": args.lookback,
        "n_test": int(test_mask.sum()), "device": str(device),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
