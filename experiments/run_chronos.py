"""
Chronos(Amazon製の時系列基盤モデル)によるゼロショット予測の評価。
学習は行わず、直近の価格系列をコンテキストとして与えるだけで予測する。

使用例:
    python run_chronos.py --ticker JPY=X --horizon 20

事前に `pip install chronos-forecasting` が必要。
HF_HOMEを設定し、大容量なモデルキャッシュはH:\\HeavyData等ローカルNTFSに置くこと
(Googleドライブ仮想ドライブ上には置かない)。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import fetch_price_series, make_direct_windows, train_test_split_by_tail, price_targets_for_test
from src.train import naive_metrics, price_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="JPY=X")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--n-test", type=int, default=252)
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--model-id", default="amazon/chronos-t5-small")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    prices = fetch_price_series(args.ticker, args.start, end)

    X_all, y_all, origin_all = make_direct_windows(prices, args.horizon, args.lookback)
    train_mask, test_mask = train_test_split_by_tail(origin_all, args.n_test)
    price_base, actual_price = price_targets_for_test(prices, origin_all[test_mask], args.horizon)

    results = {"naive": naive_metrics(price_base, actual_price)}
    print(f"naive   RMSE={results['naive']['rmse']:.4f}  MAPE={results['naive']['mape_pct']:.3f}%")

    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(args.model_id, device_map=str(device), dtype=torch.float32)

    pred_price = []
    infer_start = time.perf_counter()
    for origin_date in origin_all[test_mask]:
        pos = prices.index.get_loc(origin_date)
        context = torch.tensor(
            prices.iloc[max(0, pos - 1 - args.context_len):pos - 1].values, dtype=torch.float32
        )
        forecast = pipeline.predict(inputs=context, prediction_length=args.horizon, num_samples=args.num_samples)
        pred_price.append(float(np.median(forecast[0, :, -1].numpy())))
    infer_time = time.perf_counter() - infer_start
    pred_price = np.array(pred_price)

    metrics = price_metrics(actual_price, pred_price)
    results["chronos"] = {**metrics, "infer_per_sample_ms": infer_time / len(actual_price) * 1000}
    print(f"chronos RMSE={metrics['rmse']:.4f}  MAPE={metrics['mape_pct']:.3f}%  "
          f"infer/sample={infer_time / len(actual_price) * 1000:.2f}ms")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
