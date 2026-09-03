"""
Alpha Vantage NEWS_SENTIMENT APIから、指定期間にわたって等間隔にサンプリングした
地点のニュースセンチメントを取得する。

【重要な注意】"forex"は実在しないトピックタグ。有効なトピックは:
  blockchain, earnings, ipo, mergers_and_acquisitions, financial_markets,
  economy_fiscal, economy_monetary, economy_macro, energy_transportation,
  finance, life_sciences, manufacturing, real_estate, retail_wholesale, technology
為替・金融政策に関連させたい場合は economy_monetary や economy_macro を使うこと。

無料枠はレート制限(1日25リクエスト程度)が厳しいため、全期間から
N_SAMPLES点を均等間隔でサンプリングする方式を取る(連続日次データにはならない)。
密な日次データが必要な場合は fetch_gdelt_tone.py を使う方が良い
(ただしGDELTは2017年以降のデータしかカバーしない)。

使用例:
    python fetch_alphavantage_news.py --api-key YOUR_KEY --topics economy_monetary,economy_macro \\
        --start 2010-01-01 --end 2026-09-02 --n-samples 20 --out out/news_samples.csv
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import requests


def fetch_window(api_key: str, topics: str, time_from: pd.Timestamp, time_to: pd.Timestamp, limit: int = 200):
    params = {
        "function": "NEWS_SENTIMENT",
        "topics": topics,
        "time_from": time_from.strftime("%Y%m%dT0000"),
        "time_to": time_to.strftime("%Y%m%dT2359"),
        "limit": limit,
        "sort": "RELEVANCE",
        "apikey": api_key,
    }
    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--topics", default="economy_monetary,economy_macro")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-09-02")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--window-days", type=int, default=30, help="各サンプル地点で遡って集計する日数")
    parser.add_argument("--sleep-sec", type=float, default=15.0)
    parser.add_argument("--out", default="out/news_sentiment_samples.csv")
    args = parser.parse_args()

    sample_dates = pd.date_range(args.start, args.end, periods=args.n_samples)
    print(f"サンプリングした{args.n_samples}地点: {[d.date().isoformat() for d in sample_dates]}")

    records = []
    for i, origin_date in enumerate(sample_dates):
        time_to = origin_date
        time_from = origin_date - pd.Timedelta(days=args.window_days)
        data = fetch_window(args.api_key, args.topics, time_from, time_to)

        if "feed" not in data:
            print(f"  [{i+1}/{args.n_samples}] {origin_date.date()}: ERROR -> {json.dumps(data)[:200]}")
            records.append({"origin_date": origin_date, "n_articles": 0, "mean_sentiment": np.nan})
        else:
            feed = data["feed"]
            scores = [float(a["overall_sentiment_score"]) for a in feed]
            mean_sentiment = float(np.mean(scores)) if scores else np.nan
            print(f"  [{i+1}/{args.n_samples}] {origin_date.date()}: n={len(feed)}  mean_sentiment={mean_sentiment:.4f}")
            records.append({"origin_date": origin_date, "n_articles": len(feed), "mean_sentiment": mean_sentiment})

        if i < args.n_samples - 1:
            time.sleep(args.sleep_sec)

    result_df = pd.DataFrame(records)
    result_df.to_csv(args.out, index=False)
    print(f"\nsaved: {args.out} ({len(result_df)} rows)")


if __name__ == "__main__":
    main()
