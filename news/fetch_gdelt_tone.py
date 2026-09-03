"""
GDELT DOC 2.0 API(無料・APIキー不要)から、指定クエリの日次トーン(平均センチメント)と
記事量を取得する。

【重要な注意】
- 実用上、2017年以降のデータしか取得できない(それ以前は "Invalid query start date" エラー)
- レート制限は「5秒に1回」。これを超えると一時的にIPブロックされ、TCP接続自体が
  タイムアウトするようになることがある(数十分〜数時間で解除される様子)。焦って
  連投しないこと。

使用例:
    python fetch_gdelt_tone.py --query "yen dollar exchange rate" \\
        --start 2017-01-01 --end 2026-09-02 --out out/gdelt_tone_daily.csv
"""
import argparse
import json
import time

import pandas as pd
import requests

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_timeline(query: str, mode: str, dt_from: pd.Timestamp, dt_to: pd.Timestamp,
                    sleep_sec: float, retries: int = 3):
    params = {
        "query": query, "mode": mode,
        "startdatetime": dt_from.strftime("%Y%m%d%H%M%S"),
        "enddatetime": dt_to.strftime("%Y%m%d%H%M%S"),
        "format": "json",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except json.JSONDecodeError:
                    print(f"    JSON decode失敗(レート制限の可能性): {resp.text[:150]}")
            else:
                print(f"    HTTP {resp.status_code}: {resp.text[:150]}")
        except requests.exceptions.RequestException as e:
            print(f"    request error: {e}")
        time.sleep(sleep_sec * (attempt + 2))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="yen dollar exchange rate")
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2026-09-02")
    parser.add_argument("--sleep-sec", type=float, default=6.0)
    parser.add_argument("--out", default="out/gdelt_tone_daily.csv")
    args = parser.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    chunks = []
    cur = start
    while cur < end:
        nxt = min(cur + pd.DateOffset(years=1), end)
        chunks.append((cur, nxt))
        cur = nxt

    all_tone, all_vol = [], []
    for i, (dt_from, dt_to) in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] {dt_from.date()} - {dt_to.date()}")

        tone_data = fetch_timeline(args.query, "timelinetone", dt_from, dt_to, args.sleep_sec)
        if tone_data and "timeline" in tone_data:
            series = tone_data["timeline"][0]["data"]
            for pt in series:
                all_tone.append({"date": pd.to_datetime(pt["date"]).normalize(), "tone": pt["value"]})
            print(f"    tone: {len(series)}日")
        else:
            print("    tone: FAILED")
        time.sleep(args.sleep_sec)

        vol_data = fetch_timeline(args.query, "timelinevolraw", dt_from, dt_to, args.sleep_sec)
        if vol_data and "timeline" in vol_data:
            series = vol_data["timeline"][0]["data"]
            for pt in series:
                all_vol.append({"date": pd.to_datetime(pt["date"]).normalize(), "n_articles": pt["value"]})
            print(f"    volume: {len(series)}日")
        else:
            print("    volume: FAILED")
        if i < len(chunks) - 1:
            time.sleep(args.sleep_sec)

    tone_df = pd.DataFrame(all_tone).drop_duplicates(subset="date").sort_values("date")
    vol_df = pd.DataFrame(all_vol).drop_duplicates(subset="date").sort_values("date")
    merged = pd.merge(tone_df, vol_df, on="date", how="outer").sort_values("date")
    merged.to_csv(args.out, index=False)
    print(f"\nsaved: {args.out} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
