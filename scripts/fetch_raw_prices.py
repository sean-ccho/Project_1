#!/usr/bin/env python3
"""시가총액 계산용 비조정 종가(배당 미반영)와 주식 분할 이력 (px10y 종목, ~2025-09-30).

yfinance auto_adjust=False 의 Close 는 분할만 반영된 값이다. 분할 이력으로 그날의 실제 주가를 되돌린다.
사용법: PYTHONPATH=.:src python scripts/fetch_raw_prices.py
산출 (git 제외): data/research/raw_close_px10y.parquet, data/research/splits_px10y.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import yfinance as yf

from research_utils import RESEARCH_DIR

START, END = "2014-01-01", "2025-10-01"  # END 는 미포함 → 2025-09-30 까지


def main() -> None:
    tickers = sorted(pd.read_parquet(RESEARCH_DIR / "panel_px10y.parquet", columns=["티커"])["티커"].unique())
    closes, splits = [], []
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        d = yf.download(batch, start=START, end=END, interval="1d", auto_adjust=False, actions=True,
                        group_by="column", progress=False, threads=True)
        closes.append(d["Close"])
        sd = d["Stock Splits"].astype(float)
        s = sd.where(sd > 0).stack().rename("ratio").reset_index()
        s.columns = ["date", "티커", "ratio"]
        splits.append(s)
        print(f"[raw] {min(i + 100, len(tickers))}/{len(tickers)}", flush=True)
    close = pd.concat(closes, axis=1).sort_index().loc[:"2025-09-30"]
    close.to_parquet(RESEARCH_DIR / "raw_close_px10y.parquet")
    sp = pd.concat(splits, ignore_index=True)
    sp = sp.dropna(subset=["ratio"])
    sp.to_parquet(RESEARCH_DIR / "splits_px10y.parquet", index=False)
    print(f"[raw] 종가 {close.shape}, 분할 {len(sp)}건")


if __name__ == "__main__":
    main()
