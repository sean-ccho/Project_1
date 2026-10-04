#!/usr/bin/env python3
"""Tier 3-1: 연구용 패널 생성 (날짜 × 종목 피처 + 5/10/20일 선행수익률).

- 피처: data/cache/features/{ohlcv_hash}_v1/YYYY-MM-DD.parquet (날짜 t 종가 기준 스냅샷)
- 선행수익률: 백테스트와 맞춰 **t+1 시가 진입 → t+1+h 시가 청산** (fwd_ret_{h}d)
- --pit: 각 날짜의 S&P 500 실제 구성종목만 남긴다 (생존 편향 제거)

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/build_research_panel.py --pit
산출: data/research/panel_{pit|nonpit}.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from paper_trading.universe import Membership, load_membership

HORIZONS = (5, 10, 20)
# (feature hash, ohlcv cache) — PIT 확장 유니버스 / 현재 구성종목(생존편향) 유니버스
SETS = {
    "pit": ("fe5a595b6df7", "ohlcv_e7842ef0a144.parquet"),
    "nonpit": ("5d51596c2b74", "ohlcv_e11fe9794311.parquet"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pit", action="store_true", help="PIT 멤버십으로 필터 (권장)")
    ap.add_argument("--set", choices=list(SETS), default=None)
    args = ap.parse_args()
    key = args.set or ("pit" if args.pit else "nonpit")
    fhash, ohlcv_file = SETS[key]

    ohlcv = pd.read_parquet(ROOT / "data/cache" / ohlcv_file)
    opens = ohlcv.xs("Open", axis=1, level="Price").sort_index()
    dates_all = opens.index

    fwd = {}
    for h in HORIZONS:
        entry = opens.shift(-1)
        exit_ = opens.shift(-(1 + h))
        fwd[h] = exit_ / entry - 1.0

    mem = None
    if key == "pit":
        mem = Membership(load_membership(ROOT / "data/universe/sp500_membership.csv"))

    frames = []
    files = sorted((ROOT / "data/cache/features" / f"{fhash}_v1").glob("*.parquet"))
    for f in files:
        dt = pd.Timestamp(f.stem)
        if dt not in dates_all:
            continue
        df = pd.read_parquet(f)
        if mem is not None:
            members = mem.on(f.stem)
            df = df[df["티커"].isin(members)]
        df = df.copy()
        df.insert(0, "date", dt)
        for h in HORIZONS:
            df[f"fwd_ret_{h}d"] = df["티커"].map(fwd[h].loc[dt])
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    out_dir = ROOT / "data/research"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"panel_{key}.parquet"
    panel.to_parquet(out)

    print(f"저장: {out}")
    print(f"행 {len(panel):,} | 날짜 {panel['date'].nunique()} | 종목 {panel['티커'].nunique()}")
    print(f"기간 {panel['date'].min().date()} ~ {panel['date'].max().date()}")
    for h in HORIZONS:
        print(f"  fwd_ret_{h}d 유효 {panel[f'fwd_ret_{h}d'].notna().mean():.1%}")


if __name__ == "__main__":
    main()
