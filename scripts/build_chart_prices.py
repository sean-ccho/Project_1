#!/usr/bin/env python3
"""연구 백테스트용 차트 가격을 한 번 받아 얼린다 (2026-10-27, docs/goldencross.md 'C').

패턴 · 교차 라벨은 TradingView 차트와 같은 가격(분할만 반영 · 상장 이후 전체)으로 본다. 밤 실행은 매번
data.fetch.fetch_chart_prices 로 받고, 연구 백테스트(scripts/isb_vs_pt1.py _patched)는 이 파일만 읽는다
(없으면 받지 않고 멈춘다 — 실행마다 다른 날 받은 값이 섞이지 않게).

- 종목: PT-1 엔진 BASE 와 같은 고정 목록 (data/research/isb_pit_tickers_13y.json + SPY, max_tickers 까지)
- 일봉은 2008-01-01 부터 남긴다 (개발 구간 첫날 2016-01-04 의 5년 전보다 앞), 주·월봉은 상장 이후 전체
- 저장: data/cache/chart_prices/<종목 목록 지문>.pkl (새 폴더 — 다른 캐시는 건드리지 않는다). 이미 있으면 멈춘다

사용법: PYTHONPATH=.:src .venv/bin/python scripts/build_chart_prices.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import isb_vs_pt1 as iv  # noqa: E402
from data.fetch import chart_prices_path, fetch_chart_prices, save_chart_prices  # noqa: E402
from validate_upside_walkforward import RUN_KW  # noqa: E402

DAILY_SINCE = pd.Timestamp("2008-01-01")


def engine_tickers() -> list[str]:
    """PT-1 엔진이 fetch_chart_prices 에 넘기는 목록 (goldencross_pt1_check.cache_key 와 같은 계산)."""
    tickers = iv._pinned_tickers()
    tickers = tickers if "SPY" in tickers else ["SPY"] + tickers
    return ["SPY"] + [t for t in tickers if t != "SPY"][: RUN_KW["max_tickers"]]


def main() -> None:
    tickers = engine_tickers()
    path = ROOT / chart_prices_path(tickers)
    if path.exists():
        raise SystemExit(f"이미 있음: {path.relative_to(ROOT)} — 다시 받으면 값이 달라질 수 있어 멈춘다 (지우려면 직접)")
    print(f"[차트 가격] {len(tickers)}종목 받기 (상장 이후 전체 · 분할만 반영) → {path.relative_to(ROOT)}", flush=True)
    chart = fetch_chart_prices(tickers, daily_since=DAILY_SINCE)
    missing = [t for t in tickers if t not in chart]
    first = pd.Series({t: cp.monthly.index[0] for t, cp in chart.items() if len(cp.monthly)})
    meta = {"made": datetime.now().isoformat(timespec="seconds"), "tickers": str(len(tickers)),
            "got": str(len(chart)), "missing": ",".join(missing), "daily_since": str(DAILY_SINCE.date())}
    save_chart_prices(path, chart, meta)
    years = ((pd.Timestamp.now() - first).dt.days / 365.25)
    print(f"[차트 가격] 저장 {len(chart)}종목 · 못 받음 {len(missing)}개 {missing[:20]}", flush=True)
    print(f"  월봉 기록 길이: 중앙 {years.median():.0f}년 · 17년 이상(월봉 EMA200 있음) {int((years >= 200 / 12).sum())}종목 · "
          f"8년 미만(월봉 EMA100 없음) {int((years < 100 / 12).sum())}종목 · 파일 {path.stat().st_size / 1e6:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
