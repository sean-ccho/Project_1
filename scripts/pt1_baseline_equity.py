#!/usr/bin/env python3
"""PT-1 기준선: 지금 규칙(BASE) 그대로 개발 구간(2016-01-04 ~ 2025-09-30)을 한 번 돌려 일별 평가액을 저장한다.

idea_batch_research.py 의 'PT-1 대비' 비교(참고용)에 쓴다. 검정이 아닌 기준선이라 TRIAL_LOG 에 쓰지 않는다.
설정·가격 고정은 validate_upside_walkforward.py 와 같다 (13y PIT, 매일 리밸런스, $5,000, 디스크 가격 캐시를 나이와 상관없이 읽음).
10년 피처 캐시 이름이 지금 코드의 키와 맞아야 한다 (scripts/rename_feature_cache.py — HANDOFF §7).

사용법: PYTHONPATH=.:src python scripts/pt1_baseline_equity.py
산출 (git 제외): data/research/pt1_base_dev_equity.parquet (열 equity, 날짜 인덱스)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import paper_trading.backtest as bt
from research_utils import RESEARCH_DIR
from validate_upside_walkforward import _ORIGINAL_FETCH, RUN_KW, frozen_fetch

OUT = RESEARCH_DIR / "pt1_base_dev_equity.parquet"


def main() -> None:
    print(f"===== PT-1 기준선 (BASE, {RUN_KW['start_date']} ~ {RUN_KW['end_date']})", flush=True)
    bt.fetch_ohlcv = frozen_fetch
    try:
        result = bt.run_paper_trading_backtest(**RUN_KW)
    finally:
        bt.fetch_ohlcv = _ORIGINAL_FETCH
    eq = result["equity_curve"].dropna()
    eq.index = pd.to_datetime(eq.index)
    eq.rename("equity").to_frame().to_parquet(OUT)

    s = result["summary"]
    print(f"\nSharpe(일간) {s.get('Sharpe_일간')} · CAGR {s.get('CAGR')} · MDD {s.get('MDD')} · 거래 {s.get('총거래수')}")
    yr = eq.groupby(eq.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)  # ccs_placebo 연도별 표와 같은 방식
    print("연도별 수익 (연초~연말 평가액):")
    print(yr.map(lambda v: f"{v:+.1%}").to_string())
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
