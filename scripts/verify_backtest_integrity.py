#!/usr/bin/env python3
"""백테스트 결정성 · 캐시 정합성 · 파라미터 반영 검증.

사용법: PYTHONPATH=.:src python scripts/verify_backtest_integrity.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screener import config as cfg
from paper_trading.backtest import run_paper_trading_backtest

KW = dict(period="3y", max_tickers=100, include_fundamentals=False, initial_capital=5000.0, rebalance_every=1)


def run(label: str, **extra):
    r = run_paper_trading_backtest(**KW, **extra)
    s = r["summary"]
    stops = sum(1 for t in r["trades"] if str(t["exit_reason"]).startswith("손절"))
    print(f"[{label}] 거래={s.get('총거래수')} 최종자본={s.get('최종자본')} 손절={stops}")
    return s.get("최종자본"), [(t["ticker"], t["entry_date"], t["exit_date"]) for t in r["trades"]]


def main() -> None:
    a = run("A 캐시없음", no_cache=True)
    b = run("B 캐시사용")
    assert a == b, "캐시 사용 시 결과가 달라짐 → 캐시 정합성 버그"

    orig = copy.deepcopy(cfg.EXIT_PARAMS)
    for p in cfg.EXIT_PARAMS.values():
        p["stop_loss"] = 0.04
    try:
        c = run("C 손절4%")
    finally:
        cfg.EXIT_PARAMS.clear()
        cfg.EXIT_PARAMS.update(orig)
    assert c != a, "손절을 바꿨는데 결과 동일 → 파라미터 미적용 (문제 E)"

    print("OK: 결정성 + 캐시 정합성 + 파라미터 반영 모두 통과")


if __name__ == "__main__":
    main()
