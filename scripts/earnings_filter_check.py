#!/usr/bin/env python3
"""진단: PT-1 백테스트의 어닝 회피 필터가 얼마나 기여하고, 그중 미래 정보(예정 외 발표) 몫은 얼마인가.

같은 코드·캐시로 BASE 를 세 번 돌린다 (채택용 A/B 아님 — 2026-10-08 BASE 0.83 → 1.02 상승의 원인 분해).
  off       : 어닝 필터 없음 (MACD 단위 수정만 반영된 상태)
  scheduled : 예정 발표만 (같은 회사 45일 안 발표 묶음은 마지막 것만) — 미리 알 수 있던 일정에 가까움
  all       : 실제 8-K 전부 (예정 외 실적 경고까지 미리 아는 셈 → 낙관 편향 상한)

사용법: PYTHONPATH=.:src python scripts/earnings_filter_check.py
산출: tier3/EARNINGS_FILTER_CHECK.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from paper_trading.backtest import run_paper_trading_backtest
from paper_trading.evaluation import paired_bootstrap, verdict
from research_utils import TIER3_DIR

MODES = ("off", "scheduled", "all")
METRICS = ("총거래수", "승률", "Sharpe_일간", "CAGR", "MDD", "총수익률_자본기준", "SPY수익률", "알파_연", "알파_t")


def main() -> None:
    summaries: dict[str, dict] = {}
    returns: dict[str, pd.Series] = {}
    for mode in MODES:
        os.environ["BACKTEST_EARNINGS_CAL"] = mode
        print(f"\n===== 어닝 필터 {mode} =====", flush=True)
        r = run_paper_trading_backtest(period="5y", max_tickers=1000, rebalance_every=1, initial_capital=5000.0,
                                       end_date="2025-09-30", save_run=False, pit_universe=True)
        summaries[mode] = r["summary"]
        eq = r["equity_curve"]
        returns[mode] = eq.pct_change().dropna()
    os.environ.pop("BACKTEST_EARNINGS_CAL", None)

    rows = [{"어닝 필터": m, **{k: summaries[m].get(k) for k in METRICS}} for m in MODES]
    pairs = [("scheduled", "off"), ("all", "off"), ("all", "scheduled")]
    comp = []
    for a, b in pairs:
        common = returns[a].index.intersection(returns[b].index)
        st = paired_bootstrap(returns[a][common].tolist(), returns[b][common].tolist())
        comp.append({"비교": f"{a} − {b}", "ΔSharpe": st.get("ΔSharpe"),
                     "95%": f"[{st.get('ΔSharpe_하한', float('nan')):+.2f}, {st.get('ΔSharpe_상한', float('nan')):+.2f}]",
                     "판정": verdict(st)})
    lines = ["# 진단: 백테스트 어닝 회피 필터의 기여와 미래 정보 몫 (PT-1 BASE)", "",
             "- PIT 2022-08 ~ 2025-09, 같은 코드·피처 캐시. 채택용 A/B 가 아니라 BASE 0.83 → 1.02 상승의 원인 분해",
             "- off = 필터 없음(MACD 단위 수정만) · scheduled = 같은 회사 45일 안 발표 묶음은 마지막 것만(예정 발표 근사, 5.1% 제거) · all = 8-K 전부", "",
             "| 어닝 필터 | 거래 | 승률 | Sharpe | CAGR | MDD | 총수익 | SPY | 알파 | 알파 t |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['어닝 필터']} | {r['총거래수']} | {r['승률']:.1%} | {r['Sharpe_일간']:.2f} | {r['CAGR']:+.1%} | "
                     f"{r['MDD']:.1%} | {r['총수익률_자본기준']:+.1%} | {r['SPY수익률']:+.1%} | {r['알파_연']:+.1%} | {r['알파_t']} |")
    lines += ["", "| 비교 | ΔSharpe | 95% | 판정 |", "|---|---|---|---|"]
    lines += [f"| {c['비교']} | {c['ΔSharpe']:+.2f} | {c['95%']} | {c['판정']} |" for c in comp]
    (TIER3_DIR / "EARNINGS_FILTER_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
