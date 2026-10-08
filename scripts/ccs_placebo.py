#!/usr/bin/env python3
"""플라시보 테스트: CCS 순위가 '자격 있는 후보 중 무작위 선택'보다 나은가.

PT-1 BASE 백테스트를 그대로 돌리되, 그날 CCS 문턱을 넘은 후보(하드 필터·섹터·문턱 모두 같음) 중에서
1등 대신 **무작위로** 한 종목을 고른다. 교체 판단도 그 종목의 실제 CCS 로 한다. 씨앗을 바꿔 여러 번 돌려
"운으로 나올 수 있는 성과"의 분포를 만들고, 실제 CCS 결과가 그 분포의 어디쯤인지 본다.

사전 등록 (TRIAL_LOG family ccs_placebo, 2건):
  T1 CCS 순위 효과 : 실제 CCS Sharpe 가 무작위(전체 전략) 분포의 95백분위를 넘는가
  T2 모멘텀만 효과 : 모멘텀만·무작위 분포가 전체·무작위 분포보다 높은가 (중앙값 차이, 순위와 무관한 전략 선택 효과)
한계: 시장 경로는 2022-08~2025-09 하나뿐 — 선택의 운만 재고, 그 기간이 좋았던 운은 못 잰다.

사용법: PYTHONPATH=.:src python scripts/ccs_placebo.py [--seeds 20]
산출: tier3/CCS_PLACEBO.md, tier3/ccs_placebo.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

import paper_trading.backtest as bt
from paper_trading import candidate_selector as cs
from paper_trading.config_override import config_overrides
from research_utils import TIER3_DIR, TRIAL_LOG, log_trials, total_trials
from screener import config as _cfg

FAMILY = "ccs_placebo"
ORIGINAL = cs.select_top_candidates
RUN_KW = dict(period="5y", max_tickers=1000, rebalance_every=1, initial_capital=5000.0,
              end_date="2025-09-30", save_run=False, pit_universe=True)


def random_selector(seed: int):
    """CCS 문턱을 넘은 후보 중 무작위 1개 (나머지 규칙은 원래 함수 그대로)."""
    rng = np.random.default_rng(seed)

    def select(df: pd.DataFrame, holdings: list[dict], k: int = 1):
        picks, debug = ORIGINAL(df, holdings, k=k)
        if not picks:
            return picks, debug
        regime = debug.get("regime", "neutral")
        min_ccs = _cfg.CANDIDATE_CCS_MIN_BEAR if regime == "bear" else _cfg.CANDIDATE_CCS_MIN_NORMAL
        pool = [s for s in debug.get("all_scores", {}).values() if s["ccs"] >= min_ccs]
        if not pool:
            return picks, debug
        chosen = [pool[i] for i in rng.choice(len(pool), size=min(k, len(pool)), replace=False)]
        return [cs._candidate_from_score(s, df.loc[s["idx"]]) for s in chosen], debug
    return select


def run(label: str, seed: int | None, overrides: dict) -> dict:
    bt.select_top_candidates = ORIGINAL if seed is None else random_selector(seed)
    try:
        with config_overrides(overrides):
            s = bt.run_paper_trading_backtest(**RUN_KW)["summary"]
    finally:
        bt.select_top_candidates = ORIGINAL
    row = {"set": label, "seed": seed, "Sharpe": s.get("Sharpe_일간"), "CAGR": s.get("CAGR"), "MDD": s.get("MDD"),
           "총수익": s.get("총수익률_자본기준"), "거래": s.get("총거래수"), "승률": s.get("승률")}
    print(f"[placebo] {label} seed={seed} → Sharpe {row['Sharpe']} · 총 {row['총수익']:+.1%} · MDD {row['MDD']:.1%}", flush=True)
    return row


def _registered() -> bool:
    if not TRIAL_LOG.exists():
        return False
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    return bool(((log["family"] == FAMILY) & (log["판정"] == "등록")).any())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    if not _registered():
        log_trials("stage6", FAMILY, 2, [{"test_id": "(등록)", "판정": "등록",
                    "메모": f"CCS 순위 vs 문턱 통과 후보 중 무작위 선택 ({args.seeds}씨앗). T1 CCS Sharpe > 무작위 95백분위, "
                            "T2 모멘텀만·무작위 중앙값 > 전체·무작위 중앙값"}])
    mom = {"CANDIDATE_ALLOWED_STRATEGIES": ["모멘텀"]}
    rows = [run("CCS(실제)", None, {}), run("모멘텀만 CCS(실제)", None, mom)]
    for seed in range(args.seeds):
        rows.append(run("무작위", seed, {}))
    for seed in range(args.seeds):
        rows.append(run("모멘텀만 무작위", seed, mom))
    df = pd.DataFrame(rows)
    df.to_csv(TIER3_DIR / "ccs_placebo.csv", index=False)

    rnd, rnd_m = df[df["set"] == "무작위"], df[df["set"] == "모멘텀만 무작위"]
    ccs = float(df.loc[df["set"] == "CCS(실제)", "Sharpe"].iloc[0])
    ccs_m = float(df.loc[df["set"] == "모멘텀만 CCS(실제)", "Sharpe"].iloc[0])
    pct = float((rnd["Sharpe"] < ccs).mean())
    pct_m = float((rnd_m["Sharpe"] < ccs_m).mean())
    p95 = float(rnd["Sharpe"].quantile(0.95))
    t1 = "채택 후보" if ccs > p95 else "운과 구분 안 됨"
    med_gap = float(rnd_m["Sharpe"].median() - rnd["Sharpe"].median())
    overlap = float((rnd_m["Sharpe"].to_numpy()[:, None] > rnd["Sharpe"].to_numpy()[None, :]).mean())
    t2 = "관찰" if overlap >= 0.8 else "운과 구분 안 됨"
    log_trials("stage6", FAMILY, 2, [
        {"test_id": "T1 CCS 순위", "판정": t1, "메모": f"CCS Sharpe {ccs:.2f}, 무작위 백분위 {pct:.0%} (95백분위 {p95:.2f})"},
        {"test_id": "T2 모멘텀만", "판정": t2, "메모": f"중앙값 차이 {med_gap:+.2f}, 모멘텀만 무작위가 전체 무작위보다 높을 확률 {overlap:.0%}"}])

    def desc(g: pd.DataFrame) -> str:
        q = g["Sharpe"].quantile([0.05, 0.5, 0.95])
        return (f"Sharpe 중앙 {q[0.5]:.2f} (5~95%: {q[0.05]:.2f} ~ {q[0.95]:.2f}) · 총수익 중앙 {g['총수익'].median():+.0%} · "
                f"MDD 중앙 {g['MDD'].median():.1%}")
    lines = ["# 플라시보 테스트: CCS 순위 vs 무작위 선택 (PT-1 백테스트, PIT 2022-08 ~ 2025-09)", "",
             "- 같은 하드 필터·섹터 제한·CCS 문턱을 통과한 후보 중 1등 대신 무작위 1개를 고른다. 교체 판단은 그 종목의 실제 CCS",
             f"- 씨앗 {args.seeds}개씩. 시장 경로는 하나뿐이라 '선택의 운'만 잰다 · 누적 시도 {total_trials()}건", "",
             "| 묶음 | 결과 |", "|---|---|",
             f"| CCS 실제 (전체 전략) | Sharpe {ccs:.2f} → 무작위 분포의 **{pct:.0%} 백분위** |",
             f"| 무작위 (전체 전략) | {desc(rnd)} |",
             f"| CCS 실제 (모멘텀만) | Sharpe {ccs_m:.2f} → 모멘텀만 무작위 분포의 **{pct_m:.0%} 백분위** |",
             f"| 무작위 (모멘텀만) | {desc(rnd_m)} |", "",
             f"- **T1 CCS 순위 효과: {t1}** (실제 {ccs:.2f} vs 무작위 95백분위 {p95:.2f})",
             f"- **T2 모멘텀만 효과: {t2}** (무작위끼리 중앙값 차이 {med_gap:+.2f}, 모멘텀만이 더 높을 확률 {overlap:.0%})", "",
             "## 씨앗별", "", "| 묶음 | 씨앗 | Sharpe | CAGR | MDD | 총수익 | 거래 |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r.set} | {'' if pd.isna(r.seed) else int(r.seed)} | {r.Sharpe:.2f} | {r.CAGR:+.1%} | {r.MDD:.1%} | {r.총수익:+.1%} | {r.거래} |"
              for r in df.itertuples()]
    (TIER3_DIR / "CCS_PLACEBO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
