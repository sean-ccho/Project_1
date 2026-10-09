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
        10년: ... ccs_placebo.py --period 13y --start 2016-01-04 --end 2026-10-07 --tag 10y --family ccs_placebo_10y
산출: tier3/CCS_PLACEBO{_tag}.md, tier3/ccs_placebo{_tag}.csv
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
EQUITY: dict[str, pd.Series] = {}  # 실제 CCS 실행의 자본곡선 (연도별 표용)


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
            r = bt.run_paper_trading_backtest(**RUN_KW)
    finally:
        bt.select_top_candidates = ORIGINAL
    s = r["summary"]
    eq = r["equity_curve"].dropna() if r.get("equity_curve") is not None else pd.Series(dtype=float)
    if seed is None:
        EQUITY[label] = eq
    total = float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else float("nan")  # 일별 평가액 기준 (기간종료 청산 결측과 무관)
    row = {"set": label, "seed": seed, "Sharpe": s.get("Sharpe_일간"), "CAGR": s.get("CAGR"), "MDD": s.get("MDD"),
           "총수익": total, "거래": s.get("총거래수"), "승률": s.get("승률"),
           "SPY_Sharpe": s.get("기준_SPY_Sharpe"), "알파_연": s.get("알파_연"), "알파_t": s.get("알파_t")}
    print(f"[placebo] {label} seed={seed} → Sharpe {row['Sharpe']} · 총 {row['총수익']:+.1%} · MDD {row['MDD']:.1%}", flush=True)
    return row


def _registered(family: str) -> bool:
    if not TRIAL_LOG.exists():
        return False
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    return bool(((log["family"] == family) & (log["판정"] == "등록")).any())


def _yearly_table(spy: pd.Series | None) -> list[str]:
    """실제 CCS 실행들의 연도별 수익 (일별 평가액 기준) + SPY."""
    cols = dict(EQUITY)
    if spy is not None and not spy.empty:
        cols["SPY"] = spy
    if not cols:
        return []
    yr = pd.DataFrame({k: v.groupby(v.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1) for k, v in cols.items()})
    out = ["", "## 연도별 수익 (실제 CCS, 연초~연말 평가액)", "", "| 연도 | " + " | ".join(yr.columns) + " |",
           "|---" * (len(yr.columns) + 1) + "|"]
    out += [f"| {y} | " + " | ".join(f"{v:+.1%}" for v in row) + " |" for y, row in yr.iterrows()]
    return out


def main() -> None:
    global FAMILY
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--period", default="5y")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default="2025-09-30")
    ap.add_argument("--tag", default="", help="산출 파일 이름 꼬리 (예: 10y)")
    ap.add_argument("--family", default=FAMILY)
    args = ap.parse_args()
    FAMILY = args.family
    RUN_KW.update(period=args.period, start_date=args.start, end_date=args.end or None)
    suffix = f"_{args.tag}" if args.tag else ""
    span = f"{args.start or '(기간 시작)'} ~ {args.end or '(최근)'}"
    if not _registered(FAMILY):
        log_trials("stage6", FAMILY, 2, [{"test_id": "(등록)", "판정": "등록",
                    "메모": f"[{span}, period={args.period}] CCS 순위 vs 문턱 통과 후보 중 무작위 선택 ({args.seeds}씨앗). "
                            "T1 CCS Sharpe > 무작위 95백분위, T2 모멘텀만·무작위가 전체·무작위보다 높을 확률 ≥ 80%"}])
    mom = {"CANDIDATE_ALLOWED_STRATEGIES": ["모멘텀"]}
    rows = [run("CCS(실제)", None, {}), run("모멘텀만 CCS(실제)", None, mom)]
    for seed in range(args.seeds):
        rows.append(run("무작위", seed, {}))
    for seed in range(args.seeds):
        rows.append(run("모멘텀만 무작위", seed, mom))
    df = pd.DataFrame(rows)
    df.to_csv(TIER3_DIR / f"ccs_placebo{suffix}.csv", index=False)

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
    spy_eq = None
    try:  # SPY 자본곡선 (같은 기간, 시가→시가가 아니라 종가 기준 — 연도별 비교용)
        from data.fetch import fetch_ohlcv
        eq0 = next(iter(EQUITY.values()))
        px = fetch_ohlcv(["SPY"], period=args.period)["SPY"]["Close"].dropna()
        spy_eq = px.loc[eq0.index.min():eq0.index.max()]
    except Exception as e:  # 표만 빠진다
        print(f"[placebo] SPY 연도별 생략: {e}")
    base_row = df[df["set"] == "CCS(실제)"].iloc[0]
    lines = [f"# 플라시보 테스트: CCS 순위 vs 무작위 선택 (PT-1 백테스트, PIT {span})", "",
             f"- 실제 CCS: Sharpe {base_row['Sharpe']:.2f} vs SPY {base_row['SPY_Sharpe']} · 알파 {base_row['알파_연']:+.1%}/년 (t={base_row['알파_t']}) · "
             f"CAGR {base_row['CAGR']:+.1%} · MDD {base_row['MDD']:.1%}",
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
    lines += _yearly_table(spy_eq)
    (TIER3_DIR / f"CCS_PLACEBO{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:15]))


if __name__ == "__main__":
    main()
