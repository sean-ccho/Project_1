#!/usr/bin/env python3
"""Tier 3-B 공통: 신호 → 겹치는 h일 보유 포트폴리오 (패널 기반 빠른 백테스트).

규칙: 날짜 d 신호 상위 k개 → d+1 시가 진입, d+1+h 시가 청산. 매일 새 하위 포트폴리오(종목당 비중 1/(k·h))를
만들어 h개가 겹친다. 신호가 k개보다 적으면 남는 비중은 현금. 비용은 편도 COST_PER_SIDE.
입력 events: date, 티커, event_id, score(높을수록 우선)

후보를 무거운 PT 엔진에 넣기 전에 빠르게 거른다. 피처 캐시나 config에 의존하지 않는다.
가격은 research_utils.load_ohlcv로 읽어 개발 구간(≤ 2025-09-30)만 쓴다.

비교 대상: SPY 시가→시가, 그날 S&P 500 구성종목 동일가중(PIT).
시뮬 통과 = SPY 대비 paired_bootstrap ΔSharpe_하한 > 0. MDD는 6단계 A/B에서 본다.
한계: 상장폐지 이후 결측 수익은 0(현금)으로 처리한다. 비중 드리프트도 무시한다.

사용법:
  PYTHONPATH=.:src python scripts/signal_portfolio_sim.py --events data/research/events_pit.parquet \\
      --event-id S7_top1 --ohlcv data/cache/ohlcv_e7842ef0a144.parquet --k 3 --hold 10 --name s7_top1
산출: docs/quant_improvement/tier3/PORTFOLIO_SIM_{name}.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src를 맨 앞에: scripts/paper_trading.py 가 src/paper_trading 패키지를 가리지 않게
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from paper_trading.evaluation import alpha_beta, cagr, max_drawdown, paired_bootstrap, sharpe
from paper_trading.universe import Membership, load_membership
from research_utils import COST_PER_SIDE, TIER3_DIR, load_ohlcv

MEMBERSHIP_CSV = ROOT / "data" / "universe" / "sp500_membership.csv"
# SPY가 없는 OHLCV일 때 대체 (≤ DEV_END로 자름). etf_ohlcv(2015~, signal_event_study.py가 생성)를 먼저 쓴다
BENCH_OHLCV = [ROOT / "data" / "research" / "etf_ohlcv.parquet", ROOT / "data" / "cache" / "ohlcv_e7842ef0a144.parquet"]


def open_to_open(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """t 시가 → t+1 시가 수익률 (t에 기록). 마지막 날은 NaN."""
    opens = ohlcv.xs("Open", axis=1, level="Price").sort_index()
    return opens.shift(-1) / opens - 1.0


def simulate(signals: pd.DataFrame, r_oo: pd.DataFrame, k: int, hold: int,
             cost: float = COST_PER_SIDE) -> pd.Series:
    """신호 → 일간 포트폴리오 수익률 (비용 차감)."""
    dates = r_oo.index
    pos = {d: i for i, d in enumerate(dates)}
    col = {t: j for j, t in enumerate(r_oo.columns)}
    w = np.zeros(r_oo.shape)
    for d, g in signals.groupby("date"):
        i = pos.get(pd.Timestamp(d))
        if i is None or i + 1 + hold >= len(dates):
            continue
        names = [t for t in g.sort_values("score", ascending=False)["티커"] if t in col][:k]
        for t in names:
            w[i + 1 : i + 1 + hold, col[t]] += 1.0 / (k * hold)
    weights = pd.DataFrame(w, index=dates, columns=r_oo.columns)
    gross = (weights * r_oo.fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    return (gross - turnover * cost).loc[signals["date"].min():]


def equal_weight_pit(r_oo: pd.DataFrame, mem: Membership) -> pd.Series:
    """그날 S&P 500 구성종목(가격 있는 것) 동일가중 시가→시가 수익률."""
    out = {}
    cols = set(r_oo.columns)
    for d in r_oo.index:
        names = [t for t in mem.on(f"{d:%Y-%m-%d}") if t in cols]
        out[d] = float(r_oo.loc[d, names].mean()) if names else np.nan
    return pd.Series(out)


def spy_open_to_open(r_oo: pd.DataFrame) -> pd.Series:
    """SPY 시가→시가. 입력 OHLCV에 없으면 BENCH_OHLCV에서 가져온다 (구간이 모자라면 NaN)."""
    if "SPY" in r_oo.columns:
        return r_oo["SPY"]
    path = next(p for p in BENCH_OHLCV if p.exists())
    bench = open_to_open(load_ohlcv(path))
    return bench["SPY"].reindex(r_oo.index)


def summarize(name: str, strat: pd.Series, spy: pd.Series, ew: pd.Series, meta: dict) -> str:
    """성과·SPY/동일가중 대비 ΔSharpe 95% 구간·연도별 수익 → Markdown."""
    df = pd.DataFrame({"strat": strat, "SPY": spy, "EW": ew}).dropna()
    s, b, e = df["strat"].tolist(), df["SPY"].tolist(), df["EW"].tolist()
    ab = alpha_beta(s, {"SPY": b})
    vs_spy = paired_bootstrap(s, b)
    vs_ew = paired_bootstrap(s, e)
    passed = bool(vs_spy) and vs_spy["ΔSharpe_하한"] > 0

    lines = [f"# 패널 포트폴리오 시뮬 — {name}", ""]
    lines += [f"- {k}: {v}" for k, v in meta.items()]
    lines += [f"- 구간: {df.index.min().date()} ~ {df.index.max().date()} ({len(df)}일, 개발 구간만)",
              f"- **시뮬 통과(SPY 대비 ΔSharpe 95% 하한 > 0): {'예' if passed else '아니오'}**", "",
              "| 시계열 | CAGR | Sharpe | MDD |", "|---|---|---|---|"]
    for col, label in (("strat", "전략"), ("SPY", "SPY"), ("EW", "PIT 동일가중")):
        r = df[col].tolist()
        lines.append(f"| {label} | {cagr(r):+.1%} | {sharpe(r):.2f} | {max_drawdown(r):.1%} |")
    lines += ["", f"- 알파(연, SPY 회귀): {ab.get('알파_연', float('nan')):+.2%} "
              f"(t={ab.get('알파_t', float('nan')):+.2f}), 베타 {ab.get('베타_SPY', float('nan')):.2f}", ""]
    lines += ["| 비교 | ΔSharpe | 95% 하한 | 95% 상한 | p_개선아님 | 구간 개선 |", "|---|---|---|---|---|---|"]
    for label, st in (("vs SPY", vs_spy), ("vs PIT 동일가중", vs_ew)):
        if st:
            lines.append(f"| {label} | {st['ΔSharpe']:+.2f} | {st['ΔSharpe_하한']:+.2f} | {st['ΔSharpe_상한']:+.2f} | "
                         f"{st['p_개선아님']:.2f} | {st['구간_개선']}/{st['구간_수']} |")
        else:
            lines.append(f"| {label} | 데이터 부족 | | | | |")
    yr = (1 + df).groupby(df.index.year).prod() - 1
    lines += ["", "## 연도별 수익", "", "| 연도 | 전략 | SPY | PIT 동일가중 |", "|---|---|---|---|"]
    lines += [f"| {y} | {r.strat:+.1%} | {r.SPY:+.1%} | {r.EW:+.1%} |" for y, r in yr.iterrows()]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="parquet: date, 티커, event_id, score")
    ap.add_argument("--event-id", required=True)
    ap.add_argument("--ohlcv", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--cost", type=float, default=COST_PER_SIDE)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=str(TIER3_DIR))
    args = ap.parse_args()

    ev = pd.read_parquet(args.events)
    ev["date"] = pd.to_datetime(ev["date"])
    ev = ev[ev["event_id"] == args.event_id]
    if ev.empty:
        raise SystemExit(f"event_id={args.event_id} 이벤트 없음")

    r_oo = open_to_open(load_ohlcv(args.ohlcv))
    ev = ev[ev["date"] <= r_oo.index.max()]
    strat = simulate(ev, r_oo, args.k, args.hold, args.cost)
    # 마지막 날(r_oo NaN)은 버린다
    strat = strat.loc[: r_oo.index[-2]]
    spy = spy_open_to_open(r_oo)
    ew = equal_weight_pit(r_oo.loc[strat.index], Membership(load_membership(MEMBERSHIP_CSV)))

    meta = {"events": args.events, "event_id": args.event_id, "ohlcv": args.ohlcv,
            "k": args.k, "hold": args.hold, "편도 비용": args.cost, "이벤트 수": len(ev)}
    md = summarize(args.name, strat, spy, ew, meta)
    out = Path(args.out) / f"PORTFOLIO_SIM_{args.name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
