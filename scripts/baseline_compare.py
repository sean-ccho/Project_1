#!/usr/bin/env python3
"""전략 기준선 비교: SPY 보유 vs 단순 추세 타이밍 (개발 구간 2016-01 ~ 2025-09).

질문: 종목 선택 엣지가 없다면, "가만히 SPY" 위에 단순 시장 타이밍이 위험 대비 나은가?
  B0 SPY 보유 (기준)
  T1 SPY 종가 > EMA200 이면 보유, 아니면 현금 — 스크리너의 약세장(SPY<EMA200) 정의와 같다
  T2 SPY 종가 > SMA200 이면 보유, 아니면 현금 — 고전적 200일선 규칙
신호는 t 종가, 전환은 t+1 시가 (시가→시가 수익률). 전환 시 편도 0.1%. 현금 수익 0 (보수적 — 실제론 단기채 이자).
판정 (사전 등록, 시도 2건): T 의 SPY 대비 ΔSharpe 95% 하한 > 0 (paired_bootstrap, 20일 블록) → 채택 후보.
  MDD 개선은 따로 보고한다 (채택 근거 아님).

사용법: PYTHONPATH=.:src python scripts/baseline_compare.py
산출: tier3/BASELINE_COMPARE.md
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from ta.trend import EMAIndicator

from paper_trading.evaluation import cagr, max_drawdown, paired_bootstrap, sharpe, verdict
from research_utils import COST_PER_SIDE, RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, load_ohlcv, log_trials, total_trials

START = pd.Timestamp("2016-01-01")
FAMILY = "baseline_timing"


def timing_returns(open_: pd.Series, close: pd.Series, signal: pd.Series, cost: float = COST_PER_SIDE) -> pd.Series:
    """signal(t 종가 기준 True=보유) → t+1 시가부터 반영한 일간 시가→시가 수익률 (전환 비용 차감)."""
    r_oo = open_.shift(-1) / open_ - 1.0  # t 시가 → t+1 시가 (t에 기록)
    w = signal.astype(float).shift(1).fillna(0.0)  # t-1 종가 신호 → t 시가부터 보유
    turn = w.diff().abs().fillna(w.abs())
    return (w * r_oo - turn * cost).dropna()


def _registered() -> bool:
    if not TRIAL_LOG.exists():
        return False
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    return bool(((log["family"] == FAMILY) & (log["판정"] == "등록")).any())


def main() -> None:
    if not _registered():
        log_trials("stage9", FAMILY, 2, [{"test_id": "(등록)", "판정": "등록",
                                          "메모": "SPY vs SPY>EMA200 · SPY>SMA200 (t 종가 신호, t+1 시가 전환, 편도 0.1%, 현금 0). "
                                                  "2016-01~2025-09. 채택 = ΔSharpe 95% 하한 > 0"}])
    spy = load_ohlcv(RESEARCH_DIR / "etf_ohlcv.parquet")["SPY"]
    o, c = spy["Open"].dropna(), spy["Close"].dropna()
    ema200 = EMAIndicator(c, window=200).ema_indicator()
    sma200 = c.rolling(200).mean()
    series = {
        "B0 SPY 보유": timing_returns(o, c, pd.Series(True, index=c.index)),
        "T1 SPY>EMA200": timing_returns(o, c, c > ema200),
        "T2 SPY>SMA200": timing_returns(o, c, c > sma200),
    }
    df = pd.DataFrame(series).loc[START:].dropna()
    b0 = df["B0 SPY 보유"].tolist()

    rows, tlog = [], []
    for name in df.columns:
        r = df[name].tolist()
        row = {"전략": name, "CAGR": cagr(r), "Sharpe": sharpe(r), "MDD": max_drawdown(r),
               "보유비율": float((df[name] != 0).mean())}
        if name != "B0 SPY 보유":
            st = paired_bootstrap(r, b0)
            row.update({"ΔSharpe": st["ΔSharpe"], "95%": f"[{st['ΔSharpe_하한']:+.2f}, {st['ΔSharpe_상한']:+.2f}]",
                        "판정": verdict(st), "구간개선": f"{st['구간_개선']}/{st['구간_수']}"})
            tlog.append({"test_id": name, "t": "", "판정": row["판정"],
                         "메모": f"ΔSharpe {st['ΔSharpe']:+.2f} {row['95%']}, MDD {row['MDD']:.1%}"})
        rows.append(row)
    log_trials("stage9", FAMILY, 2, tlog)

    yr = (1 + df).groupby(df.index.year).prod() - 1
    lines = ["# 전략 기준선 비교 — SPY 보유 vs 단순 추세 타이밍", "",
             f"- 구간 {df.index.min().date()} ~ {df.index.max().date()} ({len(df)}일, 개발 구간). 시가→시가, 전환 편도 {COST_PER_SIDE:.1%}, 현금 수익 0",
             f"- 시도 2건 (사전 등록) · 누적 시도 {total_trials()}건. 채택 = SPY 대비 ΔSharpe 95% 하한 > 0", "",
             "| 전략 | CAGR | Sharpe | MDD | 보유 비율 | ΔSharpe vs SPY | 95% 구간 | 구간 개선 | 판정 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['전략']} | {r['CAGR']:+.1%} | {r['Sharpe']:.2f} | {r['MDD']:.1%} | {r['보유비율']:.0%} | "
                     f"{r.get('ΔSharpe', float('nan')):+.2f} | {r.get('95%', '—')} | {r.get('구간개선', '—')} | {r.get('판정', '기준')} |")
    lines += ["", "## 연도별 수익", "", "| 연도 | " + " | ".join(df.columns) + " |", "|---" * (len(df.columns) + 1) + "|"]
    lines += [f"| {y} | " + " | ".join(f"{v:+.1%}" for v in row) + " |" for y, row in yr.iterrows()]
    (TIER3_DIR / "BASELINE_COMPARE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
