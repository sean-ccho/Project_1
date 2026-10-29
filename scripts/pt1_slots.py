#!/usr/bin/env python3
"""PT-1 종목 수 늘리기 (pt1_slots_v1): 3종목 대신 5·10·20종목을 들면 PT-1 이 SPY 를 따라잡나.

계기: 2026-10-20 사용자 — "3개말고 더 많이 들어야 되나?" (factor_long 보고서: 3종목이라 종목 고유 위험이 연 15~17% 더해져
SPY 와 같은 Sharpe 를 내려면 연 6% 쯤의 알파가 필요하다).

사전 등록 (TRIAL_LOG family pt1_slots_v1, 3건 — 결과 계산 전에 --register):
  공통: PT1_SIZING = "equal_fixed" (사이징 수정판, 실거래 $5,000 계좌와 같은 회계). 나머지는 전부 지금 PT-1 규칙.
  N3   기준 = PAPER_TRADING_MAX_POSITIONS 3 · 하루 최대 매수 1 → 저장된 sizing_FIX_dev_equity 와 같아야 한다 (다르면 중단)
  N5   5종목 · 하루 최대 매수 2 (wfo_structure.DAILY_BUY 와 같음)
  N10  10종목 · 하루 최대 매수 3 (wfo_structure.DAILY_BUY 와 같음)
  N20  20종목 · 하루 최대 매수 5 (같은 비율로 늘림)
  기간·설정은 PT-1 기준선과 같다 (validate_upside_walkforward.RUN_KW: 2016-01-04 ~ 2025-09-30, 13y PIT S&P 500,
  매일 리밸런스, $5,000, 디스크 가격 캐시 고정). 홀드아웃(2025-10-01 ~)은 쓰지 않는다.
판정 (paired_bootstrap 10,000회, 20일 블록, p ≤ 0.05/3):
  SPY 이김   = SPY(종가) 대비 ΔSharpe 95% 하한 > 0 · p_개선아님 ≤ 0.05/3
  개선       = N3 대비 ΔSharpe 95% 하한 > 0 · p_개선아님 ≤ 0.05/3 (더 많이 들면 나아지지만 SPY 는 못 이김)
  변화 없음  = 그 밖
  참고: 평균 보유 종목 수, CAGR, MDD, 거래 수, 연도별 수익.
한계: 개발 구간은 지금까지 많은 시도에 쓰였다. 상장폐지 종목 가격이 없는 생존 편향은 기준선과 같다.

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/pt1_slots.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/pt1_slots.py --run
산출: tier3/PT1_SLOTS.md · tier3/pt1_slots.csv · data/research/slots_{N3,N5,N10,N20}_dev_equity.parquet (git 제외)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

import paper_trading.backtest as bt
from paper_trading.config_override import config_overrides
from paper_trading.evaluation import cagr, max_drawdown, paired_bootstrap, sharpe
from research_utils import RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, load_ohlcv, log_trials, total_trials
from validate_upside_walkforward import _ORIGINAL_FETCH, RUN_KW, frozen_fetch

FAMILY = "pt1_slots_v1"
STEP = "pt1_slots"
N_TESTS = 3
N_BOOT = 10_000
VARIANTS: dict[str, tuple[int, int]] = {"N3": (3, 1), "N5": (5, 2), "N10": (10, 3), "N20": (20, 5)}  # 종목 수, 하루 매수
FIX_SAVED = RESEARCH_DIR / "sizing_FIX_dev_equity.parquet"
ETF_OHLCV = RESEARCH_DIR / "etf_ohlcv.parquet"
OUT_CSV = TIER3_DIR / "pt1_slots.csv"
REPORT = TIER3_DIR / "PT1_SLOTS.md"
SPEC = (f"[{RUN_KW['start_date']}~{RUN_KW['end_date']}, PT-1 백테스트 엔진, 13y PIT S&P 500, 매일 리밸런스, $5,000, "
        "가격 캐시 고정, PT1_SIZING equal_fixed] N5(하루 2)·N10(3)·N20(5) vs N3(1, 저장된 sizing_FIX 와 같아야 함). "
        "SPY 이김 = SPY 대비 ΔSharpe 하한>0·p≤0.05/3, 개선 = N3 대비 하한>0·p≤0.05/3 (paired_bootstrap 1만·20일 블록).")


def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame(columns=["family", "판정"])


def register() -> None:
    log = _log()
    if ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        print(f"{FAMILY} 는 이미 등록됨 — 다시 등록하지 않는다")
        return
    log_trials(STEP, FAMILY, N_TESTS, [{"test_id": "(등록)", "판정": "등록", "메모": SPEC}])
    print(f"등록: {FAMILY} {N_TESTS}건 · 누적 시도 {total_trials()}건")


def _backtest(key: str) -> dict[str, Any]:
    n, daily = VARIANTS[key]
    overrides = {"PT1_SIZING": "equal_fixed", "PAPER_TRADING_MAX_POSITIONS": n, "PAPER_TRADING_MAX_DAILY_BUY": daily}
    print(f"===== {key}: {n}종목 · 하루 최대 매수 {daily} ({RUN_KW['start_date']} ~ {RUN_KW['end_date']})", flush=True)
    bt.fetch_ohlcv = frozen_fetch
    try:
        with config_overrides(overrides):
            result = bt.run_paper_trading_backtest(**RUN_KW)
    finally:
        bt.fetch_ohlcv = _ORIGINAL_FETCH
    eq = result["equity_curve"].dropna()
    eq.index = pd.to_datetime(eq.index)
    eq.rename("equity").to_frame().to_parquet(RESEARCH_DIR / f"slots_{key}_dev_equity.parquet")
    s = result["summary"]
    print(f"[slots] {key}: Sharpe(일간) {s.get('Sharpe_일간')} · CAGR {s.get('CAGR')} · MDD {s.get('MDD')} · "
          f"거래 {s.get('총거래수')}", flush=True)
    return {"equity": eq, "summary": s, "trades": result["trades"]}


def avg_holdings(trades: list[dict[str, Any]], index: pd.DatetimeIndex) -> float:
    """하루 평균 보유 종목 수 (0주 매수 제외, 진입일 ≤ d < 청산일)."""
    counts = pd.Series(0, index=index)
    for t in trades:
        if not float(t.get("shares") or 0.0):
            continue
        start, end = pd.Timestamp(t["entry_date"]), pd.Timestamp(t.get("exit_date") or index[-1] + pd.Timedelta(days=1))
        counts[(counts.index >= start) & (counts.index < end)] += 1
    return float(counts.mean())


def judge(st_spy: dict[str, float], st_n3: dict[str, float]) -> str:
    alpha = 0.05 / N_TESTS
    if st_spy and st_spy["ΔSharpe_하한"] > 0 and st_spy["p_개선아님"] <= alpha:
        return "SPY 이김"
    if st_n3 and st_n3["ΔSharpe_하한"] > 0 and st_n3["p_개선아님"] <= alpha:
        return "개선"
    return "변화 없음"


def _yearly(eq: pd.Series) -> pd.Series:
    return eq.groupby(eq.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)  # pt1_baseline_equity 와 같은 방식


def run() -> None:
    log = _log()
    if not ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        raise SystemExit("먼저 --register 로 등록한다 (결과 계산 전 사전 등록)")
    if ((log["family"] == FAMILY) & (log["판정"] != "등록")).any():
        raise SystemExit(f"{FAMILY} 결과가 이미 기록됐다")

    runs = {"N3": _backtest("N3")}
    saved = pd.read_parquet(FIX_SAVED).iloc[:, 0]
    common = saved.index.intersection(runs["N3"]["equity"].index)
    max_rel = float((runs["N3"]["equity"].loc[common] / saved.loc[common] - 1).abs().max()) if len(common) else np.inf
    if len(common) != len(saved) or max_rel > 1e-9:
        raise SystemExit(f"N3 가 저장된 sizing_FIX 와 다르다 (겹친 날 {len(common)}/{len(saved)}, 최대 차이 {max_rel:.2e}) "
                         "— 코드·캐시를 확인하기 전에는 결과를 쓰지 않는다")
    print(f"[slots] N3 = 저장된 sizing_FIX ({len(common)}일, 최대 차이 {max_rel:.1e})", flush=True)
    for key in ("N5", "N10", "N20"):
        runs[key] = _backtest(key)

    spy_r = load_ohlcv(ETF_OHLCV)["SPY"]["Close"].pct_change()
    n3_r = runs["N3"]["equity"].pct_change().dropna()
    rows, yearly = [], {}
    for key, rr in runs.items():
        r = rr["equity"].pct_change().dropna()
        df = pd.concat({"v": r, "n3": n3_r.reindex(r.index), "spy": spy_r.reindex(r.index)}, axis=1).dropna()
        st_spy = paired_bootstrap(df["v"].tolist(), df["spy"].tolist(), n_boot=N_BOOT)
        st_n3 = paired_bootstrap(df["v"].tolist(), df["n3"].tolist(), n_boot=N_BOOT) if key != "N3" else {}
        trades = [t for t in rr["trades"] if float(t.get("shares") or 0.0)]
        rows.append({
            "variant": key, "종목수": VARIANTS[key][0], "하루매수": VARIANTS[key][1],
            "Sharpe": sharpe(df["v"].tolist()), "CAGR": cagr(df["v"].tolist()), "MDD": max_drawdown(df["v"].tolist()),
            "vol": float(df["v"].std() * np.sqrt(252)), "거래": len(trades),
            "평균보유": avg_holdings(rr["trades"], rr["equity"].index),
            "SPY_Sharpe": sharpe(df["spy"].tolist()),
            "ΔSPY": st_spy.get("ΔSharpe"), "ΔSPY_하한": st_spy.get("ΔSharpe_하한"),
            "ΔSPY_상한": st_spy.get("ΔSharpe_상한"), "p_SPY": st_spy.get("p_개선아님"),
            "ΔN3": st_n3.get("ΔSharpe"), "ΔN3_하한": st_n3.get("ΔSharpe_하한"), "ΔN3_상한": st_n3.get("ΔSharpe_상한"),
            "p_N3": st_n3.get("p_개선아님"),
            "판정": "기준" if key == "N3" else judge(st_spy, st_n3)})
        yearly[key] = _yearly(rr["equity"])
    yearly["SPY"] = _yearly(load_ohlcv(ETF_OHLCV)["SPY"]["Close"].loc[runs["N3"]["equity"].index.min():
                                                                    runs["N3"]["equity"].index.max()])
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log_trials(STEP, FAMILY, N_TESTS, [
        {"test_id": r["variant"], "t": f"{r['ΔN3']:.3f}", "판정": r["판정"],
         "메모": (f"Sharpe {r['Sharpe']:.2f} · ΔN3 {r['ΔN3']:+.2f} [{r['ΔN3_하한']:+.2f}, {r['ΔN3_상한']:+.2f}] p {r['p_N3']:.4f}"
                f" · ΔSPY {r['ΔSPY']:+.2f} [{r['ΔSPY_하한']:+.2f}, {r['ΔSPY_상한']:+.2f}] p {r['p_SPY']:.4f}")}
        for _, r in res.iterrows() if r["variant"] != "N3"])
    write_report(res, pd.DataFrame(yearly))
    print(f"[slots] 누적 시도 {total_trials()}", flush=True)


def _f(v: Any, fmt: str) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, fmt)


def write_report(res: pd.DataFrame, yearly: pd.DataFrame) -> None:
    lines = [
        "# PT-1 종목 수 늘리기 — 3 · 5 · 10 · 20종목 (pt1_slots_v1)", "",
        f"- 실행 {pd.Timestamp.now():%Y-%m-%d %H:%M} · 정의·사전 등록은 `scripts/pt1_slots.py` 머리말 · 누적 시도 {total_trials()}건",
        f"- {RUN_KW['start_date']} ~ {RUN_KW['end_date']} · PT-1 백테스트 엔진 · 그날 S&P 500 (PIT) · $5,000 · 사이징 수정판(equal_fixed)",
        "- Sharpe 는 무위험을 빼지 않은 일간 기준 (지금까지 PT-1 보고서와 같은 방식). ΔSharpe = paired_bootstrap 1만 회, 20일 블록.", "",
        "| 종목 수 | 하루 최대 매수 | 평균 보유 | Sharpe | 변동성 | CAGR | MDD | 거래 | N3 대비 ΔSharpe [95%] · p | "
        "SPY 대비 ΔSharpe [95%] · p | 판정 |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        n3 = "" if r["variant"] == "N3" else (f"{r['ΔN3']:+.2f} [{r['ΔN3_하한']:+.2f}, {r['ΔN3_상한']:+.2f}] · "
                                              f"{r['p_N3']:.4f}")
        lines.append(f"| {r['종목수']} | {r['하루매수']} | {r['평균보유']:.1f} | {r['Sharpe']:.2f} | {r['vol']:.1%} | "
                     f"{r['CAGR']:.1%} | {r['MDD']:.1%} | {r['거래']} | {n3} | {r['ΔSPY']:+.2f} [{r['ΔSPY_하한']:+.2f}, "
                     f"{r['ΔSPY_상한']:+.2f}] · {r['p_SPY']:.4f} | **{r['판정']}** |")
    lines += [f"| SPY 보유 (종가) | | | {res['SPY_Sharpe'].iloc[0]:.2f} | | | | | | | 참고 |", "",
              "## 연도별 수익", "", "| 연도 | " + " | ".join(yearly.columns) + " |",
              "|---|" + "---|" * len(yearly.columns)]
    lines += [f"| {y} | " + " | ".join(_f(v, "+.1%") for v in row) + " |" for y, row in yearly.iterrows()]
    lines += ["", "## 한계", "",
              "- 개발 구간(2016~2025)은 지금까지 많은 시도에 쓰였다. 상장폐지 종목 가격이 없는 생존 편향은 기준선과 같다.",
              "- 종목 수 말고는 전부 지금 PT-1 규칙이다 (후보 기준·손절·트레일링). 후보가 모자라면 빈자리는 현금으로 남는다."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--register", action="store_true")
    g.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.register:
        register()
    else:
        run()


if __name__ == "__main__":
    main()
