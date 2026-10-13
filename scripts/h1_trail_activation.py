#!/usr/bin/env python3
"""H1 트레일링 시작 조건 (h1_trail_activate_v1): 고점이 매수가보다 +3% / +5% 오른 뒤에만 트레일링을 켜면 PT-1 이 나아지나.

계기: 실거래 PT-1 청산 52건 중 트레일링 매도 18건이 평균 −4.0% (2026-10-12 메일 '판 이유' 집계).
지금 규칙은 고점이 매수가를 조금이라도 넘으면 트레일링이 켜져, +1% 올랐다가 고점 대비 −5~6% 빠지면 손실로 판다.

사전 등록 (TRIAL_LOG family h1_trail_activate_v1, 2건 — 결과 계산 전에 --register):
  H1a  EXIT_PARAMS[모멘텀·바닥반등]["trail_activate_pct"] = 0.03  (run_hypothesis_ab.VARIANTS 와 같은 정의)
  H1b  같은 값 0.05
  나머지는 전부 지금 규칙 그대로. 기간·설정은 PT-1 기준선과 같다 (validate_upside_walkforward.RUN_KW:
  2016-01-04 ~ 2025-09-30, 13y PIT S&P 500, 매일 리밸런스, $5,000, 디스크 가격 캐시 고정).
  홀드아웃(2025-10-01 ~)은 이미 한 번 썼으므로 쓰지 않는다.
  기준 = 같은 실행에서 다시 돌린 BASE. 저장된 pt1_base_dev_equity 와 같아야 한다 (다르면 결과를 쓰지 않고 중단).
판정:
  채택 후보 = BASE 대비 verdict '채택 후보' (ΔSharpe 95% 하한 > 0 · 3구간 중 2개 이상 개선 · MDD 2%p 이내)
              · p_개선아님 ≤ 0.05/2
  관찰     = 그 밖에 BASE 대비 ΔSharpe 95% 하한 > 0
  참고     = SPY(종가) 대비 ΔSharpe, 거래 수·청산 이유 분포 (판정에 안 씀)
한계: 개발 구간은 지금까지 940건 시도에 쓰였고, 구조 워크포워드(2026-10-09)에도 '트레일링 시작'이 손잡이로 들어 있었다
     (그때는 8개 손잡이를 함께 골라 불통과). 상장폐지 종목 가격이 없는 생존 편향은 기준선과 같다.

사용법:
  PYTHONPATH=.:src python scripts/h1_trail_activation.py --register
  PYTHONPATH=.:src python scripts/h1_trail_activation.py --run
산출: tier3/H1_TRAIL_ACTIVATION.md · tier3/h1_trail_activation.csv · data/research/h1_{BASE,H1a,H1b}_dev_equity.parquet (git 제외)
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
from paper_trading.evaluation import cagr, max_drawdown, paired_bootstrap, sharpe, verdict
from research_utils import RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, load_ohlcv, log_trials, total_trials
from run_hypothesis_ab import VARIANTS
from validate_upside_walkforward import _ORIGINAL_FETCH, RUN_KW, frozen_fetch

FAMILY = "h1_trail_activate_v1"
STEP = "stage11"
TESTS = ("H1a", "H1b")
N_TESTS = len(TESTS)
P_BONF = 0.05 / N_TESTS
BASE_SAVED = RESEARCH_DIR / "pt1_base_dev_equity.parquet"
ETF_OHLCV = RESEARCH_DIR / "etf_ohlcv.parquet"
OUT_CSV = TIER3_DIR / "h1_trail_activation.csv"
REPORT = TIER3_DIR / "H1_TRAIL_ACTIVATION.md"
REASONS = ("목표가", "시간익절", "트레일링", "손절", "장기보유", "교체")
SPEC = (f"[{RUN_KW['start_date']}~{RUN_KW['end_date']}, PT-1 백테스트 엔진, 13y PIT S&P 500, 매일 리밸런스, $5,000, "
        "가격 캐시 고정] H1a/H1b = EXIT_PARAMS[모멘텀·바닥반등].trail_activate_pct 0.03/0.05 (나머지 지금 규칙). "
        "기준 = 같은 실행의 BASE (저장된 pt1_base_dev_equity 와 같아야 함). "
        "채택 = BASE 대비 verdict 채택 후보(ΔSharpe 95% 하한>0 · 3구간 중 2+ · MDD 2%p 이내) · p_개선아님 ≤ 0.05/2. "
        "SPY 대비·청산 이유 분포는 참고.")


def judge(stats: dict[str, float], v: str) -> str:
    """사전 등록 판정: 채택 후보 / 관찰 / 탈락."""
    if v == "채택 후보" and stats and stats["p_개선아님"] <= P_BONF:
        return "채택 후보"
    if stats and stats["ΔSharpe_하한"] > 0:
        return "관찰"
    return "탈락"


def exit_mix(trades: list[dict[str, Any]]) -> dict[str, tuple[int, float]]:
    """청산 이유별 (건수, 평균 수익률). 이유 문자열 앞부분으로 묶는다 (예: '트레일링(-4.1%)' → 트레일링)."""
    groups: dict[str, list[float]] = {}
    for t in trades:
        reason = str(t.get("exit_reason") or "")
        key = next((k for k in REASONS if reason.startswith(k)), "기타")
        groups.setdefault(key, []).append(float(t.get("return_pct") or 0.0))
    return {k: (len(v), float(np.mean(v))) for k, v in groups.items()}


def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame(columns=["family", "판정"])


def register() -> None:
    log = _log()
    if ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        print(f"{FAMILY} 는 이미 등록됨 — 다시 등록하지 않는다")
        return
    log_trials(STEP, FAMILY, N_TESTS, [{"test_id": "(등록)", "판정": "등록", "메모": SPEC}])
    print(f"등록: {FAMILY} {N_TESTS}건 · p ≤ {P_BONF:.3f} · 누적 시도 {total_trials()}건")


def _backtest(key: str) -> dict[str, Any]:
    label, overrides = VARIANTS[key]
    print(f"===== {key} {label} ({RUN_KW['start_date']} ~ {RUN_KW['end_date']})", flush=True)
    bt.fetch_ohlcv = frozen_fetch
    try:
        with config_overrides(overrides):
            result = bt.run_paper_trading_backtest(**RUN_KW)
    finally:
        bt.fetch_ohlcv = _ORIGINAL_FETCH
    eq = result["equity_curve"].dropna()
    eq.index = pd.to_datetime(eq.index)
    eq.rename("equity").to_frame().to_parquet(RESEARCH_DIR / f"h1_{key}_dev_equity.parquet")
    s = result["summary"]
    print(f"[h1] {key}: Sharpe(일간) {s.get('Sharpe_일간')} · CAGR {s.get('CAGR')} · MDD {s.get('MDD')} · "
          f"거래 {s.get('총거래수')}", flush=True)
    return {"equity": eq, "summary": s, "trades": result["trades"]}


def _yearly(eq: pd.Series) -> pd.Series:
    return eq.groupby(eq.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)  # pt1_baseline_equity 와 같은 방식


def run() -> None:
    log = _log()
    if not ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        raise SystemExit("먼저 --register 로 등록한다 (결과 계산 전 사전 등록)")
    already_logged = ((log["family"] == FAMILY) & (log["판정"] != "등록")).any()

    runs = {key: _backtest(key) for key in ("BASE", *TESTS)}

    print("===== 기준선 재현 확인", flush=True)
    saved = pd.read_parquet(BASE_SAVED).iloc[:, 0]
    base_eq = runs["BASE"]["equity"]
    common = saved.index.intersection(base_eq.index)
    max_rel = float((base_eq.loc[common] / saved.loc[common] - 1).abs().max()) if len(common) else float("inf")
    if len(common) != len(saved) or max_rel > 1e-9:
        raise SystemExit(f"BASE 가 저장된 기준선과 다르다 (겹친 날 {len(common)}/{len(saved)}, 최대 차이 {max_rel:.2e}) "
                         "— 코드·캐시를 확인하기 전에는 결과를 쓰지 않는다")
    print(f"[h1] BASE = 저장된 기준선 ({len(common)}일, 최대 차이 {max_rel:.1e})", flush=True)

    print("===== 비교·판정", flush=True)
    etf = load_ohlcv(ETF_OHLCV)
    spy_r = etf["SPY"]["Close"].pct_change()
    base_r = base_eq.pct_change().dropna()
    rows = []
    for key in ("BASE", *TESTS):
        r = runs[key]["equity"].pct_change().dropna()
        df = pd.concat({"v": r, "b": base_r, "spy": spy_r.reindex(r.index)}, axis=1).dropna()
        a, b, sp = df["v"].tolist(), df["b"].tolist(), df["spy"].tolist()
        st = paired_bootstrap(a, b) if key != "BASE" else {}
        v = verdict(st, max_drawdown(a), max_drawdown(b)) if st else ""
        st_spy = paired_bootstrap(a, sp)
        mix = exit_mix(runs[key]["trades"])
        trades = runs[key]["trades"]
        rows.append({
            "test_id": key, "이름": VARIANTS[key][0], "시작": df.index.min().date(), "끝": df.index.max().date(),
            "일수": len(df), "Sharpe": sharpe(a), "CAGR": cagr(a), "MDD": max_drawdown(a),
            "거래수": len(trades),
            "승률": float(np.mean([float(t.get("return_pct") or 0) > 0 for t in trades])) if trades else np.nan,
            "ΔSharpe_BASE": st.get("ΔSharpe"), "하한_BASE": st.get("ΔSharpe_하한"), "상한_BASE": st.get("ΔSharpe_상한"),
            "p_BASE": st.get("p_개선아님"), "구간_BASE": f"{st.get('구간_개선')}/{st.get('구간_수')}" if st else "",
            "verdict_BASE": v,
            "SPY_Sharpe": sharpe(sp), "ΔSharpe_SPY": st_spy.get("ΔSharpe"), "하한_SPY": st_spy.get("ΔSharpe_하한"),
            "상한_SPY": st_spy.get("ΔSharpe_상한"),
            "청산이유": " · ".join(f"{k} {n}({m:+.1%})" for k, (n, m) in mix.items()),
            "판정": judge(st, v) if key != "BASE" else "기준",
        })
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    if already_logged:
        print("결과 행이 이미 TRIAL_LOG 에 있다 — 다시 쓰지 않는다")
    else:
        log_trials(STEP, FAMILY, N_TESTS, [{
            "test_id": r["test_id"], "판정": r["판정"],
            "메모": (f"Sharpe {r['Sharpe']:.2f} vs BASE {res.iloc[0]['Sharpe']:.2f}, ΔSharpe {r['ΔSharpe_BASE']:+.2f} "
                     f"[{r['하한_BASE']:+.2f}, {r['상한_BASE']:+.2f}] p {r['p_BASE']:.3f}, 구간 {r['구간_BASE']}, "
                     f"CAGR {r['CAGR']:+.1%}, MDD {r['MDD']:.1%}, 거래 {r['거래수']}")}
            for _, r in res.iterrows() if r["test_id"] != "BASE"])

    write_report(res, {k: runs[k]["equity"] for k in runs}, etf["SPY"]["Close"])


def _f(v: Any, fmt: str) -> str:
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)


def write_report(res: pd.DataFrame, eqs: dict[str, pd.Series], spy_close: pd.Series) -> None:
    n_fail = int((res["판정"] == "탈락").sum())
    n_adopt = int((res["판정"] == "채택 후보").sum())
    lines = [
        f"# H1 트레일링 시작 조건 ({FAMILY}) — {RUN_KW['start_date']} ~ {RUN_KW['end_date']}",
        "",
        f"- 사전 등록: TRIAL_LOG family `{FAMILY}`, {N_TESTS}건 · p ≤ 0.05/{N_TESTS} · 누적 시도 {total_trials()}건",
        "- 계기: 실거래 PT-1 트레일링 매도 18건 평균 −4.0% (매수가를 조금만 넘어도 트레일링이 켜짐)",
        "- 바꾼 것: 고점이 매수가보다 +3%(H1a) / +5%(H1b) 오른 뒤에만 트레일링 시작. 나머지는 지금 규칙",
        "- 기준 = 같은 실행의 BASE (저장된 PT-1 기준선과 같은지 확인함)",
        "- 채택 후보 = BASE 대비 ΔSharpe 95% 하한 > 0 · 3구간 중 2개 이상 개선 · MDD 2%p 이내 · p ≤ 0.025",
        f"- **결과: 채택 후보 {n_adopt}건 · 탈락 {n_fail}건 · 관찰 {N_TESTS - n_adopt - n_fail}건**",
        "",
        "| 변형 | Sharpe | CAGR | MDD | 거래 | 승률 | BASE 대비 ΔSharpe [95%] | p | 구간 | SPY 대비 ΔSharpe [95%] | 판정 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    spy_sh = res.iloc[0]["SPY_Sharpe"]
    lines.append(f"| SPY 보유 (종가) | {_f(spy_sh, '.2f')} | | | | | | | | | 참고 |")
    for _, r in res.iterrows():
        base_cell = ("" if r["test_id"] == "BASE" else
                     f"{_f(r['ΔSharpe_BASE'], '+.2f')} [{_f(r['하한_BASE'], '+.2f')}, {_f(r['상한_BASE'], '+.2f')}]")
        lines.append(
            f"| {r['test_id']} {r['이름']} | {_f(r['Sharpe'], '.2f')} | {_f(r['CAGR'], '+.1%')} | {_f(r['MDD'], '.1%')} | "
            f"{r['거래수']} | {_f(r['승률'], '.0%')} | {base_cell} | {_f(r['p_BASE'], '.3f')} | {r['구간_BASE']} | "
            f"{_f(r['ΔSharpe_SPY'], '+.2f')} [{_f(r['하한_SPY'], '+.2f')}, {_f(r['상한_SPY'], '+.2f')}] | **{r['판정']}** |")
    lines += ["", "## 청산 이유 (건수, 평균 수익률)", "", "| 변형 | 청산 이유 |", "|---|---|"]
    lines += [f"| {r['test_id']} | {r['청산이유']} |" for _, r in res.iterrows()]
    yr = pd.DataFrame({k: _yearly(v) for k, v in eqs.items()})
    spy = spy_close.loc[eqs["BASE"].index.min():eqs["BASE"].index.max()]
    yr["SPY"] = _yearly(spy)
    lines += ["", "## 연도별 수익", "", "| 연도 | " + " | ".join(yr.columns) + " |", "|---" * (len(yr.columns) + 1) + "|"]
    lines += [f"| {y} | " + " | ".join(_f(v, "+.1%") for v in row) + " |" for y, row in yr.iterrows()]
    lines += ["", "## 한계", "",
              "- 개발 구간은 지금까지 많은 시도에 쓰였다. 구조 워크포워드(2026-10-09)에서도 '트레일링 시작'을 다른 손잡이와 함께 골랐고 불통과였다.",
              "- 홀드아웃(2025-10-01 ~)은 이미 써서 표본 외 확인이 없다. 채택 후보가 나와도 실거래 반영은 사용자 결정이다.",
              "- 상장폐지 종목 가격이 없는 생존 편향은 기준선과 같다.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"리포트: {REPORT}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--register", action="store_true", help="결과 계산 전 사전 등록 (TRIAL_LOG)")
    g.add_argument("--run", action="store_true", help="BASE·H1a·H1b 백테스트 → 비교·판정·리포트")
    args = p.parse_args()
    register() if args.register else run()


if __name__ == "__main__":
    main()
