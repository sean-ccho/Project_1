#!/usr/bin/env python3
"""PT-1 매수 비중 버그 수정 (pt1_sizing_fix_v1): 약세장 한도가 풀릴 때 생기는 0주 매수를 없애면 PT-1 이 어떻게 달라지나.

계기 (2026-10-13 발견, 메모 pt1-zero-share-quirk): 백테스트는 새 종목에 '남은 현금 ÷ 빈자리'를 쓰는데,
약세장(최대 2종목)에서는 두 번째 매수가 남은 현금을 전부 쓴다. 레짐이 풀려 자리가 3개가 되면 세 번째 매수는 $0 → 0주.
BASE 616건 중 41건이 0주, 19건은 자산 90% 넘게 한 종목에 들어갔다. '3종목 분산'이 실제로는 2종목 몰빵인 날이 많았다.

사전 등록 (TRIAL_LOG family pt1_sizing_fix_v1, 1건 — 결과 계산 전에 --register):
  FIX  PT1_SIZING = "equal_fixed": 빈자리를 약세장 한도 대신 PAPER_TRADING_MAX_POSITIONS(3) 기준으로 센다.
       약세장 한도는 '몇 종목'만 줄이고 한 종목 크기는 그대로 (약세장 2종목이면 현금 약 1/3 이 남는다).
       실거래 $5,000 계좌(pt1_account.allocate)가 이미 이 회계를 쓴다. 나머지는 전부 지금 규칙.
  기간·설정은 PT-1 기준선과 같다 (validate_upside_walkforward.RUN_KW: 2016-01-04 ~ 2025-09-30, 13y PIT S&P 500,
  매일 리밸런스, $5,000, 디스크 가격 캐시 고정). 홀드아웃(2025-10-01 ~)은 이미 한 번 썼으므로 쓰지 않는다.
  기준 = 같은 실행에서 다시 돌린 BASE. 저장된 pt1_base_dev_equity 와 같아야 한다 (다르면 결과를 쓰지 않고 중단).
판정 (버그 수정이라 성과 개선이 아니라 '해가 없는가'를 본다):
  채택     = BASE 대비 ΔSharpe 95% 상한 > 0 (유의하게 나빠지지 않음) · 0주 매수 0건
  개선     = 채택 중 ΔSharpe 95% 하한 > 0 · p_개선아님 ≤ 0.05 (참고 — 엣지가 생긴다는 뜻은 아니다)
  탈락     = ΔSharpe 95% 상한 ≤ 0 (유의하게 나빠짐) 또는 0주 매수가 남음
  참고     = SPY(종가) 대비 ΔSharpe, 매수 비중 분포(자산 대비), 연도별 수익 (판정에 안 씀)
한계: 개발 구간은 지금까지 많은 시도에 쓰였다. 상장폐지 종목 가격이 없는 생존 편향은 기준선과 같다.
     실거래 거래로그는 거래별 % 라 바뀌지 않고, $5,000 계좌는 이미 이 회계다 → 바뀌는 것은 연구 기준선이다.

사용법:
  PYTHONPATH=.:src python scripts/pt1_sizing_fix.py --register
  PYTHONPATH=.:src python scripts/pt1_sizing_fix.py --run
산출: tier3/PT1_SIZING_FIX.md · tier3/pt1_sizing_fix.csv · data/research/sizing_{BASE,FIX}_dev_equity.parquet (git 제외)
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

FAMILY = "pt1_sizing_fix_v1"
STEP = "stage19"
N_TESTS = 1
VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    "BASE": ("기준선 (남은 현금 ÷ 약세장 한도 기준 빈자리)", {}),
    "FIX": ("빈자리를 항상 3자리 기준으로 (equal_fixed)", {"PT1_SIZING": "equal_fixed"}),
}
BASE_SAVED = RESEARCH_DIR / "pt1_base_dev_equity.parquet"
ETF_OHLCV = RESEARCH_DIR / "etf_ohlcv.parquet"
OUT_CSV = TIER3_DIR / "pt1_sizing_fix.csv"
REPORT = TIER3_DIR / "PT1_SIZING_FIX.md"
SPEC = (f"[{RUN_KW['start_date']}~{RUN_KW['end_date']}, PT-1 백테스트 엔진, 13y PIT S&P 500, 매일 리밸런스, $5,000, "
        "가격 캐시 고정] FIX = PT1_SIZING equal_fixed (빈자리를 약세장 한도 대신 PAPER_TRADING_MAX_POSITIONS 기준, "
        "실거래 $5,000 계좌와 같은 회계). 기준 = 같은 실행의 BASE (저장된 pt1_base_dev_equity 와 같아야 함). "
        "버그 수정이라 채택 = ΔSharpe 95% 상한>0 (유의하게 나빠지지 않음) · 0주 매수 0건. "
        "개선 = 채택 중 하한>0 · p≤0.05 (참고). SPY 대비·매수 비중 분포·연도별은 참고.")


def judge(stats: dict[str, float], zero_share: int) -> str:
    """사전 등록 판정: 채택 / 개선 / 탈락."""
    if zero_share > 0 or stats["ΔSharpe_상한"] <= 0:
        return "탈락"
    if stats["ΔSharpe_하한"] > 0 and stats["p_개선아님"] <= 0.05:
        return "개선"
    return "채택"


def entry_weights(trades: list[dict[str, Any]], eq: pd.Series) -> np.ndarray:
    """매수 금액 ÷ 매수 전날 평가액 (0주 매수는 0). 매수일은 다음날 시가라 전날 종가 평가액을 쓴다."""
    out = []
    for t in trades:
        prior = eq.loc[: pd.Timestamp(t["entry_date"]) - pd.Timedelta(days=1)]
        if prior.empty:
            continue
        cost = float(t.get("shares") or 0.0) * float(t["entry_price"]) * (1 + bt.BACKTEST_COST_PER_SIDE)
        out.append(cost / float(prior.iloc[-1]))
    return np.array(out)


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
    eq.rename("equity").to_frame().to_parquet(RESEARCH_DIR / f"sizing_{key}_dev_equity.parquet")
    s = result["summary"]
    print(f"[sizing] {key}: Sharpe(일간) {s.get('Sharpe_일간')} · CAGR {s.get('CAGR')} · MDD {s.get('MDD')} · "
          f"거래 {s.get('총거래수')}", flush=True)
    return {"equity": eq, "summary": s, "trades": result["trades"]}


def _yearly(eq: pd.Series) -> pd.Series:
    return eq.groupby(eq.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)  # pt1_baseline_equity 와 같은 방식


def run() -> None:
    log = _log()
    if not ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        raise SystemExit("먼저 --register 로 등록한다 (결과 계산 전 사전 등록)")
    already_logged = ((log["family"] == FAMILY) & (log["판정"] != "등록")).any()

    runs = {key: _backtest(key) for key in VARIANTS}

    print("===== 기준선 재현 확인", flush=True)
    saved = pd.read_parquet(BASE_SAVED).iloc[:, 0]
    base_eq = runs["BASE"]["equity"]
    common = saved.index.intersection(base_eq.index)
    max_rel = float((base_eq.loc[common] / saved.loc[common] - 1).abs().max()) if len(common) else float("inf")
    if len(common) != len(saved) or max_rel > 1e-9:
        raise SystemExit(f"BASE 가 저장된 기준선과 다르다 (겹친 날 {len(common)}/{len(saved)}, 최대 차이 {max_rel:.2e}) "
                         "— 코드·캐시를 확인하기 전에는 결과를 쓰지 않는다")
    print(f"[sizing] BASE = 저장된 기준선 ({len(common)}일, 최대 차이 {max_rel:.1e})", flush=True)

    print("===== 비교·판정", flush=True)
    etf = load_ohlcv(ETF_OHLCV)
    spy_r = etf["SPY"]["Close"].pct_change()
    base_r = base_eq.pct_change().dropna()
    rows = []
    for key in VARIANTS:
        eq = runs[key]["equity"]
        r = eq.pct_change().dropna()
        df = pd.concat({"v": r, "b": base_r, "spy": spy_r.reindex(r.index)}, axis=1).dropna()
        a, b, sp = df["v"].tolist(), df["b"].tolist(), df["spy"].tolist()
        st = paired_bootstrap(a, b) if key != "BASE" else {}
        st_spy = paired_bootstrap(a, sp)
        trades = runs[key]["trades"]
        zero = sum(1 for t in trades if float(t.get("shares") or 0.0) <= 0)
        w = entry_weights(trades, eq)
        rows.append({
            "test_id": key, "이름": VARIANTS[key][0], "시작": df.index.min().date(), "끝": df.index.max().date(),
            "일수": len(df), "Sharpe": sharpe(a), "CAGR": cagr(a), "MDD": max_drawdown(a),
            "거래수": len(trades), "0주매수": zero,
            "승률": float(np.mean([float(t.get("return_pct") or 0) > 0 for t in trades])) if trades else np.nan,
            "비중_중앙": float(np.median(w[w > 0])) if (w > 0).any() else np.nan,
            "비중50%초과": int((w > 0.5).sum()), "비중90%초과": int((w > 0.9).sum()),
            "ΔSharpe_BASE": st.get("ΔSharpe"), "하한_BASE": st.get("ΔSharpe_하한"), "상한_BASE": st.get("ΔSharpe_상한"),
            "p_BASE": st.get("p_개선아님"), "구간_BASE": f"{st.get('구간_개선')}/{st.get('구간_수')}" if st else "",
            "SPY_Sharpe": sharpe(sp), "ΔSharpe_SPY": st_spy.get("ΔSharpe"), "하한_SPY": st_spy.get("ΔSharpe_하한"),
            "상한_SPY": st_spy.get("ΔSharpe_상한"),
            "판정": judge(st, zero) if key != "BASE" else "기준",
        })
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    if already_logged:
        print("결과 행이 이미 TRIAL_LOG 에 있다 — 다시 쓰지 않는다")
    else:
        b0 = res.iloc[0]
        log_trials(STEP, FAMILY, N_TESTS, [{
            "test_id": r["test_id"], "판정": r["판정"],
            "메모": (f"Sharpe {r['Sharpe']:.2f} vs BASE {b0['Sharpe']:.2f}, ΔSharpe {r['ΔSharpe_BASE']:+.2f} "
                     f"[{r['하한_BASE']:+.2f}, {r['상한_BASE']:+.2f}] p {r['p_BASE']:.3f}, 구간 {r['구간_BASE']}, "
                     f"CAGR {r['CAGR']:+.1%}, MDD {r['MDD']:.1%}, 거래 {r['거래수']}, 0주 {r['0주매수']} "
                     f"(BASE {b0['0주매수']}), 비중 90%+ {r['비중90%초과']} (BASE {b0['비중90%초과']})")}
            for _, r in res.iterrows() if r["test_id"] != "BASE"])

    write_report(res, {k: runs[k]["equity"] for k in runs}, etf["SPY"]["Close"])


def _f(v: Any, fmt: str) -> str:
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)


def write_report(res: pd.DataFrame, eqs: dict[str, pd.Series], spy_close: pd.Series) -> None:
    fix = res[res["test_id"] == "FIX"].iloc[0]
    lines = [
        f"# PT-1 매수 비중 버그 수정 ({FAMILY}) — {RUN_KW['start_date']} ~ {RUN_KW['end_date']}",
        "",
        f"- 사전 등록: TRIAL_LOG family `{FAMILY}`, {N_TESTS}건 · 누적 시도 {total_trials()}건",
        "- 계기: 약세장(최대 2종목)에서 두 번째 매수가 남은 현금을 전부 쓰고, 레짐이 풀리면 세 번째 매수가 0주가 됐다",
        "- 바꾼 것: 빈자리를 약세장 한도 대신 항상 3자리 기준으로 센다 (실거래 $5,000 계좌와 같은 회계). 나머지는 지금 규칙",
        "- 기준 = 같은 실행의 BASE (저장된 PT-1 기준선과 같은지 확인함)",
        "- 버그 수정이라 채택 = 유의하게 나빠지지 않음 (ΔSharpe 95% 상한 > 0) · 0주 매수 0건. 개선 = 하한 > 0 · p ≤ 0.05",
        f"- **결과: {fix['판정']}**",
        "",
        "| 변형 | Sharpe | CAGR | MDD | 거래 | 0주 매수 | 승률 | BASE 대비 ΔSharpe [95%] | p | 구간 | SPY 대비 ΔSharpe [95%] | 판정 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.append(f"| SPY 보유 (종가) | {_f(res.iloc[0]['SPY_Sharpe'], '.2f')} | | | | | | | | | | 참고 |")
    for _, r in res.iterrows():
        base_cell = ("" if r["test_id"] == "BASE" else
                     f"{_f(r['ΔSharpe_BASE'], '+.2f')} [{_f(r['하한_BASE'], '+.2f')}, {_f(r['상한_BASE'], '+.2f')}]")
        lines.append(
            f"| {r['test_id']} {r['이름']} | {_f(r['Sharpe'], '.2f')} | {_f(r['CAGR'], '+.1%')} | {_f(r['MDD'], '.1%')} | "
            f"{r['거래수']} | {r['0주매수']} | {_f(r['승률'], '.0%')} | {base_cell} | {_f(r['p_BASE'], '.3f')} | "
            f"{r['구간_BASE']} | {_f(r['ΔSharpe_SPY'], '+.2f')} [{_f(r['하한_SPY'], '+.2f')}, {_f(r['상한_SPY'], '+.2f')}] | "
            f"**{r['판정']}** |")
    lines += ["", "## 매수 한 건의 크기 (매수 금액 ÷ 전날 평가액, 0주 제외)", "",
              "| 변형 | 중앙값 | 50% 초과 | 90% 초과 |", "|---|---|---|---|"]
    lines += [f"| {r['test_id']} | {_f(r['비중_중앙'], '.0%')} | {r['비중50%초과']}건 | {r['비중90%초과']}건 |"
              for _, r in res.iterrows()]
    yr = pd.DataFrame({k: _yearly(v) for k, v in eqs.items()})
    spy = spy_close.loc[eqs["BASE"].index.min():eqs["BASE"].index.max()]
    yr["SPY"] = _yearly(spy)
    lines += ["", "## 연도별 수익", "", "| 연도 | " + " | ".join(yr.columns) + " |", "|---" * (len(yr.columns) + 1) + "|"]
    lines += [f"| {y} | " + " | ".join(_f(v, "+.1%") for v in row) + " |" for y, row in yr.iterrows()]
    lines += ["", "## 한계", "",
              "- 개발 구간은 지금까지 많은 시도에 쓰였다. 이 검정은 버그 수정의 영향 확인이지 새 규칙 탐색이 아니다.",
              "- 실거래 거래로그는 거래별 % 라 바뀌지 않고, 실거래 $5,000 계좌는 이미 이 회계를 쓴다. 바뀌는 것은 연구 기준선이다.",
              "- 상장폐지 종목 가격이 없는 생존 편향은 기준선과 같다.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"리포트: {REPORT}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--register", action="store_true", help="결과 계산 전 사전 등록 (TRIAL_LOG)")
    g.add_argument("--run", action="store_true", help="BASE·FIX 백테스트 → 비교·판정·리포트")
    args = p.parse_args()
    register() if args.register else run()


if __name__ == "__main__":
    main()
