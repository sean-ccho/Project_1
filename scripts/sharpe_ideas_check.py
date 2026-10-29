#!/usr/bin/env python3
"""PT-1 Sharpe 를 높일 구조 아이디어 4개 — 10년 PT-1 (sharpe_v1). 신호를 더하지 않고 위험 배분만 바꾼다.

2026-10-28 사용자 요청 ("sharpe 를 높일수있는 방법 찾아"). 지금까지(TRIAL_LOG 4,459건) 신호 · 타이밍 · 청산 · 슬롯 수 · CCS v2 ·
시장 변동성 관리는 모두 운과 구분되지 않았고, CCS 순위도 문턱을 넘은 후보 중 무작위와 구분되지 않았다 (ccs_placebo_10y).
그래서 '무엇을 사느냐' 대신 '위험을 어떻게 나누느냐' — 근거가 있고 자유도가 거의 없는 것만 고른다:
  INVVOL 종목 비중 ∝ 1/변동성 (엔진의 PT1_SIZING inv_vol: 한 자리 × min(1, 그날 중앙 ATR% ÷ 종목 ATR%), 남는 돈은 현금)
         — 위험 균등. 근거: 위험 기여를 맞추면 수익보다 변동성이 더 줄어든다 (저변동성 이상현상과 같은 뿌리)
  LOWVOL 그날 후보표 ATR% 중앙값보다 변동성이 큰 종목은 사지 않음 — 저변동성 이상현상
         (Ang · Hodrick · Xing · Zhang 2006, Baker · Bradley · Wurgler 2011, Frazzini · Pedersen 2014)
  MOM    모멘텀 전략만 매수 (바닥반등 안 삼) — 모멘텀은 가장 오래 확인된 이상현상, 단기 반전(바닥반등)은 대형주에서 약함.
         (ab_ccs_fix · holdout_h2 에서 3년 · 1년 창으로 본 적 있음 — 10년 창 · 지금 라벨로는 처음)
  SECT1  같은 섹터는 1종목까지 (CANDIDATE_MAX_SAME_SECTOR 2 → 1) — 3종목끼리 상관을 낮춰 변동성을 줄인다
기준 = 지금 코드(C 라벨) + 빈자리 균등 크기(PT1_SIZING equal_fixed — 실계좌 $5,000 보기와 같은 셈, 2026-10-14 버그 수정 채택).
피처 캐시 C(260b31a3aead_v1)는 읽기만 한다 (바꾸는 곳이 모두 피처 계산 뒤).

사전 등록 (TRIAL_LOG family sharpe_v1, 4건 — 결과 보기 전에 --register):
  시험마다 ΔSharpe(아이디어 − 기준), 일간 수익, 20일 블록 부트스트랩 10,000회.
  '채택 후보' = 재표본 중 Δ ≤ 0 비율 < 0.05/2/4 (= 0.625%, 4건 본페로니 · 한쪽). 채택 후보도 바로 실전에 넣지 않고
  홀드아웃(2025-10-01 이후, 개발 구간 밖)에서 Sharpe 가 기준보다 높아야 제안한다. 그 밖 = '효과 없음'.
  참고(판정 없음): 최대낙폭 · 변동성 · 거래 겹침 · SPY 와 섞은 비중별 Sharpe · 진입 변동성 5분위별 거래 수익 · 전략별 거래.

사용법: PYTHONPATH=.:src .venv/bin/python scripts/sharpe_ideas_check.py --register
        PYTHONPATH=.:src .venv/bin/python scripts/sharpe_ideas_check.py --run --workers 4
결과: docs/quant_improvement/tier3/SHARPE_IDEAS.md · sharpe_ideas.csv · data/research/sharpe_ideas_equity.parquet
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import goldencross_pt1_check as gc  # noqa: E402
import isb_engine as isb  # noqa: E402
import isb_vs_pt1 as iv  # noqa: E402
import paper_trading.backtest as bt  # noqa: E402
from paper_trading.config_override import config_overrides  # noqa: E402
from paper_trading.evaluation import cagr, max_drawdown, sharpe  # noqa: E402
from research_utils import RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, log_trials  # noqa: E402
from validate_upside_walkforward import RUN_KW  # noqa: E402

FAMILY = "sharpe_v1"
STEP = "sharpe"
N_BOOT = 10_000
CODE_HASH = "b9fc8bd3"
C_KEY = "260b31a3aead"
C_EQUITY = RESEARCH_DIR / "goldencross_pt1_c_equity.parquet"
REPORT = TIER3_DIR / "SHARPE_IDEAS.md"
OUT_CSV = TIER3_DIR / "sharpe_ideas.csv"
EQUITY_OUT = RESEARCH_DIR / "sharpe_ideas_equity.parquet"
BASE_OVERRIDES = {"PT1_SIZING": "equal_fixed"}

TESTS: dict[str, str] = {
    "INVVOL": "종목 비중 ∝ 1/변동성 (inv_vol: 한 자리 × min(1, 중앙 ATR% ÷ 종목 ATR%), 남는 돈 현금)",
    "LOWVOL": "그날 후보표 ATR% 중앙값보다 변동성 큰 종목은 안 삼",
    "MOM": "모멘텀 전략만 매수 (바닥반등 안 삼)",
    "SECT1": "같은 섹터 1종목까지 (2 → 1)",
}
ALPHA_ONE_SIDED = 0.05 / 2 / len(TESTS)
BLENDS = (0.0, 0.5, 0.8, 1.0)       # SPY 비중 (나머지 PT-1, 매일 비중 맞춤)


def overrides_for(test: str) -> dict[str, Any]:
    if test == "INVVOL":
        return {"PT1_SIZING": "inv_vol"}
    if test == "MOM":
        return {**BASE_OVERRIDES, "CANDIDATE_ALLOWED_STRATEGIES": ["모멘텀"]}
    return dict(BASE_OVERRIDES)


def apply_idea(test: str) -> None:
    """모듈 속성 바꿔치기가 필요한 아이디어 (이 프로세스에서만)."""
    import paper_trading.candidate_selector as cs

    if test == "LOWVOL":
        orig = cs._apply_hard_filters

        def filters(df: pd.DataFrame, current_holdings: list[dict[str, Any]], rsi_max: float | None = None):
            passed, rej = orig(df, current_holdings, rsi_max)
            atr = pd.to_numeric(df.get("ATR%", pd.Series(dtype=float)), errors="coerce")
            atr = atr[(atr > 0) & (atr < 1)]             # inv_vol_factor 와 같은 정상값 · 같은 기준(그날 후보표 전체 중앙값)
            if passed.empty or atr.empty:
                return passed, rej
            own = pd.to_numeric(passed.get("ATR%", pd.Series(np.nan, index=passed.index)), errors="coerce")
            keep = own.isna() | (own <= atr.median())   # 값이 없으면 inv_vol 처럼 걸지 않는다
            rej["고변동성"] = int((~keep).sum())
            passed = passed[keep]
            rej["_통과"] = len(passed)
            return passed, rej

        cs._apply_hard_filters = filters
    elif test == "SECT1":
        cs.CANDIDATE_MAX_SAME_SECTOR = 1
    elif test not in ("CODE", "INVVOL", "MOM"):
        raise ValueError(test)


def _engine(test: str) -> tuple[str, iv.RunResult, float]:
    t0 = time.time()
    apply_idea(test)
    with config_overrides(overrides_for(test)):
        res = iv.run_overlay(iv.Overlay("BASE"))
    return test, res, time.time() - t0


# ============================================================
# 참고 통계 (판정 없음)
# ============================================================
def _ann_vol(r: pd.Series) -> float:
    return float(r.std() * math.sqrt(252))


def blend_table(pt: pd.Series, spy: pd.Series) -> pd.DataFrame:
    """SPY 비중 w · PT-1 1−w 로 매일 비중을 맞춘 계좌 (비용 없음 — 참고)."""
    df = pd.concat({"pt": pt, "spy": spy}, axis=1).dropna()
    rows = []
    for w in BLENDS:
        r = w * df["spy"] + (1 - w) * df["pt"]
        rows.append({"SPY 비중": f"{w:.0%}", "Sharpe": sharpe(r.tolist()), "연수익": cagr(r.tolist()),
                     "최대낙폭": max_drawdown(r.tolist()), "연변동성": _ann_vol(r)})
    return pd.DataFrame(rows)


def trade_tables(trades: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전략별 · 진입 변동성(ATR%) 5분위별 거래 수익 (0주 매수 제외)."""
    t = pd.DataFrame([{"strategy": x.get("strategy", ""), "ret": float(x["return_pct"]),
                       "days": x.get("holding_days"), "atr": (x.get("entry_features") or {}).get("atr_pct")}
                      for x in trades if float(x.get("shares", 1) or 0) > 0])
    by_strat = t.groupby("strategy").agg(거래=("ret", "size"), 평균=("ret", "mean"), 중앙=("ret", "median"),
                                          승률=("ret", lambda s: float((s > 0).mean())), 표준편차=("ret", "std"),
                                          보유일=("days", "mean"))
    t["atr"] = pd.to_numeric(t["atr"], errors="coerce")
    q = t.dropna(subset=["atr"]).copy()
    q["분위"] = pd.qcut(q["atr"], 5, labels=["1 (낮음)", "2", "3", "4", "5 (높음)"])
    by_atr = q.groupby("분위", observed=True).agg(ATR_중앙=("atr", "median"), 거래=("ret", "size"), 평균=("ret", "mean"),
                                                   표준편차=("ret", "std"), 승률=("ret", lambda s: float((s > 0).mean())))
    by_atr["평균÷표준편차"] = by_atr["평균"] / by_atr["표준편차"]
    return by_strat, by_atr


# ============================================================
# 등록 · 실행 · 비교
# ============================================================
def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame(columns=["family", "판정"])


def _registered() -> bool:
    log = _log()
    return bool(((log["family"] == FAMILY) & (log["판정"] == "등록")).any())


def _logged(test: str) -> bool:
    log = _log()
    return bool(((log["family"] == FAMILY) & (log["test_id"] == test)).any())


def register() -> None:
    if _registered():
        print("이미 등록됨", flush=True)
        return
    tests = " / ".join(f"{k}: {v}" for k, v in TESTS.items())
    memo = (f"[{RUN_KW['start_date']}~{RUN_KW['end_date']}, PT-1 엔진 BASE 설정 · 가격 · 종목 목록 고정, 피처 캐시 C({C_KEY}) 읽기만, "
            f"기준 = 지금 코드 + PT1_SIZING equal_fixed] 신호 없이 위험 배분만 바꾸는 {len(TESTS)}건 — {tests}. "
            f"ΔSharpe(아이디어−기준), 20일 블록 부트스트랩 {N_BOOT:,}회. 채택 후보 = 재표본 중 Δ≤0 비율 < {ALPHA_ONE_SIDED:.3%} "
            f"({len(TESTS)}건 본페로니, 한쪽) — 그래도 홀드아웃(2025-10-01~)에서 기준보다 높아야 실전 제안. 그 밖 = 효과 없음. "
            "참고(판정 없음): MDD · 변동성 · 겹침 · SPY 섞은 비중별 Sharpe · ATR 5분위 · 전략별 거래.")
    log_trials(STEP, FAMILY, len(TESTS), [{"test_id": "(등록)", "판정": "등록", "메모": memo}])
    print(f"등록: {FAMILY} {len(TESTS)}건", flush=True)


def run(workers: int) -> None:
    if not _registered():
        raise SystemExit("먼저 --register (결과 보기 전에 기준 등록)")
    code = bt._code_hash()
    if code != CODE_HASH or gc.cache_key(code) != C_KEY:
        raise SystemExit(f"코드 지문 {code} ≠ {CODE_HASH} — 시험한 C 코드가 아니라 멈춤")
    iv.ensure_pinned()
    dates = gc.target_dates()
    n_cache = len(gc.cached(C_KEY, dates))
    if n_cache != len(dates):
        raise SystemExit("C 캐시가 다 있지 않다 — 멈춤")
    print(f"[sharpe] 피처 {len(dates)}일 · 시험 {len(TESTS)}건 · 워커 {workers}개", flush=True)
    t0 = time.time()
    runs: dict[str, iv.RunResult] = {}
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
        futs = [ex.submit(_engine, t) for t in ["CODE", *TESTS]]
        for fut in as_completed(futs):
            test, res, sec = fut.result()
            runs[test] = res
            print(f"  {test} 끝 · 거래 {len(res.trades)} · {sec:.0f}초 · {len(runs)}/{len(TESTS) + 1}", flush=True)
    if len(gc.cached(C_KEY, dates)) != n_cache:
        print("  ⚠️ 피처 캐시 파일 수가 바뀌었다", flush=True)
    compare(runs)
    print(f"전체 {gc._hm(time.time() - t0)}", flush=True)


def compare(runs: dict[str, iv.RunResult]) -> None:
    print("===== 비교 · 판정", flush=True)
    eq = {k: r.equity for k, r in runs.items()}
    saved = pd.read_parquet(C_EQUITY)["NEW_EF"].reindex(eq["CODE"].index)
    diff = saved.sub(eq["CODE"]).abs().max()
    print(f"  기준 vs 저장된 C 균등 크기 곡선 최대 차이 {diff:.6f}", flush=True)
    if not diff < 1e-6:
        raise SystemExit("기준이 C 시험 결과와 다르다 — 멈춤")
    raw = pd.read_parquet(isb.PRICE_CACHE)
    idx = eq["CODE"].index
    spy = raw["SPY"]["Close"].reindex(idx)
    eq["SPY"] = spy / spy.iloc[0] * float(eq["CODE"].iloc[0])
    pd.DataFrame(eq).to_parquet(EQUITY_OUT)
    rets = {k: v.pct_change().dropna() for k, v in eq.items()}

    base = gc._stats(runs["CODE"].equity, runs["CODE"].trades)
    base["연변동성"] = _ann_vol(rets["CODE"])
    rows, log_rows = [], []
    for t, desc in TESTS.items():
        st = gc._stats(runs[t].equity, runs[t].trades)
        m, _ = iv._cmp(rets[t], rets["CODE"], N_BOOT)
        p = m["p_개선아님"]
        verdict = "채택 후보 (홀드아웃 확인)" if p < ALPHA_ONE_SIDED else "효과 없음"
        ov, _, _ = gc._overlap(runs["CODE"].trades, runs[t].trades)
        rows.append({"test": t, "내용": desc, "Sharpe": st["Sharpe"], "ΔSharpe": m["ΔSharpe"], "하한": m["ΔSharpe_하한"],
                     "상한": m["ΔSharpe_상한"], "p_개선아님": p, "구간": f"{m['구간_개선']}/{m['구간_수']}", "연수익": st["CAGR"],
                     "최대낙폭": st["MDD"], "연변동성": _ann_vol(rets[t]), "거래": st["거래수"], "겹침": ov, "판정": verdict})
        log_rows.append({"test_id": t, "판정": verdict,
                         "메모": (f"{desc}: Sharpe {st['Sharpe']:.2f} vs 기준 {base['Sharpe']:.2f}, ΔSharpe {m['ΔSharpe']:+.2f} "
                                f"[{m['ΔSharpe_하한']:+.2f}, {m['ΔSharpe_상한']:+.2f}] p(개선 아님) {p:.4f} (기준 < {ALPHA_ONE_SIDED:.4f}), "
                                f"CAGR {st['CAGR']:+.1%}, MDD {st['MDD']:.1%}, 변동성 {_ann_vol(rets[t]):.1%}, 거래 {st['거래수']} (겹침 {ov})")})
        print(f"  {t}: {verdict} — {log_rows[-1]['메모']}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    todo = [r for r in log_rows if not _logged(r["test_id"])]
    if todo:
        log_trials(STEP, FAMILY, len(TESTS), todo)

    spy_r = rets["SPY"]
    spy_st = {"Sharpe": sharpe(spy_r.tolist()), "CAGR": cagr(spy_r.tolist()), "MDD": max_drawdown(spy_r.tolist()),
              "연변동성": _ann_vol(spy_r)}
    both = pd.concat({"pt": rets["CODE"], "spy": spy_r}, axis=1).dropna()
    corr = float(both["pt"].corr(both["spy"]))
    beta = float(both["pt"].cov(both["spy"]) / both["spy"].var())
    blends = blend_table(rets["CODE"], spy_r)
    by_strat, by_atr = trade_tables(runs["CODE"].trades)
    write_report(out, base, spy_st, corr, beta, blends, by_strat, by_atr)


def _p(v: float) -> str:
    return f"{v:+.1%}"


def write_report(out: pd.DataFrame, base: dict[str, Any], spy: dict[str, float], corr: float, beta: float,
                 blends: pd.DataFrame, by_strat: pd.DataFrame, by_atr: pd.DataFrame) -> None:
    L = [
        "# PT-1 Sharpe 구조 아이디어 4개 — 10년 (sharpe_v1)",
        "",
        f"*{pd.Timestamp.now(tz='America/Toronto'):%Y-%m-%d %H:%M} ET · scripts/sharpe_ideas_check.py · 사전 등록 TRIAL_LOG {FAMILY} "
        f"{len(TESTS)}건*",
        "",
        "신호를 더하지 않고 위험 배분만 바꿨다. 같은 10년(2016-01 ~ 2025-09) · 같은 가격 · 같은 종목 목록 · 같은 피처(C 캐시, 읽기만).",
        f"기준 = 지금 코드 + 빈자리 균등 크기: Sharpe {base['Sharpe']:.2f} · 연 {base['CAGR']:+.1%} · 최대낙폭 {base['MDD']:.1%} · "
        f"변동성 {base['연변동성']:.1%} · 거래 {base['거래수']}. SPY: Sharpe {spy['Sharpe']:.2f} · 연 {spy['CAGR']:+.1%} · "
        f"최대낙폭 {spy['MDD']:.1%} · 변동성 {spy['연변동성']:.1%}.",
        f"판정 기준 (등록): 재표본 중 Δ ≤ 0 비율 < {ALPHA_ONE_SIDED:.3%} ({len(TESTS)}건 본페로니, 한쪽) 이면 채택 후보 → 홀드아웃 확인.",
        "",
        "| 시험 | 내용 | Sharpe | ΔSharpe | 95% 구간 | p(개선 아님) | 연수익 | 최대낙폭 | 변동성 | 거래 (겹침) | 판정 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *[f"| {r.test} | {r.내용} | {r.Sharpe:.2f} | {r.ΔSharpe:+.2f} | [{r.하한:+.2f}, {r.상한:+.2f}] | {r.p_개선아님:.3f} | "
          f"{r.연수익:+.1%} | {r.최대낙폭:.1%} | {r.연변동성:.1%} | {r.거래} ({r.겹침}) | {r.판정} |" for r in out.itertuples()],
        "",
        "## 참고 1 — PT-1(기준)과 SPY",
        "",
        f"일간 상관 {corr:.2f} · 베타 {beta:.2f}. PT-1 Sharpe 가 SPY Sharpe × 상관({spy['Sharpe'] * corr:.2f})보다 낮으면 "
        "SPY 에 PT-1 을 섞을수록 Sharpe 가 내려간다 (두 자산 최대 Sharpe 조건).",
        "",
        "| SPY 비중 (나머지 PT-1) | Sharpe | 연수익 | 최대낙폭 | 연변동성 |",
        "|---|---|---|---|---|",
        *[f"| {r['SPY 비중']} | {r.Sharpe:.2f} | {r.연수익:+.1%} | {r.최대낙폭:.1%} | {r.연변동성:.1%} |" for _, r in blends.iterrows()],
        "",
        "## 참고 2 — 전략별 거래 (기준)",
        "",
        "| 전략 | 거래 | 평균 | 중앙 | 승률 | 표준편차 | 보유일 |",
        "|---|---|---|---|---|---|---|",
        *[f"| {k} | {r.거래} | {_p(r.평균)} | {_p(r.중앙)} | {r.승률:.0%} | {r.표준편차:.1%} | {r.보유일:.1f} |"
          for k, r in by_strat.iterrows()],
        "",
        "## 참고 3 — 진입 변동성(ATR%) 5분위별 거래 (기준)",
        "",
        "| 분위 | ATR% 중앙 | 거래 | 평균 | 표준편차 | 승률 | 평균 ÷ 표준편차 |",
        "|---|---|---|---|---|---|---|",
        *[f"| {k} | {r.ATR_중앙:.1%} | {r.거래} | {_p(r.평균)} | {r.표준편차:.1%} | {r.승률:.0%} | {r['평균÷표준편차']:+.2f} |"
          for k, r in by_atr.iterrows()],
        "",
        "## 구현",
        "",
        "- 기준 · LOWVOL · MOM · SECT1 은 PT1_SIZING equal_fixed, INVVOL 은 PT1_SIZING inv_vol",
        "- LOWVOL: 하드 필터 뒤 ATR% 가 그날 후보표 전체(0 < ATR% < 1) 중앙값보다 큰 종목을 뺌 (ATR% 없으면 남김)",
        "- MOM: CANDIDATE_ALLOWED_STRATEGIES = ['모멘텀'] · SECT1: candidate_selector.CANDIDATE_MAX_SAME_SECTOR = 1",
        "- 기준 결과는 C 시험의 저장된 균등 크기 곡선(NEW_EF)과 같음을 확인 (차이 0)",
    ]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"보고서: {REPORT.relative_to(ROOT)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.register:
        register()
    if args.run:
        run(args.workers)


if __name__ == "__main__":
    main()
