#!/usr/bin/env python3
"""문서 기준 vs 지금 코드 — 문서와 다른 PT-1 규칙 8곳을 하나씩 문서대로 바꿔 10년 PT-1 로 비교 (doc_vs_code_v1).

2026-10-28 사용자 요청 ("기준이 좋은지 실제 코드가 좋은지도 확인해봐"): 문서(patterns_reference · screener_filtering_guide ·
paper_trading_candidate_selection · architecture)와 코드가 다른 곳 중 PT-1 이 사는 종목을 바꾸는 것만 시험한다.
같은 10년(2016-01-04 ~ 2025-09-30) · 같은 PT-1 규칙 · 같은 가격 · 같은 종목 목록에서 한 번에 한 곳만 문서대로 바꾼다.

피처 캐시(C — 260b31a3aead_v1, 지금 코드 b9fc8bd3)는 읽기만 한다. 바꾸는 곳이 모두 피처 계산 뒤(라벨 후처리 · 점수 · 필터)라
다시 계산하지 않아도 그 규칙으로 처음부터 계산한 것과 같다. GI2 는 임박 판정에서 간격 조건만 다르고 나머지 조건이 같으므로
'간격이 2% 넘는 임박 라벨을 지운 것' = '2% 규칙으로 계산한 라벨' (그 자리에 다른 교차 라벨이 생길 수 없다).

사전 등록 (TRIAL_LOG family doc_vs_code_v1, 8건 — 결과 보기 전에 --register):
  시험마다 ΔSharpe(문서 − 코드), 일간 수익, 20일 블록 부트스트랩 10,000회.
  판정: '문서가 확실히 좋음' = 재표본 중 Δ ≤ 0 비율 < 0.05/2/8 (= 0.3125%, 8건 본페로니 · 한쪽) → 문서대로 바꾸자고 제안.
        그 밖 = '코드 유지' → 문서를 코드에 맞춘다. 참고(판정 없음): 빈자리 균등 크기(equal_fixed) · 거래 겹침.
참고 (시험 아님): 도지 몸통 기준 5% vs 10% 가 붙는 종목-일 비율 — 도지는 점수 · PT-1 에 안 쓴다 (signals · candidate_selector).

사용법: PYTHONPATH=.:src .venv/bin/python scripts/doc_vs_code_check.py --register
        PYTHONPATH=.:src .venv/bin/python scripts/doc_vs_code_check.py --run --workers 3
결과: docs/quant_improvement/tier3/DOC_VS_CODE.md · doc_vs_code.csv · data/research/doc_vs_code_equity.parquet
"""

from __future__ import annotations

import argparse
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
from screener.cache import feature_cache_dir  # noqa: E402
from validate_upside_walkforward import RUN_KW  # noqa: E402

FAMILY = "doc_vs_code_v1"
STEP = "doc_vs_code"
N_BOOT = 10_000
CODE_HASH = "b9fc8bd3"           # 지금 코드 = C (2026-10-28 푸쉬) — 이 지문이 아니면 멈춘다
C_KEY = "260b31a3aead"           # C 피처 캐시 — 읽기만
C_EQUITY = RESEARCH_DIR / "goldencross_pt1_c_equity.parquet"
REPORT = TIER3_DIR / "DOC_VS_CODE.md"
OUT_CSV = TIER3_DIR / "doc_vs_code.csv"
EQUITY_OUT = RESEARCH_DIR / "doc_vs_code_equity.parquet"
EF = {"PT1_SIZING": "equal_fixed"}

# (문서 · 코드 · 어디 문서에 적힌 값인지)
TESTS: dict[str, tuple[str, str, str]] = {
    "GI2": ("교차 임박 간격 2% 안", "5% 안", "patterns_reference · screener_filtering_guide (예전)"),
    "RW": ("레짐 가중치 bull .25/.15/.25/.15/.20 · neutral .25/.20/.20/.20/.15 · bear .20/.25/.15/.30/.10",
           "bull .25/.20/.15/.20/.20 · neutral .25/.20/.15/.25/.15 · bear .20/.25/.10/.30/.15",
           "architecture · paper_trading_candidate_selection (전략/타이밍/알파/리스크/컨플루언스)"),
    "MACD": ("바닥반등 MACD 히스토그램 > −0.5 (달러) 이면 +0.1", "히스토그램 ÷ 주가 > −1% 이면 +0.1",
             "paper_trading_candidate_selection B-4"),
    "F52W": ("바닥반등 52주 위치 필터 없음", "52주 위치 ≤ 0.65 만", "paper_trading_candidate_selection Phase 1"),
    "FADX": ("모멘텀 ADX 필터 없음", "모멘텀은 ADX ≥ 25 만", "paper_trading_candidate_selection Phase 1"),
    "FRSI": ("바닥반등 RSI 30~35 차단 없음", "바닥반등 RSI 30~35 차단", "paper_trading_candidate_selection Phase 1"),
    "RTREND": ("리스크 점수에 주·월봉 정배열 가점 없음", "월봉 +0.10/0.05/0.02 · 주봉 +0.05/0.025",
               "paper_trading_candidate_selection D"),
    "CONF": ("컨플루언스 상승 패턴 9개 (정배열 없음)", "11개 (정배열(완전) · 정배열(눌림목) 포함)",
             "paper_trading_candidate_selection E-3"),
}
ALPHA_ONE_SIDED = 0.05 / 2 / len(TESTS)

# v2 (2026-10-28, v1 결과를 본 뒤 문서를 고치다 찾은 차이 — 1건, 기준은 v1 과 같게 0.3125% 로 둔다):
# config 주석(LIQUIDITY_QUANTILE '하위 25% 제거 = 상위 75% 유지') · architecture · screener_filtering_guide 는 하위 25% 제거인데
# processing.liquidity_filter 는 quantile(1 − 0.25) 이상만 남겨 상위 25% 만 남긴다 (S&P 553~609종목 중 139~153개)
TESTS_V2: dict[str, tuple[str, str, str]] = {
    "LIQ": ("유동성: 시장별 거래대금 하위 25% 만 뺌 (상위 75% 남김)", "상위 25% 만 남김",
            "config LIQUIDITY_QUANTILE 주석 · architecture · screener_filtering_guide"),
}


def use_v2() -> None:
    """유동성 필터 시험 (doc_vs_code_v2) — 등록 · 결과 행 · 보고서 · 결과 파일 이름만 바꾼다."""
    global FAMILY, TESTS, REPORT, OUT_CSV, EQUITY_OUT
    FAMILY, TESTS = "doc_vs_code_v2", TESTS_V2
    REPORT = TIER3_DIR / "DOC_VS_CODE_LIQ.md"
    OUT_CSV = TIER3_DIR / "doc_vs_code_liq.csv"
    EQUITY_OUT = RESEARCH_DIR / "doc_vs_code_liq_equity.parquet"


def liquidity_doc(df: pd.DataFrame) -> pd.DataFrame:
    """processing.liquidity_filter 를 주석 · 문서대로: 시장별 거래대금 하위 LIQUIDITY_QUANTILE 만 뺀다 (나머지 조건은 같음)."""
    from screener import processing as pr

    out = df.copy()
    thresholds = out.groupby("시장")["최근20일평균거래대금"].transform(lambda s: s.quantile(pr.LIQUIDITY_QUANTILE))
    mask = out["최근20일평균거래대금"] >= thresholds
    mask &= out["최근20일평균거래대금"] >= pr.LIQUIDITY_DOLLAR_MIN
    if pr.LIQUIDITY_TURNOVER_MIN > 0 and "거래대금회전율" in out.columns:
        mask &= out["거래대금회전율"] >= pr.LIQUIDITY_TURNOVER_MIN
    if pr.LIQUIDITY_WHITELIST:
        mask |= out["티커"].isin(pr.LIQUIDITY_WHITELIST)
    return out[mask]

DOC_REGIME_WEIGHTS = {
    "bull": {"strategy": 0.25, "timing": 0.15, "alpha": 0.25, "risk": 0.15, "confluence": 0.20},
    "neutral": {"strategy": 0.25, "timing": 0.20, "alpha": 0.20, "risk": 0.20, "confluence": 0.15},
    "bear": {"strategy": 0.20, "timing": 0.25, "alpha": 0.15, "risk": 0.30, "confluence": 0.10},
}
DOC_BULLISH = {"이중바닥", "역헤드앤숄더", "하락쐐기", "강세잉걸핑", "모닝스타", "상승삼각형", "컵앤핸들", "골든크로스", "골든크로스임박"}
GAP_COLS = {"일봉패턴": "일봉_MA갭", "주봉패턴": "주봉_MA갭", "월봉패턴": "월봉_MA갭"}
IMMINENT = ("골든크로스임박", "데드크로스임박")
DOC_GAP = 0.02


# ============================================================
# 문서 규칙 (워커 프로세스 안에서만 바꿔 끼운다)
# ============================================================
def drop_wide_imminent(features: pd.DataFrame, limit: float = DOC_GAP) -> pd.DataFrame:
    """간격(|EMA10 − EMA20| / EMA20, 라벨이 붙은 끝난 봉)이 limit 이상인 임박 라벨을 지운다 — 임박 규칙의 간격만 limit 으로."""
    out = features.copy()
    for pcol, gcol in GAP_COLS.items():
        if pcol not in out.columns or gcol not in out.columns:
            continue
        wide = pd.to_numeric(out[gcol], errors="coerce").abs() >= limit

        def strip(s: Any) -> Any:
            if not isinstance(s, str) or not s:
                return s
            return ", ".join(t for t in (x.strip() for x in s.split(",")) if t and t not in IMMINENT)

        out.loc[wide, pcol] = out.loc[wide, pcol].map(strip)
    return out


def _trend_bonus(row: pd.Series) -> float:
    """candidate_selector._score_risk_quality 의 주·월봉 정배열 가점 (같은 계산)."""
    weekly_pat, monthly_pat = str(row.get("주봉패턴", "")), str(row.get("월봉패턴", ""))
    b = 0.10 if "정배열(완전)" in monthly_pat else 0.05 if "정배열(눌림목)" in monthly_pat else \
        0.02 if "정배열(형성중)" in monthly_pat else 0.0
    return b + (0.05 if "정배열(완전)" in weekly_pat else 0.025 if "정배열(눌림목)" in weekly_pat else 0.0)


def check_risk_copy() -> None:
    """_risk_without_trend + 정배열 가점 = 원래 리스크 점수인지 C 캐시 하루치로 맞춰 본다 (잘리지 않는 행만)."""
    import paper_trading.candidate_selector as cs

    f = pd.read_parquet(feature_cache_dir(C_KEY) / f"{gc.target_dates()[-1]}.parquet")
    f["전략구분"] = np.where(np.arange(len(f)) % 2 == 0, "📉 바닥반등", "📈 모멘텀")
    bad = 0
    for _, row in f.iterrows():
        orig, mine, b = cs._score_risk_quality(row), _risk_without_trend(row), _trend_bonus(row)
        if 0.0 < mine and mine + b < 1.0 and abs(orig - (mine + b)) > 1e-12:
            bad += 1
    if bad:
        raise SystemExit(f"RTREND 복사본이 원래 리스크 점수와 {bad}행 다르다 — 멈춤")
    print(f"  RTREND 복사본 확인: {len(f)}행 같음", flush=True)


def _risk_without_trend(row: pd.Series) -> float:
    """candidate_selector._score_risk_quality 에서 주·월봉 정배열 가점만 뺀 것 (그 밖은 같은 계산 — run() 이 check_risk_copy 로 확인)."""
    import paper_trading.candidate_selector as cs

    f = cs._safe_float
    score = 0.0
    strategy = str(row.get("전략구분", ""))
    atr_pct, atr_med = f(row.get("ATR%")), f(row.get("atr_med_252"))
    if atr_pct > 0 and atr_med > 0:
        vol_ratio = atr_pct / atr_med
        if vol_ratio < 0.8:
            score += 0.3
        elif vol_ratio < 1.0:
            score += 0.15
    w52 = f(row.get("52주포지션"), 0.5)
    if "바닥반등" in strategy:
        if 0.10 <= w52 <= 0.40:
            score += 0.2
    elif "모멘텀" in strategy:
        if 0.55 <= w52 <= 0.85:
            score += 0.2
    if f(row.get("fund_roe")) > 0.10:
        score += 0.15
    if f(row.get("fund_debt_to_equity"), 999) < 100:
        score += 0.1
    if f(row.get("fund_institutional_holders_pct")) > 0.60:
        score += 0.1
    short_pct = f(row.get("fund_short_pct_float"))
    if short_pct < 0.05:
        score += 0.05
    elif short_pct > 0.15:
        score -= 0.05
    return max(0.0, min(score, 1.0))


def apply_doc_rule(test: str) -> None:
    """문서 규칙 하나를 이 프로세스에서 켠다 (모듈 속성 바꿔치기). 피처 계산 · 캐시 파일은 건드리지 않는다."""
    import paper_trading.candidate_selector as cs
    import screener.backtest as sb
    from screener import config as cfg

    if test == "GI2":
        orig = sb.liquidity_filter
        sb.liquidity_filter = lambda features, *a, **k: orig(drop_wide_imminent(features), *a, **k)
    elif test == "RW":
        for regime, w in DOC_REGIME_WEIGHTS.items():
            cfg.CANDIDATE_REGIME_WEIGHTS[regime].update(w)      # 같은 dict 객체 (candidate_selector 가 들고 있음)
    elif test == "MACD":
        orig_timing = cs._score_entry_timing

        def timing(row: pd.Series) -> float:
            t = orig_timing(row)
            if "모멘텀" in str(row.get("전략구분", "")) or "바닥반등" not in str(row.get("전략구분", "")):
                return t
            macd_hist = cs._safe_float(row.get("macd_hist"))
            close = cs._safe_float(row.get("현재가격", row.get("close")))
            code_pt = 0.1 if close > 0 and macd_hist / close > cs._BOTTOM_MACD_REL_MIN else 0.0
            doc_pt = 0.1 if macd_hist > -0.5 else 0.0
            return min(t - code_pt + doc_pt, 1.0)             # 바닥반등 최대 0.95 라 잘리는 일 없음

        cs._score_entry_timing = timing
    elif test == "F52W":
        cs.CANDIDATE_BOTTOM_MAX_52W_POS = float("inf")
    elif test == "FADX":
        cs.ADX_BUY_MIN = float("-inf")
    elif test == "FRSI":
        orig_filters = cs._apply_hard_filters

        def filters(df: pd.DataFrame, current_holdings: list[dict[str, Any]], rsi_max: float | None = None):
            # 데드존(30 ≤ RSI < 35, 바닥반등)에 걸리는 행만 RSI 를 29.999 로 넘겨 그 필터만 비키게 하고,
            # 통과한 행은 원래 값으로 돌려준다 (다른 필터의 RSI 기준 75 · 70 은 두 값 모두 같은 쪽)
            if not df.index.is_unique:
                raise SystemExit("후보표 인덱스가 겹친다 — FRSI 래퍼를 쓸 수 없음")
            shifted = df.copy()
            strat = shifted.get("전략구분", pd.Series("", index=shifted.index)).astype(str)
            rsi = pd.to_numeric(shifted.get("RSI", pd.Series(50, index=shifted.index)), errors="coerce").fillna(50)
            dead = strat.str.contains("바닥", na=False) & (rsi >= 30) & (rsi < 35)
            if dead.any():
                shifted.loc[dead, "RSI"] = 29.999
            passed, rej = orig_filters(shifted, current_holdings, rsi_max)
            return df.loc[passed.index], rej

        cs._apply_hard_filters = filters
    elif test == "RTREND":
        cs._score_risk_quality = _risk_without_trend
    elif test == "CONF":
        cs._BULLISH_PATTERNS = set(DOC_BULLISH)
    elif test == "LIQ":
        sb.liquidity_filter = liquidity_doc
    elif test != "CODE":
        raise ValueError(test)


def _engine(test: str, ef: bool) -> tuple[str, bool, iv.RunResult, float]:
    """PT-1 엔진 한 번 (BASE 경로 · 가격 · 종목 목록 고정, 지금 코드의 피처 캐시 = C)."""
    t0 = time.time()
    apply_doc_rule(test)
    with config_overrides(EF if ef else {}):
        res = iv.run_overlay(iv.Overlay("BASE"))
    return test, ef, res, time.time() - t0


# ============================================================
# 참고 통계 (판정 없음)
# ============================================================
def imminent_counts(dates: list[str]) -> pd.DataFrame:
    """종목-일마다 임박 라벨 수 — 5% (지금) vs 2% (문서). 25일에 하나씩 본다."""
    rows = []
    for d in dates[::25]:
        f = pd.read_parquet(feature_cache_dir(C_KEY) / f"{d}.parquet")
        g = drop_wide_imminent(f)
        for pcol, tf in gc.PATTERN_COLS.items():
            for tok in IMMINENT:
                has = lambda s: isinstance(s, str) and tok in [x.strip() for x in s.split(",")]  # noqa: E731
                rows.append({"date": d, "봉": tf, "라벨": tok, "5%": int(f[pcol].map(has).sum()),
                             "2%": int(g[pcol].map(has).sum())})
    return pd.DataFrame(rows).groupby(["봉", "라벨"])[["5%", "2%"]].mean().round(1)


def doji_rates() -> pd.DataFrame:
    """일봉 몸통 ÷ 위아래 폭 < 10% (지금 코드) vs < 5% (예전 문서) 가 붙는 종목-일 비율 (고정 종목 · 개발 구간)."""
    raw = pd.read_parquet(isb.PRICE_CACHE)
    tickers = [t for t in iv._pinned_tickers() if t in raw.columns.get_level_values(0) and t != "SPY"]
    ratios = []
    for t in tickers:
        px = raw[t].loc[RUN_KW["start_date"]:RUN_KW["end_date"]].dropna(subset=["Open", "High", "Low", "Close"])
        rng = px["High"] - px["Low"]
        body = (px["Close"] - px["Open"]).abs()
        r = (body / rng.where(rng > 0)).dropna()
        ratios.append(r)
    allr = pd.concat(ratios)
    return pd.DataFrame({"기준": ["몸통 < 폭의 10% (지금 코드)", "몸통 < 폭의 5% (예전 문서)"],
                         "종목-일 비율": [float((allr < 0.10).mean()), float((allr < 0.05).mean())],
                         "종목 하나 기준 1년에": [float((allr < 0.10).mean() * 252), float((allr < 0.05).mean() * 252)]})


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
    tests = " / ".join(f"{k}: 문서 {d} vs 코드 {c}" for k, (d, c, _) in TESTS.items())
    memo = (f"[{RUN_KW['start_date']}~{RUN_KW['end_date']}, PT-1 엔진 BASE 설정 · 가격 · 종목 목록 고정, 피처 캐시 C({C_KEY}) 읽기만] "
            f"문서와 코드가 다른 PT-1 규칙 {len(TESTS)}곳을 하나씩 문서대로 — {tests}. "
            f"시험마다 ΔSharpe(문서−코드), 20일 블록 부트스트랩 {N_BOOT:,}회. 문서가 확실히 좋음 = 재표본 중 Δ≤0 비율 < "
            f"{ALPHA_ONE_SIDED:.4%} (8건 본페로니, 한쪽) → 문서대로 바꾸자고 제안, 그 밖 = 코드 유지 (문서를 코드에 맞춤). "
            "참고(판정 없음): 균등 크기 · 거래 겹침 · 임박 라벨 수 · 도지 비율."
            + (" v1 결과를 본 뒤 찾은 차이라 기준은 v1 과 같게 둔다 (낮추지 않음)." if FAMILY.endswith("v2") else ""))
    log_trials(STEP, FAMILY, len(TESTS), [{"test_id": "(등록)", "판정": "등록", "메모": memo}])
    print(f"등록: {FAMILY} {len(TESTS)}건", flush=True)


def run(workers: int) -> None:
    if not _registered():
        raise SystemExit("먼저 --register (결과 보기 전에 기준 등록)")
    code = bt._code_hash()
    if code != CODE_HASH or gc.cache_key(code) != C_KEY:
        raise SystemExit(f"코드 지문 {code} · 캐시 {gc.cache_key(code)} ≠ {CODE_HASH} · {C_KEY} — 시험한 C 코드가 아니라 멈춤")
    iv.ensure_pinned()
    dates = gc.target_dates()
    n_cache = len(gc.cached(C_KEY, dates))
    print(f"[doc] 피처 날짜 {len(dates)}일 · C 캐시에 {n_cache}일 · 시험 {len(TESTS)}건 · 워커 {workers}개", flush=True)
    if n_cache != len(dates):
        raise SystemExit("C 캐시가 다 있지 않다 — 엔진이 피처를 새로 계산하게 되므로 멈춤")
    check_risk_copy()
    t0 = time.time()
    jobs = [(t, ef) for t in ["CODE", *TESTS] for ef in (False, True)]
    runs: dict[tuple[str, bool], iv.RunResult] = {}
    # 시험마다 새 프로세스 (문서 규칙을 모듈에 바꿔 끼우므로 프로세스를 다시 쓰면 다음 시험에 남는다)
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
        futs = [ex.submit(_engine, t, ef) for t, ef in jobs]
        for fut in as_completed(futs):
            test, ef, res, sec = fut.result()
            runs[(test, ef)] = res
            print(f"  {test}{' · 균등 크기' if ef else ''} 끝 · 거래 {len(res.trades)} · {sec:.0f}초 · "
                  f"{len(runs)}/{len(jobs)}", flush=True)
    if len(gc.cached(C_KEY, dates)) != n_cache:
        print("  ⚠️ 피처 캐시 파일 수가 바뀌었다 (새로 계산한 날이 있음)", flush=True)
    compare(runs, dates)
    print(f"전체 {gc._hm(time.time() - t0)}", flush=True)


def compare(runs: dict[tuple[str, bool], iv.RunResult], dates: list[str]) -> None:
    print("===== 비교 · 판정", flush=True)
    eq = {f"{t}{'_EF' if ef else ''}": r.equity for (t, ef), r in runs.items()}
    saved = pd.read_parquet(C_EQUITY)
    for k, src in (("CODE", "NEW"), ("CODE_EF", "NEW_EF")):
        diff = saved[src].reindex(eq[k].index).sub(eq[k]).abs().max()
        print(f"  {k} vs 저장된 C 곡선 ({src}) 최대 차이 {diff:.6f}", flush=True)
        if not diff < 1e-6:
            raise SystemExit("지금 코드 결과가 C 시험 결과와 다르다 — 비교를 믿을 수 없어 멈춤")
    raw = pd.read_parquet(isb.PRICE_CACHE)
    idx = eq["CODE"].index
    spy = raw["SPY"]["Close"].reindex(idx)
    eq["SPY"] = spy / spy.iloc[0] * float(eq["CODE"].iloc[0])
    pd.DataFrame(eq).to_parquet(EQUITY_OUT)
    rets = {k: v.pct_change().dropna() for k, v in eq.items()}

    rows = []
    base = gc._stats(runs[("CODE", False)].equity, runs[("CODE", False)].trades)
    base_ef = sharpe(rets["CODE_EF"].tolist())
    log_rows = []
    for t, (doc, code_txt, where) in TESTS.items():
        st = gc._stats(runs[(t, False)].equity, runs[(t, False)].trades)
        m, _ = iv._cmp(rets[t], rets["CODE"], N_BOOT)
        e, _ = iv._cmp(rets[f"{t}_EF"], rets["CODE_EF"], N_BOOT)
        p = m["p_개선아님"]
        verdict = "문서가 확실히 좋음" if p < ALPHA_ONE_SIDED else "코드 유지"
        ov, n_code, n_doc = gc._overlap(runs[("CODE", False)].trades, runs[(t, False)].trades)
        rows.append({"test": t, "문서": doc, "코드": code_txt, "문서 위치": where,
                     "Sharpe_문서": st["Sharpe"], "Sharpe_코드": base["Sharpe"], "ΔSharpe": m["ΔSharpe"],
                     "하한": m["ΔSharpe_하한"], "상한": m["ΔSharpe_상한"], "p_문서나음아님": p,
                     "구간_문서나음": f"{m['구간_개선']}/{m['구간_수']}",
                     "CAGR_문서": st["CAGR"], "MDD_문서": st["MDD"], "거래_문서": st["거래수"], "겹침": ov,
                     "균등_ΔSharpe": e["ΔSharpe"], "균등_하한": e["ΔSharpe_하한"], "균등_상한": e["ΔSharpe_상한"],
                     "판정": verdict})
        log_rows.append({"test_id": t, "판정": verdict,
                         "메모": (f"문서 {doc} vs 코드 {code_txt}: Sharpe {st['Sharpe']:.2f} vs {base['Sharpe']:.2f}, ΔSharpe "
                                f"{m['ΔSharpe']:+.2f} [{m['ΔSharpe_하한']:+.2f}, {m['ΔSharpe_상한']:+.2f}] p(문서 나음 아님) {p:.4f} "
                                f"(기준 < {ALPHA_ONE_SIDED:.4f}), 거래 {st['거래수']} (겹침 {ov}), 균등 크기 Δ {e['ΔSharpe']:+.2f}")})
        print(f"  {t}: {verdict} — {log_rows[-1]['메모']}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    todo = [r for r in log_rows if not _logged(r["test_id"])]
    if todo:
        log_trials(STEP, FAMILY, len(TESTS), todo)
    spy_r = rets["SPY"]
    spy_st = {"Sharpe": sharpe(spy_r.tolist()), "CAGR": cagr(spy_r.tolist()), "MDD": max_drawdown(spy_r.tolist())}
    if FAMILY.endswith("v2"):
        write_report(out, base, base_ef, spy_st, None, None, liquidity_counts(dates))
        return
    print("  임박 라벨 수 · 도지 비율 (참고) 계산 중...", flush=True)
    write_report(out, base, base_ef, spy_st, imminent_counts(dates), doji_rates())


def liquidity_counts(dates: list[str]) -> pd.DataFrame:
    """유동성 필터를 통과하는 종목 수 — 지금 코드 vs 문서 (25일에 하루씩, 하루 평균)."""
    from screener.processing import liquidity_filter

    rows = []
    for d in dates[::25]:
        f = pd.read_parquet(feature_cache_dir(C_KEY) / f"{d}.parquet")
        rows.append({"전체": len(f), "코드": len(liquidity_filter(f)), "문서": len(liquidity_doc(f))})
    return pd.DataFrame(rows).mean().round(0).to_frame("하루 평균 종목 수")


def _f(v: float, pct: bool = False, sign: bool = True) -> str:
    if pct:
        return f"{v:+.1%}" if sign else f"{v:.1%}"
    return f"{v:+.2f}" if sign else f"{v:.2f}"


def write_report(out: pd.DataFrame, base: dict[str, Any], base_ef: float, spy: dict[str, float],
                 imm: pd.DataFrame | None, doji: pd.DataFrame | None, liq: pd.DataFrame | None = None) -> None:
    lines = [
        f"# 문서 기준 vs 지금 코드 — PT-1 10년 ({FAMILY})",
        "",
        f"*{pd.Timestamp.now(tz='America/Toronto'):%Y-%m-%d %H:%M} ET · scripts/doc_vs_code_check.py · 사전 등록 TRIAL_LOG {FAMILY} "
        f"{len(TESTS)}건*",
        "",
        "문서(patterns_reference · screener_filtering_guide · paper_trading_candidate_selection · architecture)와 코드가 다른 곳 중 "
        "PT-1 이 사는 종목을 바꾸는 곳을 하나씩 문서대로 바꿔 돌렸다. 같은 10년 · 같은 가격 · 같은 종목 목록 · 같은 피처(C 캐시, 읽기만).",
        "",
        f"판정 기준 (등록): 재표본 중 Δ ≤ 0 비율 < {ALPHA_ONE_SIDED:.4%} (8건 본페로니, 한쪽) 이면 '문서가 확실히 좋음', "
        "그 밖은 '코드 유지'.",
        "",
        f"지금 코드: Sharpe {base['Sharpe']:.2f} · 연 {base['CAGR']:+.1%} · 최대낙폭 {base['MDD']:.1%} · 거래 {base['거래수']} "
        f"(균등 크기 Sharpe {base_ef:.2f}) · SPY {spy['Sharpe']:.2f} · 연 {spy['CAGR']:+.1%} · 최대낙폭 {spy['MDD']:.1%}",
        "",
        "| 시험 | 문서 | 코드 | Sharpe 문서 | ΔSharpe (문서−코드) | 95% 구간 | p(문서 나음 아님) | 거래 (겹침) | 균등 크기 Δ | 판정 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in out.itertuples():
        lines.append(f"| {r.test} | {r.문서} | {r.코드} | {r.Sharpe_문서:.2f} | {r.ΔSharpe:+.2f} | [{r.하한:+.2f}, {r.상한:+.2f}] | "
                     f"{r.p_문서나음아님:.3f} | {r.거래_문서} ({r.겹침}) | {r.균등_ΔSharpe:+.2f} | {r.판정} |")
    if liq is not None:
        lines += [
            "",
            "## 유동성 필터를 통과하는 종목 수 (참고 — 하루 평균, 25일에 하루씩)",
            "",
            "| | 종목 수 |",
            "|---|---|",
            *[f"| {k} | {v:.0f} |" for k, v in liq["하루 평균 종목 수"].items()],
            "",
            "## 구현",
            "",
            "- LIQ: screener.backtest 의 liquidity_filter 를 시장별 quantile(LIQUIDITY_QUANTILE) 이상(하위 25% 만 뺌)으로. "
            "거래대금 $5M 하한 · 회전율 · 화이트리스트는 같음",
            "- 지금 코드 결과는 C 시험(GOLDENCROSS_PT1_C)의 저장된 곡선과 같음을 확인 (차이 0)",
        ]
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"보고서: {REPORT.relative_to(ROOT)}", flush=True)
        return
    lines += [
        "",
        "## 교차 임박 라벨 수 (참고 — 하루 평균 종목 수, 25일에 하루씩)",
        "",
        "| 봉 | 라벨 | 5% (지금) | 2% (문서) |",
        "|---|---|---|---|",
        *[f"| {b} | {lab} | {row['5%']:.1f} | {row['2%']:.1f} |" for (b, lab), row in imm.iterrows()],
        "",
        "## 도지 (참고 — 점수 · PT-1 에 안 씀, 화면 표시만)",
        "",
        "| 기준 | 종목-일 비율 | 종목 하나가 1년에 |",
        "|---|---|---|",
        *[f"| {r.기준} | {r._2:.1%} | {r._3:.0f}번 |" for r in doji.itertuples()],
        "",
        "## 시험별 구현 (피처 계산 뒤 단계만 바꿈)",
        "",
        "- GI2: 피처를 읽은 뒤(유동성 필터 직전) 일·주·월봉 패턴 칸에서 간격(라벨 봉의 MA갭)이 2% 이상인 골든·데드크로스임박 토큰을 지움",
        "- RW: config.CANDIDATE_REGIME_WEIGHTS 를 문서 표로 (같은 dict 객체)",
        "- MACD: 바닥반등 타이밍 점수에서 코드의 MACD 가점(주가 대비 > −1%)을 빼고 문서 가점(히스토그램 > −0.5)을 더함",
        "- F52W · FADX: candidate_selector 의 기준값을 무한대로 (필터가 아무것도 안 거름)",
        "- FRSI: 하드 필터에 넣는 사본에서 데드존 행의 RSI 만 29.999 로, 통과한 행은 원래 값으로 돌려줌",
        "- RTREND: 리스크 점수를 정배열 가점만 뺀 같은 계산으로",
        "- CONF: candidate_selector._BULLISH_PATTERNS 를 문서의 9개로",
        "- 지금 코드 결과는 C 시험(GOLDENCROSS_PT1_C)의 저장된 곡선과 같음을 확인 (차이 0)",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"보고서: {REPORT.relative_to(ROOT)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--v2", action="store_true", help="유동성 필터 시험 (doc_vs_code_v2)")
    args = ap.parse_args()
    if args.v2:
        use_v2()
    if args.register:
        register()
    if args.run:
        run(args.workers)


if __name__ == "__main__":
    main()
