#!/usr/bin/env python3
"""골든크로스 통일 3단계: 새 교차 라벨(차트 EMA10 주황 · EMA20 보라)로 PT-1 10년 백테스트 — 예전 라벨과 비교.

docs/goldencross.md 3단계. 1단계에서 바꾼 교차 라벨(일·주·월봉)이 ★ 점수 · CCS 를 거쳐 PT-1 이 사는 종목을 바꾼다.
같은 10년(2016-01-04 ~ 2025-09-30) · 같은 PT-1 규칙 · 같은 가격(디스크 고정) · 같은 종목 목록(고정)에서 라벨만 다르게 돌린다.

사전 등록 (TRIAL_LOG family goldencross_v1, 1건 — 결과 계산 전에 --register):
  GC-PT1 = 새 라벨 PT-1 vs 예전 라벨 PT-1 (BASE = 저장된 기준선), 일간 수익 Sharpe 차이, 20일 블록 부트스트랩 10,000회.
  판정: 95% 구간 상한 < 0 → '확실히 나빠짐' (멈추고 점수 가중치부터 다시 본다). 그 밖 → '비슷함' (통일을 위해 진행).
  좋아져도 채택 근거로 쓰지 않는다 — 이 변경은 성과가 아니라 라벨 · 계산 · 차트를 맞추기 위한 것.
  참고 (판정 없음): 빈자리 균등 크기(PT1_SIZING equal_fixed, 실계좌 $5,000 보기와 같은 셈) 전후 · 거래 겹침 · 연도별 ·
  라벨이 붙은 종목-일 수.

순서 (--run): 2단계(chart_ma_tv_check.py, 실패하면 멈춤) →
  1/4 예전 라벨 BASE 재현 — 1단계 전 코드 지문으로 예전 캐시를 읽어 저장된 기준선과 같은지 (다르면 멈춤)
  2/4 새 라벨 캐시 만들기 — 날짜를 워커 수만큼 겹치지 않게 나눠 동시에 (이어 하기 됨: 이미 만든 날은 건너뜀)
  3/4 새 라벨 PT-1
  4/4 비교 · 판정 · 보고서 (docs/quant_improvement/tier3/GOLDENCROSS_PT1.md · goldencross_pt1.csv)
예전 캐시(data/cache/features/4fc9fd085b13_v1)는 읽기만 한다. 새 캐시는 새 이름의 폴더에 생긴다.

B안 (--variant b, 2026-10-27 · TRIAL_LOG family goldencross_v2 · 1건): 위(A) 코드에서 '주·월봉 정배열(완전/눌림목)이면
  골든크로스 · 골든크로스임박 태그를 지우는' 예전 규칙만 뺀 코드. 순서 · 판정은 같다 (B vs 예전 라벨).
  A 결과 파일 · 캐시(d0cfda835240_v1)는 읽기만 하고 참고로 같이 보여 준다 (B vs A — 판정 없음).
  결과: GOLDENCROSS_PT1_B.md · goldencross_pt1_b.csv · goldencross_pt1_b_equity.parquet (A 파일은 그대로).

C (--variant c, 2026-10-27 사용자 결정 · TRIAL_LOG family goldencross_v3 · 1건): B 코드에 더해 모든 패턴을 차트와 같은 바탕으로 —
  가격 패턴 · 캔들 · 정배열 · 차트반전도 분할만 반영한 가격 · 주·월봉 끝난 봉만 · Pine 계산, 정배열은 차트 4선 EMA10/20/100/200
  (있는 선으로만), 주·월봉은 상장 이후 전체 기록 (얼린 차트 가격 파일 — scripts/build_chart_prices.py). 월말이 주말이면
  그 전 금요일 장 뒤에 끝난 봉. 순서 · 판정은 같다 (C vs 예전 라벨). B · A 결과 파일(goldencross_pt1_b_*)은 읽기만 하고 참고로.
  2단계 대신 scripts/chart_tv_check_c.py (4선 · 캔들 · 정배열 상태 OCR 대조, 실패하면 멈춤).
  결과: GOLDENCROSS_PT1_C.md · goldencross_pt1_c.csv · goldencross_pt1_c_equity.parquet.

사용법: PYTHONPATH=.:src .venv/bin/python scripts/goldencross_pt1_check.py --register
        PYTHONPATH=.:src .venv/bin/python scripts/goldencross_pt1_check.py --run --workers 6
        PYTHONPATH=.:src .venv/bin/python scripts/goldencross_pt1_check.py --variant b --register
        PYTHONPATH=.:src .venv/bin/python scripts/goldencross_pt1_check.py --variant b --run --workers 6 --skip-step2
        PYTHONPATH=.:src .venv/bin/python scripts/goldencross_pt1_check.py --variant c --register
        PYTHONPATH=.:src .venv/bin/python scripts/goldencross_pt1_check.py --variant c --run --workers 6
진행 보기: bash scripts/watch_backtest.sh output/logs/goldencross_pt1_check_<날짜>_<시각>.log
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import isb_engine as isb  # noqa: E402
import isb_vs_pt1 as iv  # noqa: E402
import paper_trading.backtest as bt  # noqa: E402
from paper_trading.config_override import config_overrides  # noqa: E402
from paper_trading.evaluation import cagr, max_drawdown, sharpe  # noqa: E402
from research_utils import RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, log_trials, total_trials  # noqa: E402
from data.fetch import chart_prices_path  # noqa: E402
from screener.cache import feature_cache_dir  # noqa: E402
from validate_upside_walkforward import RUN_KW  # noqa: E402

FAMILY = "goldencross_v1"
STEP = "goldencross"
TEST = "GC-PT1"
N_BOOT = 10_000
OLD_COMMIT = "6c8188fc"   # 1단계 전 코드 (예전 라벨) — 예전 10년 캐시의 지문
OLD_SOURCES = ("screener/features.py", "screener/alpha_model.py", "screener/patterns.py", "screener/fundamentals.py",
               "screener/processing.py", "screener/signals.py", "screener/backtest.py", "screener/config.py")
OLD_KEY = "4fc9fd085b13"
EF = {"PT1_SIZING": "equal_fixed"}
RUNS = {"OLD": "예전 라벨", "NEW": "새 라벨", "OLD_EF": "예전 라벨 · 균등 크기", "NEW_EF": "새 라벨 · 균등 크기"}
REPORT = TIER3_DIR / "GOLDENCROSS_PT1.md"
OUT_CSV = TIER3_DIR / "goldencross_pt1.csv"
EQUITY_OUT = RESEARCH_DIR / "goldencross_pt1_equity.parquet"
# B안 (--variant b) 일 때 A = 위 결과 파일들 (goldencross_v1) — 읽기만 한다
VARIANT = "a"
A_KEY = "d0cfda835240"
A_CSV, A_EQUITY = OUT_CSV, EQUITY_OUT
A_RUNS = {"A": "새 라벨 A", "A_EF": "새 라벨 A · 균등 크기"}
# C 일 때 B · A = B안 결과 파일들 (goldencross_v2 — A 곡선도 그 파일에 있다) — 읽기만 한다
B_KEY = "0e8a90945db7"
B_CSV = TIER3_DIR / "goldencross_pt1_b.csv"
B_EQUITY = RESEARCH_DIR / "goldencross_pt1_b_equity.parquet"
B_RUNS = {"B": "B안", "B_EF": "B안 · 균등 크기", "A": "새 라벨 A", "A_EF": "새 라벨 A · 균등 크기"}
TOKENS_C = ("정배열(완전)", "정배열(눌림목)", "정배열(형성중)", "차트반전", "이중바닥", "컵앤핸들", "강세잉걸핑", "도지")
C_TV_CHECK_NOTE = ("scripts/chart_tv_check_c.py OCR 대조 (2026-10-27 21:15, S&P 143종목) 기준 미달 — 일봉 EMA100 · EMA200 98.6% "
                   "(기준 99%) · 주봉 EMA200 97.2% (기준 98%), 나머지(캔들 · 교차 · 정배열 상태 100%, 짧은 선) 통과. 원인: OCR 오독 1건(APP "
                   "일봉 EMA100) · TradingView 가 분사 · 합병만큼 과거 가격을 깎는 종목 3개(APTV · FLEX · RTX) · 기록 차이 2개(FISV · VST) — "
                   "계산식 차이 0. 사용자 결정으로 진행 (2026-10-27).")
POLL_SECONDS = 30
TOKENS = ("골든크로스", "골든크로스임박", "데드크로스", "데드크로스임박")
PATTERN_COLS = {"일봉패턴": "일봉", "주봉패턴": "주봉", "월봉패턴": "월봉"}


# ============================================================
# 캐시 이름 (엔진과 같은 계산)
# ============================================================
def old_code_hash() -> str:
    """1단계 전 코드의 지문 (paper_trading.backtest._code_hash 와 같은 계산, 그때의 파일 목록)."""
    h = hashlib.md5()
    for rel in OLD_SOURCES:
        raw = subprocess.run(["git", "show", f"{OLD_COMMIT}:src/{rel}"], cwd=ROOT, capture_output=True,
                             check=True).stdout
        h.update(bt._hashed_bytes(rel, raw))
    return h.hexdigest()[:8]


def _engine_tickers() -> list[str]:
    """엔진이 받는 종목 목록 (고정 종목 목록 + SPY, max_tickers 까지)."""
    tickers = iv._pinned_tickers()
    tickers = tickers if "SPY" in tickers else ["SPY"] + tickers
    return ["SPY"] + [t for t in tickers if t != "SPY"][: RUN_KW["max_tickers"]]


def cache_key(code: str) -> str:
    """엔진의 피처 캐시 키 (고정 종목 목록 + SPY · 13y · 펀더멘탈 없음)."""
    tickers = _engine_tickers()
    key = ",".join(sorted(tickers)) + "|" + RUN_KW["period"] + f"|fund=0|code={code}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def target_dates() -> list[str]:
    """엔진이 피처를 계산하는 날 = 예전 캐시에 있는 개발 구간 날짜 (시뮬레이션 마지막 날 전날까지)."""
    end = RUN_KW["end_date"]
    return sorted(p.stem for p in feature_cache_dir(OLD_KEY).glob("????-??-??.parquet")
                  if RUN_KW["start_date"] <= p.stem < end)


def cached(key: str, dates: list[str]) -> set[str]:
    have = {p.stem for p in feature_cache_dir(key).glob("????-??-??.parquet")}
    return have & set(dates)


def use_variant_b() -> None:
    """B안: 등록 · 결과 행 · 보고서 · 결과 파일 이름만 바꾼다 (순서 · 판정은 같음). 워커는 이 값들을 쓰지 않는다."""
    global VARIANT, FAMILY, TEST, RUNS, REPORT, OUT_CSV, EQUITY_OUT
    VARIANT, FAMILY, TEST = "b", "goldencross_v2", "GC-PT1-B"
    RUNS = {"OLD": "예전 라벨", "NEW": "새 라벨 B", "OLD_EF": "예전 라벨 · 균등 크기", "NEW_EF": "새 라벨 B · 균등 크기"}
    REPORT = TIER3_DIR / "GOLDENCROSS_PT1_B.md"
    OUT_CSV = TIER3_DIR / "goldencross_pt1_b.csv"
    EQUITY_OUT = RESEARCH_DIR / "goldencross_pt1_b_equity.parquet"


def use_variant_c() -> None:
    """C: 모든 패턴을 차트 바탕으로 (등록 · 결과 행 · 보고서 · 결과 파일 이름만 바뀐다 — 순서 · 판정은 같음)."""
    global VARIANT, FAMILY, TEST, RUNS, REPORT, OUT_CSV, EQUITY_OUT
    VARIANT, FAMILY, TEST = "c", "goldencross_v3", "GC-PT1-C"
    RUNS = {"OLD": "예전 라벨", "NEW": "새 라벨 C", "OLD_EF": "예전 라벨 · 균등 크기", "NEW_EF": "새 라벨 C · 균등 크기"}
    REPORT = TIER3_DIR / "GOLDENCROSS_PT1_C.md"
    OUT_CSV = TIER3_DIR / "goldencross_pt1_c.csv"
    EQUITY_OUT = RESEARCH_DIR / "goldencross_pt1_c_equity.parquet"


# ============================================================
# 워커 (spawn — 함수는 모듈 맨 위에)
# ============================================================
def _engine(name: str, code: str | None, overrides: dict[str, Any]) -> tuple[str, iv.RunResult, float]:
    """PT-1 엔진 한 번 (BASE 경로 — 래퍼 없음, 가격 · 종목 목록 고정). code 를 주면 그 지문의 캐시를 읽는다."""
    t0 = time.time()
    if code:
        bt._code_hash = lambda: code
    with config_overrides(overrides):
        res = iv.run_overlay(iv.Overlay("BASE"))
    return name, res, time.time() - t0


def _warm(start: str, end: str) -> tuple[str, str, float]:
    """새 라벨 피처 캐시를 [start, end) 날짜에 만든다 (매매 결과는 버림). 가격 · 종목 목록은 BASE 와 같게 고정."""
    t0 = time.time()
    ctx = iv.RunCtx(iv.Overlay("BASE"), None)
    with iv._patched(ctx), contextlib.redirect_stdout(io.StringIO()):
        bt.run_paper_trading_backtest(**{**RUN_KW, "start_date": start, "end_date": end})
    return start, end, time.time() - t0


def _hm(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600}시간 {sec % 3600 // 60:02d}분" if sec >= 3600 else f"{sec // 60}분"


# ============================================================
# 단계
# ============================================================
def step2() -> None:
    print("===== 2단계 OCR 대조 (chart_ma_tv_check.py)", flush=True)
    rc = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts" / "chart_ma_tv_check.py")], cwd=ROOT).returncode
    if rc != 0:
        raise SystemExit(f"2단계 통과 못 함 (종료 코드 {rc}) — 3단계로 가지 않는다. "
                         "docs/quant_improvement/tier3/CHART_MA_TV_CHECK.md 를 본다")


def stage_old(workers: int) -> dict[str, iv.RunResult]:
    print("===== 3단계 1/4 예전 라벨 BASE 재현 (예전 캐시 읽기만)", flush=True)
    code = old_code_hash()
    if cache_key(code) != OLD_KEY:
        raise SystemExit(f"예전 코드 지문 {code} → 캐시 {cache_key(code)} ≠ {OLD_KEY} — 멈춤")
    n_before = len(list(feature_cache_dir(OLD_KEY).glob("*.parquet")))
    out: dict[str, iv.RunResult] = {}
    with ProcessPoolExecutor(max_workers=min(2, workers)) as ex:
        futs = [ex.submit(_engine, "OLD", code, {}), ex.submit(_engine, "OLD_EF", code, EF)]
        for fut in futs:
            name, res, sec = fut.result()
            out[name] = res
            print(f"  {RUNS[name]} 끝 · 거래 {len(res.trades)} · {sec:.0f}초 · 메모리 {res.peak_rss_gb:.1f}GB", flush=True)
    if len(list(feature_cache_dir(OLD_KEY).glob("*.parquet"))) != n_before:
        raise SystemExit("예전 캐시 파일 수가 바뀌었다 — 예전 라벨을 새로 계산한 날이 있다는 뜻이라 멈춤")
    err = iv.check_base(out["OLD"])
    if err:
        raise SystemExit(f"{err} — 멈춤")
    return out


def stage_warm(workers: int, new_key: str, dates: list[str]) -> None:
    print(f"===== 3단계 2/4 새 라벨 캐시 만들기 ({len(dates)}일 · 워커 {workers}개 · 폴더 {new_key}_v1)", flush=True)
    todo = [d for d in dates if d not in cached(new_key, dates)]
    if not todo:
        print("  이미 다 있음", flush=True)
        return
    n = min(workers, len(todo))
    bounds = np.array_split(np.arange(len(todo)), n)
    nxt = {d: dates[i + 1] if i + 1 < len(dates) else RUN_KW["end_date"] for i, d in enumerate(dates)}
    chunks = [(todo[b[0]], nxt[todo[b[-1]]]) for b in bounds if len(b)]
    t0, done0 = time.time(), len(dates) - len(todo)
    print(f"  남은 {len(todo)}일을 {len(chunks)}구간으로: " + " · ".join(f"{a}~{b}" for a, b in chunks), flush=True)
    with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
        pending = {ex.submit(_warm, a, b) for a, b in chunks}
        finished = 0
        while pending:
            done, pending = wait(pending, timeout=POLL_SECONDS, return_when=FIRST_COMPLETED)
            for fut in done:
                a, b, sec = fut.result()          # 워커 오류면 여기서 멈춘다
                finished += 1
                print(f"  구간 {a}~{b} 끝 ({_hm(sec)}) · {finished}/{len(chunks)}", flush=True)
            have = len(cached(new_key, dates))
            el = time.time() - t0
            rate = (have - done0) / el if el > 0 else 0.0
            eta = (len(dates) - have) / rate if rate > 0 else float("nan")
            print(f"  새 캐시 {have}/{len(dates)}일 · 지난 {_hm(el)} · 남은 약 "
                  f"{_hm(eta) if np.isfinite(eta) else '계산 중'} [{have / len(dates) * 100:5.1f}%]", flush=True)
    missing = sorted(set(dates) - cached(new_key, dates))
    if missing:
        print(f"  ⚠️ 캐시가 안 생긴 날 {len(missing)}일 (피처가 비어 저장 안 됨일 수 있음): {missing[:5]}", flush=True)


def stage_new(workers: int, new_key: str, dates: list[str]) -> dict[str, iv.RunResult]:
    print("===== 3단계 3/4 새 라벨 PT-1 (10년)", flush=True)
    n_before = len(cached(new_key, dates))
    out: dict[str, iv.RunResult] = {}
    with ProcessPoolExecutor(max_workers=min(2, workers)) as ex:
        futs = [ex.submit(_engine, "NEW", None, {}), ex.submit(_engine, "NEW_EF", None, EF)]
        for fut in futs:
            name, res, sec = fut.result()
            out[name] = res
            print(f"  {RUNS[name]} 끝 · 거래 {len(res.trades)} · {sec:.0f}초 · 메모리 {res.peak_rss_gb:.1f}GB", flush=True)
    if len(cached(new_key, dates)) != n_before:
        print("  ⚠️ 3/4 에서 피처를 새로 계산한 날이 있다 (2/4 가 덜 끝났다는 뜻)", flush=True)
    return out


# ============================================================
# 비교
# ============================================================
def _stats(eq: pd.Series, trades: list[dict[str, Any]]) -> dict[str, Any]:
    r = eq.pct_change().dropna()
    rets = [float(t["return_pct"]) for t in trades if float(t.get("shares", 1) or 0) > 0]
    return {"Sharpe": sharpe(r.tolist()), "CAGR": cagr(r.tolist()), "MDD": max_drawdown(r.tolist()),
            "거래수": len(trades), "0주매수": sum(1 for t in trades if not float(t.get("shares", 1) or 0) > 0),
            "승률": float(np.mean([x > 0 for x in rets])) if rets else float("nan"),
            "거래당평균": float(np.mean(rets)) if rets else float("nan")}


def _yearly(eq: pd.Series) -> pd.Series:
    return eq.groupby(eq.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)


def _overlap(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> tuple[int, int, int]:
    ka = {(t["ticker"], t["entry_date"]) for t in a}
    kb = {(t["ticker"], t["entry_date"]) for t in b}
    return len(ka & kb), len(ka), len(kb)


def label_counts(new_key: str, dates: list[str]) -> pd.DataFrame:
    """종목-일마다 교차 라벨: 예전 캐시 vs 새 캐시, B안이면 A 캐시도 (하루 평균 몇 종목에 붙었나 · 골든 여부가 같은 비율)."""
    keys = {"예전": OLD_KEY, **({"A": A_KEY} if VARIANT == "b" else {}), **({"B": B_KEY} if VARIANT == "c" else {}),
            "새": new_key}
    tokens = TOKENS + (TOKENS_C if VARIANT == "c" else ())
    rows = []
    cols = ["티커", *PATTERN_COLS]
    for d in dates:
        try:
            fr = {k: pd.read_parquet(feature_cache_dir(v) / f"{d}.parquet", columns=cols).set_index("티커")
                  for k, v in keys.items()}
        except Exception:
            continue
        both = fr["예전"].index.intersection(fr["새"].index)
        for col, tf in PATTERN_COLS.items():
            toks = {k: f[col].reindex(both).fillna("").astype(str).str.split(", ") for k, f in fr.items()}
            rec = {"date": d, "tf": tf, "종목": len(both)}
            for tok in tokens:
                for k, t in toks.items():
                    rec[f"{k}_{tok}"] = int(t.map(lambda x, q=tok: q in x).sum())
            og = toks["예전"].map(lambda x: "골든크로스" in x)
            ng = toks["새"].map(lambda x: "골든크로스" in x)
            rec["골든_같음"] = int((og == ng).sum())
            rows.append(rec)
    return pd.DataFrame(rows)


def compare(runs: dict[str, iv.RunResult], new_key: str, dates: list[str]) -> None:
    print("===== 3단계 4/4 비교 · 판정", flush=True)
    raw = pd.read_parquet(isb.PRICE_CACHE)
    eqs = {k: r.equity for k, r in runs.items()}
    idx = eqs["OLD"].index
    spy = raw["SPY"]["Close"].reindex(idx)
    eqs["SPY"] = spy / spy.iloc[0] * float(eqs["OLD"].iloc[0])
    stats = {k: _stats(runs[k].equity, runs[k].trades) for k in RUNS}
    if VARIANT == "b":                     # 참고용 A — 3단계 결과 파일 그대로 (다시 돌리지 않음, 거래 목록은 없음)
        a_eq = pd.read_parquet(A_EQUITY)
        same_old = a_eq["OLD"].reindex(idx).sub(eqs["OLD"]).abs().max()
        if not same_old < 1e-6:
            print(f"  ⚠️ A 결과 파일의 예전 라벨 곡선이 이번 것과 다르다 (최대 차이 {same_old}) — B vs A 는 믿지 말 것", flush=True)
        eqs["A"], eqs["A_EF"] = a_eq["NEW"].reindex(idx), a_eq["NEW_EF"].reindex(idx)
        a_st = pd.read_csv(A_CSV, encoding="utf-8-sig").set_index("run")
        for k, src in (("A", "NEW"), ("A_EF", "NEW_EF")):
            stats[k] = {c: int(v) if c in ("거래수", "0주매수") else float(v)
                        for c, v in a_st.loc[src].drop("이름").items()}
    if VARIANT == "c":                     # 참고용 B · A — B안 결과 파일 그대로 (다시 돌리지 않음)
        b_eq = pd.read_parquet(B_EQUITY)
        same_old = b_eq["OLD"].reindex(idx).sub(eqs["OLD"]).abs().max()
        if not same_old < 1e-6:
            print(f"  ⚠️ B 결과 파일의 예전 라벨 곡선이 이번 것과 다르다 (최대 차이 {same_old}) — C vs B 는 믿지 말 것", flush=True)
        b_st = pd.read_csv(B_CSV, encoding="utf-8-sig").set_index("run")
        for k, src in (("B", "NEW"), ("B_EF", "NEW_EF"), ("A", "A"), ("A_EF", "A_EF")):
            eqs[k] = b_eq[src].reindex(idx)
            stats[k] = {c: int(v) if c in ("거래수", "0주매수") else float(v)
                        for c, v in b_st.loc[src].drop("이름").items()}
    pd.DataFrame(eqs).to_parquet(EQUITY_OUT)

    spy_r = eqs["SPY"].pct_change().dropna()
    stats["SPY"] = {"Sharpe": sharpe(spy_r.tolist()), "CAGR": cagr(spy_r.tolist()), "MDD": max_drawdown(spy_r.tolist())}
    rets = {k: v.pct_change().dropna() for k, v in eqs.items()}
    main, _ = iv._cmp(rets["NEW"], rets["OLD"], N_BOOT)
    ef, _ = iv._cmp(rets["NEW_EF"], rets["OLD_EF"], N_BOOT)
    ref: dict[str, dict[str, float]] = {}
    if VARIANT == "b":
        ref["B vs A (참고)"], _ = iv._cmp(rets["NEW"], rets["A"], N_BOOT)
        ref["B vs A · 균등 크기 (참고)"], _ = iv._cmp(rets["NEW_EF"], rets["A_EF"], N_BOOT)
    if VARIANT == "c":
        ref["C vs B (참고)"], _ = iv._cmp(rets["NEW"], rets["B"], N_BOOT)
        ref["C vs B · 균등 크기 (참고)"], _ = iv._cmp(rets["NEW_EF"], rets["B_EF"], N_BOOT)
    worse = main["ΔSharpe_상한"] < 0
    verdict = "확실히 나빠짐" if worse else "비슷함"
    p_worse = 1.0 - main["p_개선아님"]   # 재표본 중 Δ > 0 비율 = '나빠짐' 쪽 한쪽 p
    ov, n_old, n_new = _overlap(runs["OLD"].trades, runs["NEW"].trades)
    print("  라벨 통계 (예전 캐시 vs 새 캐시) 읽는 중...", flush=True)
    lab = label_counts(new_key, dates)

    names = {**RUNS, **(A_RUNS if VARIANT == "b" else {}), **(B_RUNS if VARIANT == "c" else {})}
    rows = []
    for k in [*names, "SPY"]:
        rows.append({"run": k, "이름": names.get(k, "SPY 들고 있기"), **stats[k]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    memo = (f"Sharpe {stats['NEW']['Sharpe']:.2f} vs 예전 {stats['OLD']['Sharpe']:.2f}, ΔSharpe {main['ΔSharpe']:+.2f} "
            f"[{main['ΔSharpe_하한']:+.2f}, {main['ΔSharpe_상한']:+.2f}] p(나빠짐) {p_worse:.4f}, CAGR "
            f"{stats['NEW']['CAGR']:+.1%} vs {stats['OLD']['CAGR']:+.1%}, MDD {stats['NEW']['MDD']:.1%} vs "
            f"{stats['OLD']['MDD']:.1%}, 거래 {stats['NEW']['거래수']} vs {stats['OLD']['거래수']} (겹침 {ov}), "
            f"균등 크기 참고 {stats['NEW_EF']['Sharpe']:.2f} vs {stats['OLD_EF']['Sharpe']:.2f}")
    if ref:
        rk, ok = ("B vs A (참고)", "A") if VARIANT == "b" else ("C vs B (참고)", "B")
        r = ref[rk]
        memo += (f"; 참고 {rk[:6]} {stats['NEW']['Sharpe']:.2f} vs {stats[ok]['Sharpe']:.2f}, ΔSharpe {r['ΔSharpe']:+.2f} "
                 f"[{r['ΔSharpe_하한']:+.2f}, {r['ΔSharpe_상한']:+.2f}]")
    if not _logged():
        log_trials(STEP, FAMILY, 1, [{"test_id": TEST, "판정": verdict, "메모": memo}])
    write_report(stats, main, ef, verdict, p_worse, (ov, n_old, n_new), lab, eqs, ref)
    print(f"\n3단계 판정: {verdict} — {memo}", flush=True)


REPORT_TEXT = {
    "a": {
        "title": "골든크로스 통일 3단계 — 새 라벨로 PT-1 10년",
        "worse": "새 라벨의 PT-1 이 예전보다 확실히 나쁘다 → 계획대로 멈추고 점수 가중치부터 다시 본다.",
        "same": "차이가 운과 구분되지 않는다 → 계획대로 라벨 · 계산 · 차트 통일을 진행한다 (4단계). "
                "좋아졌더라도 채택 근거가 아니다 — 이 변경은 성과가 아니라 차트와 맞추기 위한 것.",
        "changed": "- 바꾼 것: 교차 라벨(일·주·월봉 골든 · 골든임박 · 데드 · 데드임박)을 TradingView 차트의 EMA10(주황) · "
                   "EMA20(보라)로 — 분할만 반영 가격 · 끝난 봉 · TradingView 방식 EMA (screener/chart_ma.py). "
                   "★ 점수 가중치 · PT-1 규칙은 그대로",
    },
    "b": {
        "title": "골든크로스 통일 3단계 B안 — 정배열 규칙을 뺀 새 라벨로 PT-1 10년",
        "worse": "B안의 PT-1 이 예전보다 확실히 나쁘다 → 계획대로 B안은 멈추고, A 로 갈지 점수 겹침을 다시 볼지 정한다.",
        "same": "차이가 운과 구분되지 않는다 → 계획대로 B안으로 4단계(사이트 · 시트 · 메일 표시)를 진행한다. "
                "좋아졌더라도 채택 근거가 아니다 — 이 변경은 화면 라벨을 차트와 맞추기 위한 것.",
        "changed": "- 바꾼 것: A(goldencross_v1 — 교차 라벨을 차트 EMA10 주황 · EMA20 보라로)에서, 주·월봉이 정배열(완전/눌림목)이면 "
                   "골든크로스 · 골든크로스임박 태그를 지우던 예전 규칙만 뺐다 (screener/patterns.py). 정배열은 차트에 없는 선"
                   "(주봉 수정주가 EMA10/20/50 · 월봉 EMA3/6/12)이라 차트에선 아직 교차 전이어도 켜진다. "
                   "★ 점수 가중치 · PT-1 규칙은 그대로 — 이제 골든 · 임박 점수와 정배열 점수가 같이 붙을 수 있다",
    },
    "c": {
        "title": "골든크로스 통일 C — 모든 패턴을 차트와 같은 바탕으로 바꾼 뒤 PT-1 10년",
        "worse": "C 의 PT-1 이 예전보다 확실히 나쁘다 → 계획대로 C 는 멈추고, B 로 갈지 다시 정한다.",
        "same": "차이가 운과 구분되지 않는다 → 계획대로 C 로 4단계(사이트 · 시트 · 메일 표시)를 진행한다. "
                "좋아졌더라도 채택 근거가 아니다 — 이 변경은 화면 라벨 · 패턴을 차트와 맞추기 위한 것.",
        "changed": "- 바꾼 것: B(교차 라벨 = 차트 EMA10/20 · 정배열이 골든 태그를 안 지움)에 더해 모든 패턴을 TradingView 차트와 같은 "
                   "바탕으로 — 가격 패턴 · 캔들 · 차트반전도 분할만 반영 가격 · 주·월봉 끝난 봉만 · Pine 계산(EMA · RSI · MACD), "
                   "정배열 3단계는 차트 4선 EMA10/20/100/200 (있는 선으로만, 예전: 주봉 EMA10/20/50 · 월봉 EMA3/6/12), "
                   "주·월봉 선은 상장 이후 전체 기록 (얼린 차트 가격 파일), 주말에 끝나는 달은 그 전 금요일 장 뒤에 끝난 봉. "
                   "★ 점수 가중치 · PT-1 규칙은 그대로",
    },
}


def _tv_check_line() -> str:
    """C 선행 확인(chart_tv_check_c.py) 판정 줄 — 보고서에 그대로 옮긴다."""
    path = TIER3_DIR / "CHART_TV_CHECK_C.md"
    if not path.exists():
        return "- 선행 확인: CHART_TV_CHECK_C.md 없음 (OCR 대조를 안 돌렸다)"
    line = next((x for x in path.read_text(encoding="utf-8").splitlines() if x.startswith("- **S&P 500")), "판정 줄 없음")
    line = line.lstrip("- ").replace(" — 백테스트로 가지 않는다**", "** — 사용자 결정으로 진행 (아래 한계)")
    return f"- 선행 확인 (TradingView 캡처 OCR 대조, CHART_TV_CHECK_C.md): {line}"


def write_report(stats: dict[str, dict[str, Any]], main: dict[str, float], ef: dict[str, float], verdict: str,
                 p_worse: float, overlap: tuple[int, int, int], lab: pd.DataFrame, eqs: dict[str, pd.Series],
                 ref: dict[str, dict[str, float]]) -> None:
    s, txt, b, c = stats, REPORT_TEXT[VARIANT], VARIANT == "b", VARIANT == "c"
    ov, n_old, n_new = overlap
    start, end = eqs["OLD"].index[0].date(), eqs["OLD"].index[-1].date()
    lines = [
        f"# {txt['title']} ({start} ~ {end})", "",
        f"**결론: {verdict}.** " + (txt["worse"] if verdict == "확실히 나빠짐" else txt["same"]), "",
        f"- 사전 등록: TRIAL_LOG family `{FAMILY}` · 1건 · '확실히 나빠짐' = ΔSharpe 95% 구간 상한 < 0 · "
        f"부트스트랩 {N_BOOT:,}회 (20일 블록) · 누적 시도 {total_trials():,}건",
        txt["changed"],
        "- 같은 것: 가격(디스크 고정 13y) · 종목 목록(고정 789개, 그날 S&P 500 만 후보) · 매일 리밸런스 · $5,000 · 편도 0.1%",
        "- 확인: 예전 라벨 실행 = 저장된 기준선 (pt1_base_dev_equity, 차이 0) · 예전 캐시는 읽기만 함"
        + (" · A 는 3단계 결과 파일(goldencross_pt1.csv · goldencross_pt1_equity.parquet)을 읽기만 함" if b else "")
        + (" · B · A 는 B안 결과 파일(goldencross_pt1_b.csv · goldencross_pt1_b_equity.parquet)을 읽기만 함" if c else ""),
        *([_tv_check_line()] if c else []), "",
        "## 결과", "",
        "| 실행 | Sharpe | 연수익 | 최대낙폭 | 거래 | 0주 매수 | 승률 | 거래당 평균 |", "|---|---|---|---|---|---|---|---|",
    ]
    for k, name in [*RUNS.items(), *(A_RUNS.items() if b else []), *(B_RUNS.items() if c else []), ("SPY", "SPY 들고 있기")]:
        x = s[k]
        if k == "SPY":
            lines.append(f"| {name} | {x['Sharpe']:.2f} | {x['CAGR']:+.1%} | {x['MDD']:.1%} | | | | |")
        else:
            lines.append(f"| {name} | {x['Sharpe']:.2f} | {x['CAGR']:+.1%} | {x['MDD']:.1%} | {x['거래수']} | {x['0주매수']} | "
                         f"{x['승률']:.0%} | {x['거래당평균']:+.2%} |")
    new_name = "B안" if b else ("C" if c else "새 라벨")
    lines += ["", f"| 비교 | ΔSharpe | 95% 구간 | p(나빠짐) | 3구간 중 {new_name}이 나은 구간 | 판정 |", "|---|---|---|---|---|---|",
              f"| {new_name} vs 예전 라벨 (본 검정) | {main['ΔSharpe']:+.2f} | [{main['ΔSharpe_하한']:+.2f}, {main['ΔSharpe_상한']:+.2f}] | "
              f"{p_worse:.3f} | {main['구간_개선']}/{main['구간_수']} | **{verdict}** |"]
    for name, r in [("균등 크기 (참고)", ef), *ref.items()]:
        lines.append(f"| {name} | {r['ΔSharpe']:+.2f} | [{r['ΔSharpe_하한']:+.2f}, {r['ΔSharpe_상한']:+.2f}] | "
                     f"{1 - r['p_개선아님']:.3f} | {r['구간_개선']}/{r['구간_수']} | 참고 |")
    yr_runs = {"OLD": "예전 라벨", **({"A": "A"} if b else {}), **({"B": "B안"} if c else {}), "NEW": new_name,
               "OLD_EF": "예전 · 균등", **({"A_EF": "A · 균등"} if b else {}), **({"B_EF": "B안 · 균등"} if c else {}),
               "NEW_EF": f"{new_name if (b or c) else '새'} · 균등", "SPY": "SPY"}
    lines += ["", f"- 거래 겹침 (종목 · 산 날 같음, {new_name} vs 예전): {ov}건 — 예전 {n_old}건의 {ov / max(n_old, 1):.0%}, "
              f"{new_name} {n_new}건의 {ov / max(n_new, 1):.0%}" + (" (A 는 거래 목록을 저장하지 않아 B vs A 겹침은 없음)" if b else "")
              + (" (B · A 는 거래 목록을 저장하지 않아 C vs B 겹침은 없음)" if c else ""),
              "", "## 연도별 수익", "", "| 연도 | " + " | ".join(yr_runs.values()) + " |", "|---" * (len(yr_runs) + 1) + "|"]
    yr = {k: _yearly(v) for k, v in eqs.items()}
    for y in yr["OLD"].index:
        lines.append(f"| {y} | " + " | ".join(f"{yr[k].get(y, float('nan')):+.1%}" for k in yr_runs) + " |")
    if len(lab):
        cols = ["예전", *(["A"] if b else []), *(["B"] if c else []), "새"]
        head = {"새": new_name if (b or c) else "새", "B": "B안"}
        lines += ["", "## 라벨이 얼마나 바뀌었나 (하루 평균 종목 수, 그날 피처가 있는 종목 전체)", "",
                  "| 봉 | 라벨 | " + " | ".join(head.get(x, x) for x in cols) + " |",
                  "|---" * (len(cols) + 2) + "|"]
        for tf in PATTERN_COLS.values():
            g = lab[lab["tf"] == tf]
            for tok in TOKENS + (TOKENS_C if c else ()):
                if tf == "일봉" and tok.startswith("정배열"):
                    continue
                lines.append(f"| {tf} | {tok} | " + " | ".join(f"{g[f'{x}_{tok}'].mean():.1f}" for x in cols) + " |")
        lines += [""] + [f"- {tf}: 골든크로스 라벨이 붙었는지가 예전과 같은 종목-일 "
                         f"{lab[lab['tf'] == tf]['골든_같음'].sum() / max(lab[lab['tf'] == tf]['종목'].sum(), 1):.1%}"
                         for tf in PATTERN_COLS.values()]
    lines += ["", "## 한계", "",
              "- 10년 한 번의 경로다. 3종목 계좌라 고르는 순서가 조금만 바뀌어도 결과가 크게 흔들린다 (2026-10-22 무작위 순서 시험).",
              "- Sharpe 차이의 95% 구간이 넓어서 (±0.3 안팎) 큰 차이가 아니면 운과 구분되지 않는다.",
              "- 나중에 상장폐지된 종목 중 가격이 없는 것은 빠진다 (예전 기준선과 같은 조건).",
              "- 2025-10 이후 홀드아웃은 쓰지 않았다."]
    if b:
        lines.append("- B vs A 는 판정 없는 참고다 — 같은 10년 한 경로라 둘의 차이도 운과 잘 구분되지 않는다.")
    if c:
        lines += ["- C vs B 는 판정 없는 참고다 — 같은 10년 한 경로라 둘의 차이도 운과 잘 구분되지 않는다.",
                  f"- 선행 확인: {C_TV_CHECK_NOTE}",
                  "- 2026-10-27 22:10 성능만 고쳐 다시 돌렸다 (교차 판정을 마지막 60봉에서만 · 선은 전체 기록 · chart_view 경계 찾기). "
                  "고치기 전 코드와 21,398 종목-날짜(백테스트 캐시 25일 + 따로 고른 12일)에서 피처 71칸이 모두 같았다. "
                  "느렸던 큰 원인은 맥의 coreaudiod 메모리 누수(1.9GB) · 화면 공유였다.",
                  "- 차트 가격 파일(상장 이후 전체)은 2026-10-27 에 받았다. 가격이 있는 628종목은 다 들어 있고, 2026년에 합병 · "
                  "상장폐지된 몇 종목(EA · AVB · EQR · LEG · WBD · QRVO 등)은 차트 가격이 한 줄뿐이라 백테스트 가격(5년)으로 계산됐다 — "
                  "그 종목은 월봉 EMA100/200 이 없어 정배열이 형성중까지만이다.",
                  "- 월봉 EMA100/200 은 야후와 TradingView 의 기록 시작이 다른 종목에서 조금 다르다 (scripts/chart_tv_check_c.py)."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# 등록 · 실행
# ============================================================
def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame(columns=["family", "판정"])


def _registered() -> bool:
    log = _log()
    return bool(((log["family"] == FAMILY) & (log["판정"] == "등록")).any())


def _logged() -> bool:
    log = _log()
    return bool(((log["family"] == FAMILY) & (log["test_id"] == TEST)).any())


def register() -> None:
    if _registered():
        print("이미 등록됨", flush=True)
        return
    fixed = f"[{RUN_KW['start_date']}~{RUN_KW['end_date']}, PT-1 엔진 BASE 설정 · 가격 · 종목 목록 고정] "
    if VARIANT == "c":
        memo = (fixed + "C = goldencross_v2(B안) 코드에 더해 모든 패턴(가격 패턴 · 캔들 · 정배열 · 차트반전)을 차트와 같은 바탕으로 — "
                "분할만 반영 가격 · 주·월봉 끝난 봉만 · Pine 계산 · 정배열은 차트 4선 EMA10/20/100/200 (있는 선으로만) · 주·월봉 선은 "
                "상장 이후 전체 기록(얼린 차트 가격 파일) · 주말에 끝나는 달은 그 전 금요일 장 뒤 끝난 봉 vs 예전 라벨(저장된 기준선). "
                f"{TEST} = ΔSharpe(C−예전), 20일 블록 부트스트랩 {N_BOOT:,}회. 확실히 나빠짐 = 95% 상한<0 → C 멈춤 (B 로 갈지 다시 정함), "
                "그 밖 = 비슷함 → C로 4단계 진행 (좋아져도 채택 근거 아님 — 화면 라벨 · 패턴을 차트와 맞추는 변경). "
                "참고(판정 없음): C vs B(v2 결과 파일) ΔSharpe · 균등 크기 전후 · 거래 겹침 · 연도별 · 라벨 수(정배열 · 차트반전 등 포함). "
                f"선행 확인: {C_TV_CHECK_NOTE}")
    elif VARIANT == "b":
        memo = (fixed + "B안 = goldencross_v1 의 새 교차 라벨(차트 EMA10/20) 코드에서 '주·월봉 정배열(완전/눌림목)이면 골든크로스 · "
                "골든크로스임박 태그를 지운다'는 예전 규칙만 뺀 코드 vs 예전 라벨(저장된 기준선). "
                f"{TEST} = ΔSharpe(B−예전), 20일 블록 부트스트랩 {N_BOOT:,}회. 확실히 나빠짐 = 95% 상한<0 → B 멈춤 (A 로 갈지 "
                "점수 겹침을 다시 볼지 정함), 그 밖 = 비슷함 → B로 4단계 진행 (좋아져도 채택 근거 아님 — 화면 라벨을 차트와 맞추는 변경). "
                "참고(판정 없음): B vs A(v1 결과 파일) ΔSharpe · 균등 크기 전후 · 거래 겹침 · 연도별 · 라벨 수.")
    else:
        memo = (fixed + "교차 라벨을 차트 EMA10/20 "
                "(분할만 반영 · 끝난 봉 · TV 방식 EMA, 일·주·월 + 데드 · 데드임박 거울)으로 바꾼 코드 vs 예전 라벨(저장된 기준선). "
                f"{TEST} = ΔSharpe(새−예전), 20일 블록 부트스트랩 {N_BOOT:,}회. 확실히 나빠짐 = 95% 상한<0 → 멈춤, 그 밖 = 비슷함 → 진행 "
                "(좋아져도 채택 근거 아님). 참고(판정 없음): 균등 크기(equal_fixed) 전후 · 거래 겹침 · 연도별 · 라벨 수.")
    log_trials(STEP, FAMILY, 1, [{"test_id": "(등록)", "판정": "등록", "메모": memo}])
    print(f"등록: {FAMILY} 1건", flush=True)


def run(workers: int, skip_step2: bool) -> None:
    if not _registered():
        raise SystemExit("먼저 --register (결과 보기 전에 기준 등록)")
    t0 = time.time()
    if not skip_step2:
        if VARIANT == "c":
            print("===== C 확인 (chart_tv_check_c.py — 4선 · 캔들 · 정배열 OCR 대조)", flush=True)
            rc = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts" / "chart_tv_check_c.py")],
                                cwd=ROOT).returncode
            if rc != 0:
                raise SystemExit(f"C 확인 통과 못 함 (종료 코드 {rc}) — 백테스트로 가지 않는다 (CHART_TV_CHECK_C.md)")
        else:
            step2()
    iv.ensure_pinned()
    new_key = cache_key(bt._code_hash())
    if new_key == OLD_KEY:
        raise SystemExit("새 코드의 캐시 이름이 예전과 같다 — 1단계 코드가 안 들어간 상태라 멈춤")
    if VARIANT == "b":
        if new_key == A_KEY:
            raise SystemExit("B 코드의 캐시 이름이 A 와 같다 — 정배열 규칙을 안 뺀 상태라 멈춤")
        missing = [p.name for p in (A_CSV, A_EQUITY) if not p.exists()]
        if missing or not feature_cache_dir(A_KEY).exists():
            raise SystemExit(f"참고용 A 결과가 없다 ({missing or A_KEY}) — 멈춤")
    if VARIANT == "c":
        if new_key in (A_KEY, B_KEY):
            raise SystemExit("C 코드의 캐시 이름이 A · B 와 같다 — C 코드가 안 들어간 상태라 멈춤")
        missing = [p.name for p in (B_CSV, B_EQUITY) if not p.exists()]
        if missing:
            raise SystemExit(f"참고용 B 결과가 없다 ({missing}) — 멈춤")
        chart_file = ROOT / chart_prices_path(_engine_tickers())
        if not chart_file.exists():
            raise SystemExit(f"차트 가격 파일 {chart_file.name} 없음 — scripts/build_chart_prices.py 로 먼저 얼린다")
    dates = target_dates()
    print(f"[gc] 개발 구간 피처 날짜 {len(dates)}일 · 예전 캐시 {OLD_KEY} · 새 캐시 {new_key}", flush=True)
    runs = stage_old(workers)
    stage_warm(workers, new_key, dates)
    runs.update(stage_new(workers, new_key, dates))
    compare(runs, new_key, dates)
    print(f"전체 {_hm(time.time() - t0)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-step2", action="store_true", help="2단계를 이미 통과했으면")
    ap.add_argument("--variant", choices=("a", "b", "c"), default="a",
                    help="b = 정배열이 골든 · 임박 태그를 지우던 규칙을 뺀 B안 · c = 모든 패턴을 차트 바탕으로 (C)")
    args = ap.parse_args()
    if args.variant == "b":
        use_variant_b()
    if args.variant == "c":
        use_variant_c()
    if args.register:
        register()
    if args.run:
        run(args.workers, args.skip_step2)


if __name__ == "__main__":
    main()
