#!/usr/bin/env python3
"""골든크로스 + 내부자 클러스터매수 + 매출 증가 (gc_insider_rev_v1): PT-1 메일의 '⚡ 골든크로스 임박/직후' 목록에서
클러스터매수 · 매출 증가 종목만 사면 PT-1·SPY 보다 나은가.

계기: 2026-10-13 사용자 — "페이퍼트레이딩 이메일 골든크로스 리스트에서 클러스터매수, 매출증가 한 회사들만 사게 되면".
정의 (메일과 같게, 그날까지 알 수 있던 값만):
  골든크로스 목록 = paper_trading.golden_cross.extract_golden_cross(그날 PT-1 후보표). 일봉 EMA20/50 · 주봉 10/40 · 월봉 3/10
                   이평선이 골든크로스 임박(간격 −5% ~ 0) 또는 직후(0 ~ +5%). 순서도 메일과 같다(여러 주기 → 일봉 포함 → 간격 0 근처).
                   후보표 = PT-1 백테스트 엔진의 날짜별 순위표 (그날 S&P 500 구성 종목 · 유동성 필터).
  클러스터매수   = 그날 포함 90일 안에 제출된 Form 4 장내 매수(P)의 서로 다른 신고자 ≥ 2명
                   (메일 내부자(90일) 칸 = openinsider fd=90 · INSIDER_CLUSTER_MIN_BUYERS). 신고자 = SEC 원본의 첫 신고자 CIK,
                   임원·이사·10% 대주주 모두 (openinsider 와 같음).
  매출 증가      = 그날까지 공시된 가장 최근 분기 매출 > 전년 같은 분기 (메일 헬스체크 칸 '매출+N%' = yfinance revenueGrowth).
                   SEC companyfacts 10-Q·10-K 의 처음 제출된 값, 4분기는 연간 − 9개월 누적. 분기 말이 200일 넘게 지났거나
                   값을 모르면 통과 못 함.
사전 등록 (TRIAL_LOG family gc_insider_rev_v1, 2건 — 결과 계산 전에 --register):
  GCCR-pt1  PT-1 엔진·계좌($5,000·3슬롯·약세장 2슬롯·하루 1건): 목록 ∩ 클러스터 ∩ 매출↑ 중 메일 순서 첫 종목(보유 중 제외)을
            산다, 없으면 안 삼. 매도는 PT-1 규칙 그대로(그 종목의 전략구분 기준 익절·손절·트레일링·보유일), 교체 끔. vs PT-1 BASE
  GCCR-ev   신호 품질: 목록 ∩ 클러스터 ∩ 매출↑ 에 새로 든 날의 다음 시가부터 63거래일 보유하는 달력시간 포트폴리오
            (종목당 ≤ 5%, 남는 비중 SPY, 편도 0.1%) vs SPY + 이벤트 통계(63일 초과수익 vs 그날 구성 종목 동일가중, NW t)
  판정: 채택 후보 = verdict '채택 후보'(ΔSharpe 95% 하한 > 0 · 3구간 중 2+ · MDD 2%p 이내) · p ≤ 0.05/2,
        GCCR-ev 는 이벤트 판정도 '후보'. 관찰 = 그 밖에 ΔSharpe 하한 > 0 (GCCR-ev 는 이벤트 '후보'·'관찰'도)
  참고 (판정 없음): 목록 전체 · 클러스터만 · 매출만 (엔진·달력시간 모두), 21거래일 보유, SPY·PT-1 대비, 건수·연도별
기간 2016-01-04 ~ 2025-09-30 (홀드아웃 안 씀). 확인: BASE = 저장된 기준선, 래퍼 통과 모드 = BASE (아니면 멈춘다).
한계: 라이브 메일 목록엔 NASDAQ 종목도 있지만 여기선 그날 S&P 500 구성 종목만 (생존 편향 없는 자료가 그것뿐).

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/gc_insider_rev.py --lists      # 목록·조건 건수만 (성과 계산 없음)
  PYTHONPATH=.:src .venv/bin/python scripts/gc_insider_rev.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/gc_insider_rev.py --run
산출: tier3/GC_INSIDER_REV.md · tier3/gc_insider_rev.csv · data/research/gc_* (git 제외)
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import math
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import isb_engine as isb  # noqa: E402
import isb_vs_pt1 as iv  # noqa: E402
import paper_trading.backtest as bt  # noqa: E402
from fetch_insider_data import INSIDER_DIR, _read_table  # noqa: E402
from idea_batch_research import active_from_entries, calendar_time_returns, event_stats, member_mask  # noqa: E402
from paper_trading import candidate_selector as cs  # noqa: E402
from paper_trading.config_override import config_overrides  # noqa: E402
from paper_trading.evaluation import cagr, max_drawdown, sharpe, verdict  # noqa: E402
from paper_trading.golden_cross import extract_golden_cross  # noqa: E402
from paper_trading.universe import Membership, load_membership  # noqa: E402
from pead_research import grade  # noqa: E402
from research_utils import (DEV_END, RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, bonferroni_z, load_ohlcv,  # noqa: E402
                            log_trials, total_trials)
from screener import config as scfg  # noqa: E402
from signal_portfolio_sim import open_to_open  # noqa: E402
from validate_upside_walkforward import RUN_KW  # noqa: E402

FAMILY = "gc_insider_rev_v1"
STEP = "stage15"
TESTS = ("GCCR-pt1", "GCCR-ev")
N_TESTS = len(TESTS)
P_BONF = 0.05 / N_TESTS
N_BOOT, N_BOOT_REF = iv.N_BOOT, iv.N_BOOT_REF
HOLD, HOLD_REF = 63, 21
CLUSTER_DAYS = int(getattr(scfg, "INSIDER_LOOKBACK_DAYS", 90))
CLUSTER_MIN = int(getattr(scfg, "INSIDER_CLUSTER_MIN_BUYERS", 2))
REV_STALE_DAYS = 200
START = iv.START
VARIANTS: dict[str, tuple[str, bool, bool]] = {   # 키: (이름, 클러스터 필요, 매출 증가 필요)
    "GCCR": ("목록 ∩ 클러스터매수 ∩ 매출 증가", True, True),
    "GC": ("골든크로스 목록 전체", False, False),
    "GCC": ("목록 ∩ 클러스터매수", True, False),
    "GCR": ("목록 ∩ 매출 증가", False, True),
}
REV_TAGS = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax"]   # build_sec_data.FLOW_TAGS["revenue"] 와 같음
REV_FORMS = {"10-Q", "10-Q/A", "10-K", "10-K/A", "10-KT"}
INSIDER = RESEARCH_DIR / "insider_trades.parquet"
SEC_DIR = RESEARCH_DIR / "sec"
OWNER_PATH = RESEARCH_DIR / "insider_first_owner.parquet"
REV_PATH = RESEARCH_DIR / "rev_yoy_quarterly.parquet"
GC_PATH = RESEARCH_DIR / "gc_lists_daily.parquet"
RUNS_PATH = RESEARCH_DIR / "gc_insider_rev_runs.parquet"
OUT_CSV = TIER3_DIR / "gc_insider_rev.csv"
REPORT = TIER3_DIR / "GC_INSIDER_REV.md"
NAN = float("nan")


def spec() -> str:
    """사전 등록 문구."""
    return (f"[{RUN_KW['start_date']}~{RUN_KW['end_date']}, PT-1 백테스트 엔진·13y PIT S&P 500·$5,000·3슬롯] "
            "골든크로스 목록 = extract_golden_cross(그날 PT-1 후보표, 일봉 20/50·주봉 10/40·월봉 3/10 임박 −5~0%·직후 0~5%, 메일 순서); "
            f"클러스터매수 = 그날 포함 {CLUSTER_DAYS}일 안 Form 4 장내 매수 신고자(첫 신고자 CIK) ≥ {CLUSTER_MIN}명; "
            f"매출 증가 = 그날까지 공시된 최근 분기 매출 YoY > 0 (SEC companyfacts 처음 제출 값, Q4 = 연간 − 9개월, "
            f"분기 말 {REV_STALE_DAYS}일 초과·모름은 불통과). "
            "GCCR-pt1 = 목록∩클러스터∩매출↑ 중 메일 순서 첫 종목 매수(없으면 안 삼) + PT-1 매도 규칙, 교체 끔 vs PT-1 BASE; "
            f"GCCR-ev = 새로 든 날 다음 시가부터 {HOLD}거래일 달력시간(종목당 ≤5%, 나머지 SPY, 편도 0.1%) vs SPY + "
            "이벤트 통계(초과수익 vs 그날 구성종목 동일가중, NW t, 공통 판정). "
            f"채택 = verdict 채택 후보(ΔSharpe 95% 하한>0 · 3구간 중 2+ · MDD 2%p 이내) · p ≤ 0.05/{N_TESTS} "
            f"(부트스트랩 {N_BOOT:,}회), GCCR-ev 는 이벤트 '후보'도. 참고 = 목록 전체·클러스터만·매출만, {HOLD_REF}일 보유, SPY·PT-1 대비.")


# ============================================================
# 내부자 클러스터매수
# ============================================================
def build_owner_map(force: bool = False) -> pd.DataFrame:
    """신고서(accession)마다 첫 신고자 CIK — fetch_insider_data.py 가 받아 둔 로컬 SEC zip 의 REPORTINGOWNER."""
    if OWNER_PATH.exists() and not force:
        return pd.read_parquet(OWNER_PATH)
    need = set(pd.read_parquet(INSIDER, columns=["accession"])["accession"])
    parts = []
    for z in sorted(INSIDER_DIR.glob("*_form345.zip")):
        with zipfile.ZipFile(z) as zf:
            o = _read_table(zf, "REPORTINGOWNER", ["ACCESSION_NUMBER", "RPTOWNERCIK"])
        parts.append(o[o["ACCESSION_NUMBER"].isin(need)].drop_duplicates("ACCESSION_NUMBER", keep="first"))
    out = (pd.concat(parts).drop_duplicates("ACCESSION_NUMBER", keep="first")
           .rename(columns={"ACCESSION_NUMBER": "accession", "RPTOWNERCIK": "owner"}))
    out["owner"] = out["owner"].str.strip()
    out.to_parquet(OWNER_PATH, index=False)
    print(f"[gc] 신고자 표: {len(out):,}건 / 매수 신고서 {len(need):,}건", flush=True)
    return out


class ClusterBook:
    """종목마다 Form 4 장내 매수 신고(제출일, 첫 신고자) → 그날 포함 N일 안의 서로 다른 신고자 수."""

    def __init__(self, trades: pd.DataFrame, owners: pd.DataFrame, days: int = CLUSTER_DAYS) -> None:
        t = trades[["accession", "filing_date", "티커"]].drop_duplicates("accession").merge(owners, on="accession",
                                                                                           how="left")
        t["owner"] = t["owner"].fillna(t["accession"])
        self.days = np.timedelta64(days, "D")
        self.by: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for tk, g in t.sort_values("filing_date").groupby("티커"):
            self.by[str(tk)] = (pd.to_datetime(g["filing_date"]).to_numpy("datetime64[D]"),
                                g["owner"].astype(str).to_numpy())

    def buyers(self, ticker: str, day: pd.Timestamp) -> int:
        got = self.by.get(ticker)
        if got is None:
            return 0
        d, o = got
        t = np.datetime64(pd.Timestamp(day).date(), "D")
        lo, hi = np.searchsorted(d, t - self.days, side="right"), np.searchsorted(d, t, side="right")
        return len(set(o[lo:hi].tolist()))


# ============================================================
# 분기 매출 증가율 (그날까지 공시된 값)
# ============================================================
def quarterly_revenue(gaap: dict[str, Any]) -> pd.DataFrame:
    """분기(80~100일) 매출의 처음 제출된 값 (end, filed, val). 빠진 4분기 = 연간(350~380일) − 같은 회계연도 9개월 누적."""
    rows = []
    for pri, tag in enumerate(REV_TAGS):
        for x in gaap.get(tag, {}).get("units", {}).get("USD", []):
            if x.get("form") in REV_FORMS and x.get("start") and x.get("val") is not None:
                rows.append((pri, x["start"], x["end"], float(x["val"]), x["filed"]))
    if not rows:
        return pd.DataFrame(columns=["end", "filed", "val"])
    df = pd.DataFrame(rows, columns=["pri", "start", "end", "val", "filed"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    df["dur"] = (df["end"] - df["start"]).dt.days

    def first(x: pd.DataFrame) -> pd.DataFrame:
        return x.sort_values(["filed", "pri"]).drop_duplicates("end")

    q = first(df[df["dur"].between(80, 100)])
    fy = first(df[df["dur"].between(350, 380)])
    m9 = first(df[df["dur"].between(260, 290)])
    out = [q[["end", "filed", "val"]]]
    have = q["end"].to_numpy()
    add = []
    for r in fy.itertuples():
        if have.size and (np.abs((have - r.end.to_datetime64()) / np.timedelta64(1, "D")) <= 7).any():
            continue
        c = m9[(m9["end"] >= r.end - pd.Timedelta(days=105)) & (m9["end"] <= r.end - pd.Timedelta(days=75))
               & ((m9["start"] - r.start).abs() <= pd.Timedelta(days=7))]
        if c.empty:
            continue
        add.append({"end": r.end, "filed": max(r.filed, c.iloc[0]["filed"]), "val": r.val - c.iloc[0]["val"]})
    if add:
        out.append(pd.DataFrame(add))
    return pd.concat(out).sort_values("end").reset_index(drop=True)


def yoy_table(qrev: pd.DataFrame) -> pd.DataFrame:
    """분기마다 전년 같은 분기(약 364일 전 ±20일) 대비 증가율. known = 두 값이 모두 공시된 날."""
    if qrev.empty:
        return pd.DataFrame(columns=["end", "known", "yoy"])
    q = qrev.sort_values("end").reset_index(drop=True)
    ends = q["end"].to_numpy()
    rows = []
    for r in q.itertuples():
        target = (r.end - pd.Timedelta(days=364)).to_datetime64()
        j = int(np.argmin(np.abs(ends - target)))
        if abs((ends[j] - target) / np.timedelta64(1, "D")) > 20 or j == r.Index:
            continue
        prev = q.iloc[j]
        if not (prev["val"] > 0):
            continue
        rows.append({"end": r.end, "known": max(r.filed, prev["filed"]), "yoy": r.val / prev["val"] - 1.0})
    return pd.DataFrame(rows, columns=["end", "known", "yoy"])


def build_rev_table(force: bool = False) -> pd.DataFrame:
    """전 종목 분기 매출 YoY (티커, end, known, yoy) — SEC companyfacts(로컬). 티커 = sec/ticker_cik.csv."""
    if REV_PATH.exists() and not force:
        return pd.read_parquet(REV_PATH)
    cik_map = pd.read_csv(SEC_DIR / "ticker_cik.csv").dropna(subset=["cik"])
    tickers: dict[int, list[str]] = {}
    for t, c in zip(cik_map["티커"], cik_map["cik"]):
        tickers.setdefault(int(c), []).append(str(t))
    parts = []
    for p in sorted((SEC_DIR / "companyfacts").glob("CIK*.json.gz")):
        cik = int(p.name[3:13])
        if cik not in tickers:
            continue
        with gzip.open(p) as f:
            gaap = json.load(f).get("facts", {}).get("us-gaap", {})
        y = yoy_table(quarterly_revenue(gaap))
        for t in tickers[cik]:
            parts.append(y.assign(티커=t))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["end", "known", "yoy", "티커"])
    out.to_parquet(REV_PATH, index=False)
    print(f"[gc] 분기 매출 YoY: {out['티커'].nunique()}종목 · {len(out):,}분기", flush=True)
    return out


class RevBook:
    """종목마다 그날까지 공시된 가장 최근 분기의 매출 YoY (모르면 nan)."""

    def __init__(self, table: pd.DataFrame, stale_days: int = REV_STALE_DAYS) -> None:
        self.stale = np.timedelta64(stale_days, "D")
        self.by: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for tk, g in table.sort_values(["known", "end"]).groupby("티커"):
            known, yoy, end, best = [], [], [], None
            for r in g.itertuples():
                if best is None or r.end >= best:      # 나중에 나온 옛 분기(정정 등)는 최신 분기를 덮지 않는다
                    best = r.end
                    known.append(r.known)
                    yoy.append(r.yoy)
                    end.append(r.end)
            self.by[str(tk)] = (pd.to_datetime(known).to_numpy("datetime64[D]"), np.asarray(yoy, dtype=float),
                                pd.to_datetime(end).to_numpy("datetime64[D]"))

    def yoy(self, ticker: str, day: pd.Timestamp) -> float:
        got = self.by.get(ticker)
        if got is None:
            return NAN
        k, y, e = got
        t = np.datetime64(pd.Timestamp(day).date(), "D")
        i = int(np.searchsorted(k, t, side="right")) - 1
        if i < 0 or t - e[i] > self.stale:
            return NAN
        return float(y[i])


# ============================================================
# 엔진 실행 (골든크로스 선택기)
# ============================================================
_BOOKS: dict[str, Any] = {}


def books() -> tuple[ClusterBook, RevBook]:
    """워커마다 한 번 읽는 조건표."""
    if not _BOOKS:
        _BOOKS["c"] = ClusterBook(pd.read_parquet(INSIDER), build_owner_map())
        _BOOKS["r"] = RevBook(build_rev_table())
    return _BOOKS["c"], _BOOKS["r"]


def gc_rows(df: pd.DataFrame, day: pd.Timestamp, cb: ClusterBook, rb: RevBook) -> list[dict[str, Any]]:
    """그날 후보표 → 메일 순서 골든크로스 목록 + 클러스터·매출 조건 (SPY 제외)."""
    out = []
    for rank, g in enumerate(x for x in extract_golden_cross(df) if x["ticker"] != "SPY"):
        tk = g["ticker"]
        n_buy = cb.buyers(tk, day)
        yoy = rb.yoy(tk, day)
        out.append({**g, "rank": rank, "buyers": n_buy, "cluster": n_buy >= CLUSTER_MIN, "yoy": yoy,
                    "rev_up": bool(yoy == yoy and yoy > 0)})
    return out


def passes(row: dict[str, Any], need_c: bool, need_r: bool) -> bool:
    return (row["cluster"] or not need_c) and (row["rev_up"] or not need_r)


def make_selector(variant: str | None, ctx: iv.RunCtx, record: list[dict[str, Any]] | None = None) -> Callable:
    """bt.select_top_candidates 자리에 끼울 함수. variant None = 목록만 기록하고 사지 않음 (건수 확인용)."""
    cb, rb = books()
    need_c, need_r = (VARIANTS[variant][1], VARIANTS[variant][2]) if variant else (False, False)

    def select(df: pd.DataFrame, holdings: list[dict[str, Any]], k: int = 1) -> tuple[list[dict[str, Any]], dict]:
        debug: dict[str, Any] = {"regime": cs.detect_market_regime(df)}
        day = ctx.date
        rows = gc_rows(df, day, cb, rb)
        if record is not None:
            record.extend({"date": day, "ticker": r["ticker"], "rank": r["rank"], "tf_count": r["tf_count"],
                           "has_daily": r["has_daily"], "buyers": r["buyers"], "yoy": r["yoy"]} for r in rows)
        if variant is None:
            return [], debug
        held = {h["ticker"] for h in holdings}
        picks = []
        for r in rows:
            if len(picks) >= max(1, k):
                break
            if r["ticker"] in held or not passes(r, need_c, need_r):
                continue
            row = df.loc[df["티커"].astype(str) == r["ticker"]].iloc[0]
            picks.append({"ticker": r["ticker"], "entry_price": float(row.get("close", np.nan)),
                          "strategy": r["strategy"] or "골든크로스", "star_rating": r["star"], "ccs_score": 0.0,
                          "sector": r["sector"], "ccs_breakdown": {}})
        return picks, debug

    return select


def run_variant(variant: str | None, record: list[dict[str, Any]] | None = None) -> iv.RunResult:
    """엔진 한 번: 래퍼(가격 고정·후보표 재사용·날짜) + 골든크로스 선택기 + 교체 끔. 매도는 PT-1 규칙 그대로."""
    ov = iv.Overlay(f"GC-{variant or 'lists'}")
    ctx = iv.RunCtx(ov, None)
    sink = io.StringIO()
    with iv._patched(ctx), config_overrides({"PT1_REPLACE_ENABLED": False}), contextlib.redirect_stdout(sink):
        bt.select_top_candidates = make_selector(variant, ctx, record)
        result = bt.run_paper_trading_backtest(**RUN_KW)
    eq = result["equity_curve"].dropna()
    eq.index = pd.to_datetime(eq.index)
    return iv.RunResult(ov.key, eq, result["trades"], result["summary"], [], 0, 0, 0, 0, ctx.max_weight, 0, 0.0)


def _chain(with_variants: bool) -> dict[str, Any]:
    """한 워커에서 차례로: 래퍼 통과 모드(PROBE) → 목록 기록 → (변형 4개). 후보표 재사용으로 두 번째부터 빠르다."""
    iv.enable_snapshot_cache()
    out: dict[str, Any] = {"PROBE": iv.run_overlay(iv.overlay_specs()["PROBE"])}
    rec: list[dict[str, Any]] = []
    run_variant(None, rec)
    out["lists"] = pd.DataFrame(rec)
    if with_variants:
        for v in VARIANTS:
            out[v] = run_variant(v)
    return out


# ============================================================
# 목록 → 이벤트 (달력시간·이벤트 통계용)
# ============================================================
def qualify_matrix(lists: pd.DataFrame, need_c: bool, need_r: bool, index: pd.DatetimeIndex,
                   columns: pd.Index) -> pd.DataFrame:
    """날짜 × 종목: 그날 목록에 있고 조건을 통과했는지."""
    x = lists.copy()
    ok = np.ones(len(x), dtype=bool)
    if need_c:
        ok &= x["buyers"].to_numpy() >= CLUSTER_MIN
    if need_r:
        y = x["yoy"].to_numpy(dtype=float)
        ok &= np.nan_to_num(y, nan=-1.0) > 0
    x = x[ok]
    m = np.zeros((len(index), len(columns)), dtype=bool)
    rows = index.get_indexer(pd.to_datetime(x["date"]))
    cols = columns.get_indexer(x["ticker"])
    keep = (rows >= 0) & (cols >= 0)
    m[rows[keep], cols[keep]] = True
    return pd.DataFrame(m, index=index, columns=columns)


def new_entries(q: pd.DataFrame) -> pd.DataFrame:
    """새로 든 날(전 거래일엔 없음) → 진입 목록 (signal_date, 티커, entry_pos = 다음 거래일 위치)."""
    fresh = q & ~q.shift(1, fill_value=False)
    st = fresh.stack(future_stack=True)
    st = st[st]
    ev = st.index.to_frame(index=False)
    ev.columns = ["signal_date", "티커"]
    ev["entry_pos"] = q.index.searchsorted(ev["signal_date"], side="right")
    return ev[ev["entry_pos"] < len(q.index)].sort_values(["signal_date", "티커"]).reset_index(drop=True)


# ============================================================
# 등록 · 실행 · 판정
# ============================================================
def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame(columns=["family", "판정"])


def _registered() -> bool:
    log = _log()
    return bool(((log["family"] == FAMILY) & (log["판정"] == "등록")).any())


def register() -> None:
    if _registered():
        print(f"{FAMILY} 는 이미 등록됨 — 다시 등록하지 않는다")
        return
    log_trials(STEP, FAMILY, N_TESTS, [{"test_id": "(등록)", "판정": "등록", "메모": spec()}])
    print(f"등록: {FAMILY} {N_TESTS}건 · p ≤ {P_BONF:.3f} · 누적 시도 {total_trials()}건")


def list_counts(lists: pd.DataFrame) -> pd.DataFrame:
    """연도별: 목록에 든 종목-일, 조건별 통과 종목-일, 새로 든 사건 수 (성과 아님)."""
    x = lists[(pd.to_datetime(lists["date"]) >= START) & (pd.to_datetime(lists["date"]) <= DEV_END)].copy()
    x["date"] = pd.to_datetime(x["date"])
    x["c"] = x["buyers"] >= CLUSTER_MIN
    x["r"] = np.nan_to_num(x["yoy"].to_numpy(dtype=float), nan=-1.0) > 0
    x["cr"] = x["c"] & x["r"]
    days = pd.DatetimeIndex(sorted(x["date"].unique()))
    cols = pd.Index(sorted(x["ticker"].unique()))
    rows = []
    for key, (name, nc, nr) in VARIANTS.items():
        ev = new_entries(qualify_matrix(x, nc, nr, days, cols))
        per_year = ev.groupby(ev["signal_date"].dt.year).size()
        flag = {"GC": None, "GCC": "c", "GCR": "r", "GCCR": "cr"}[key]
        sub = x if flag is None else x[x[flag]]
        rows.append({"변형": key, "이름": name, "종목일": len(sub), "있던 날": sub["date"].nunique(),
                     "새로 든 사건": len(ev), **{str(y): int(per_year.get(y, 0)) for y in range(2016, 2026)}})
    return pd.DataFrame(rows)


def lists_only() -> None:
    """목록·조건 건수만 확인 (엔진은 사지 않고 목록만 기록 → 성과 계산 없음)."""
    build_owner_map()
    build_rev_table()
    iv.ensure_pinned()
    isb.build_events("W")
    with ProcessPoolExecutor(max_workers=1) as ex:
        lists = ex.submit(_lists_worker).result()
    lists.to_parquet(GC_PATH, index=False)
    print(list_counts(lists).to_string(index=False))


def _lists_worker() -> pd.DataFrame:
    rec: list[dict[str, Any]] = []
    run_variant(None, rec)
    return pd.DataFrame(rec)


def run() -> None:
    if not _registered():
        raise SystemExit("먼저 --register 로 등록한다 (결과 계산 전 사전 등록)")
    log = _log()
    already_logged = ((log["family"] == FAMILY) & (log["판정"] != "등록")).any()
    build_owner_map()
    build_rev_table()
    iv.ensure_pinned()
    isb.build_events("W")                                  # 래퍼 통과 모드(PROBE)가 쓰는 표
    print("===== 엔진: BASE · 통과 모드 → 목록 기록 → 변형 4개 (후보표 재사용)", flush=True)
    with ProcessPoolExecutor(max_workers=2) as ex:
        f_base, f_chain = ex.submit(iv.run_overlay, iv.overlay_specs()["BASE"]), ex.submit(_chain, True)
        base, chain = f_base.result(), f_chain.result()
    errors = [e for e in (iv.check_base(base), iv.check_probe(base, chain["PROBE"])) if e]
    if errors:
        raise SystemExit("확인 실패 — 결과를 쓰지 않는다: " + " / ".join(errors))
    lists = chain["lists"]
    lists.to_parquet(GC_PATH, index=False)
    res = {"BASE": base, **{v: chain[v] for v in VARIANTS}}
    pd.DataFrame({k: r.equity for k, r in res.items()}).to_parquet(RUNS_PATH)
    finish(res, lists, log_results=not already_logged)


def _cmp(a: pd.Series, b: pd.Series, n_boot: int) -> tuple[dict[str, float], pd.DataFrame]:
    return iv._cmp(a, b, n_boot)


def judge(st: dict[str, float], v: str, ev_grade: str | None = None) -> str:
    """사전 등록 판정."""
    bonf = bool(st) and st["p_개선아님"] <= P_BONF
    if v == "채택 후보" and bonf and ev_grade in (None, "후보"):
        return "채택 후보"
    if (st and st["ΔSharpe_하한"] > 0) or ev_grade in ("후보", "관찰"):
        return "관찰"
    return "탈락"


def finish(res: dict[str, iv.RunResult], lists: pd.DataFrame, log_results: bool = True) -> pd.DataFrame:
    print("===== 비교·판정", flush=True)
    mem = Membership(load_membership(iv.MEMBERSHIP))
    raw = load_ohlcv(isb.PRICE_CACHE)
    opens = raw.xs("Open", axis=1, level="Price").sort_index()
    r_oo = open_to_open(raw)
    idx, cols = opens.index, opens.columns
    member = member_mask(idx, cols, mem)
    tradable = opens.notna()
    spy_oo = (opens["SPY"].shift(-1) / opens["SPY"] - 1.0)
    spy_c = load_ohlcv(iv.ETF_OHLCV)["SPY"]["Close"].pct_change()
    base_r = res["BASE"].equity.pct_change().dropna()
    z = bonferroni_z(N_TESTS)
    rows: list[dict[str, Any]] = [{"test_id": "BASE", "이름": "PT-1 그대로 (기준)", "종류": "엔진",
                                   "Sharpe": sharpe(base_r.tolist()), "CAGR": cagr(base_r.tolist()),
                                   "MDD": max_drawdown(base_r.tolist()), "거래수": len(res["BASE"].trades), "판정": "기준"}]
    series: dict[str, pd.Series] = {"PT-1": base_r, "SPY(종가)": spy_c.loc[base_r.index.min():base_r.index.max()]}
    for v, (name, _, _) in VARIANTS.items():
        rr = res[v]
        r = rr.equity.pct_change().dropna()
        tid = f"{v}-pt1"
        st, df = _cmp(r, base_r, N_BOOT if tid in TESTS else N_BOOT_REF)
        a, b = df["a"].tolist(), df["b"].tolist()
        vd = verdict(st, max_drawdown(a), max_drawdown(b))
        st_spy, df_spy = _cmp(r, spy_c, N_BOOT_REF)
        days_held = (rr.equity.diff().fillna(0) != 0)
        rows.append({
            "test_id": tid, "이름": f"PT-1 계좌 · {name}", "종류": "엔진", "기준": "PT-1 BASE",
            "Sharpe": sharpe(a), "CAGR": cagr(a), "MDD": max_drawdown(a), "거래수": len(rr.trades),
            "승률": float(np.mean([t["return_pct"] > 0 for t in rr.trades])) if rr.trades else NAN,
            "ΔSharpe": st.get("ΔSharpe"), "하한": st.get("ΔSharpe_하한"), "상한": st.get("ΔSharpe_상한"),
            "p": st.get("p_개선아님"), "구간": f"{st.get('구간_개선')}/{st.get('구간_수')}" if st else "",
            "verdict": vd, "SPY_Sharpe": sharpe(df_spy["b"].tolist()), "ΔSharpe_SPY": st_spy.get("ΔSharpe"),
            "하한_SPY": st_spy.get("ΔSharpe_하한"), "상한_SPY": st_spy.get("ΔSharpe_상한"),
            "주식 보유일 비율": float(days_held.loc[START:].mean()), "최대비중": rr.max_weight,
            "청산이유": iv.exit_mix(rr.trades), "판정": judge(st, vd) if tid in TESTS else "참고"})
        series[f"{v}-pt1"] = r

    spy_s = spy_oo.loc[START:DEV_END].iloc[:-1]
    for v, (name, nc, nr) in VARIANTS.items():
        q = qualify_matrix(lists, nc, nr, idx, cols)
        ent = new_entries(q.loc[:DEV_END])
        for hold in (HOLD, HOLD_REF):
            tid = f"{v}-ev" + ("" if hold == HOLD else f"{hold}")
            act = active_from_entries(ent, idx, cols, hold)
            ret, n_names, w_sum = calendar_time_returns(act, r_oo, tradable, spy_oo)
            s = ret.loc[START:DEV_END].iloc[:-1]
            st, df = _cmp(s, spy_s, N_BOOT if tid in TESTS else N_BOOT_REF)
            a, b = df["a"].tolist(), df["b"].tolist()
            vd = verdict(st, max_drawdown(a), max_drawdown(b))
            ev = event_stats(ent, opens, member, hold)
            g = grade(ev, z)
            rows.append({
                "test_id": tid, "이름": f"신호 품질 · {name} · {hold}거래일", "종류": "달력시간", "기준": "SPY",
                "Sharpe": sharpe(a), "CAGR": cagr(a), "MDD": max_drawdown(a), "거래수": len(ent),
                "ΔSharpe": st.get("ΔSharpe"), "하한": st.get("ΔSharpe_하한"), "상한": st.get("ΔSharpe_상한"),
                "p": st.get("p_개선아님"), "구간": f"{st.get('구간_개선')}/{st.get('구간_수')}" if st else "",
                "verdict": vd, "SPY_Sharpe": sharpe(b), "평균 보유 종목": float(n_names.loc[START:DEV_END].mean()),
                "종목 비중 평균": float(w_sum.loc[START:DEV_END].mean()), "이벤트_판정": g,
                **{f"ev_{k}": val for k, val in ev.items() if k != "yearly"},
                "판정": judge(st, vd, g) if tid in TESTS else "참고"})
            if hold == HOLD:
                series[tid] = s
    series["SPY(시가)"] = spy_s
    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)
    if log_results:
        log_trials(STEP, FAMILY, N_TESTS, [{
            "test_id": x["test_id"], "t": f"{x['ev_t_nw']:.2f}" if pd.notna(x.get("ev_t_nw", np.nan)) else "",
            "판정": x["판정"],
            "메모": (f"Sharpe {x['Sharpe']:.2f} vs {x['기준']} "
                     f"{(table.iloc[0]['Sharpe'] if x['기준'] == 'PT-1 BASE' else x['SPY_Sharpe']):.2f}, "
                     f"ΔSharpe {x['ΔSharpe']:+.2f} [{x['하한']:+.2f}, {x['상한']:+.2f}] p {x['p']:.4f}, 구간 {x['구간']}, "
                     f"CAGR {x['CAGR']:+.1%}, MDD {x['MDD']:.1%}, 거래 {int(x['거래수'])}"
                     + (f", 이벤트 {x['이벤트_판정']}" if isinstance(x.get('이벤트_판정'), str) else ""))}
            for _, x in table.iterrows() if x["test_id"] in TESTS])
    else:
        print("TRIAL_LOG 에 결과를 쓰지 않는다 (이미 있음)", flush=True)
    write_report(table, list_counts(lists), series, res, lists)
    return table


def write_report(table: pd.DataFrame, counts: pd.DataFrame, series: dict[str, pd.Series],
                 res: dict[str, iv.RunResult], lists: pd.DataFrame, path: Path = REPORT) -> None:
    t = table.set_index("test_id")
    b, p1, ev = t.loc["BASE"], t.loc["GCCR-pt1"], t.loc["GCCR-ev"]
    passed = [x for x in TESTS if t.at[x, "판정"] == "채택 후보"]
    if passed:
        head = f"**결론: 사전 등록 기준을 통과한 것이 있다 — {', '.join(passed)}.**"
    else:
        head = (f"**결론: 골든크로스 목록에서 클러스터매수·매출 증가 종목만 사도 PT-1·SPY 보다 낫다고 볼 수 없다.** "
                f"PT-1 계좌로 사면 Sharpe {p1['Sharpe']:.2f} vs PT-1 {b['Sharpe']:.2f}, "
                f"신호 자체(63거래일 보유)는 {ev['Sharpe']:.2f} vs SPY {ev['SPY_Sharpe']:.2f}.")
    c = counts.set_index("변형")
    lines = [
        f"# 골든크로스 + 클러스터매수 + 매출 증가 ({FAMILY}) — {RUN_KW['start_date']} ~ {RUN_KW['end_date']}", "", head, "",
        f"- 사전 등록: TRIAL_LOG family `{FAMILY}` · {N_TESTS}건 · p ≤ 0.05/{N_TESTS} · 부트스트랩 {N_BOOT:,}회 · "
        f"누적 시도 {total_trials():,}건",
        "- 계기: 사용자 — \"페이퍼트레이딩 이메일 골든크로스 리스트에서 클러스터매수, 매출증가 한 회사들만 사게 되면\" (2026-10-13)",
        "- 골든크로스 목록: 메일과 같은 함수(`extract_golden_cross`)를 날짜별 PT-1 후보표(그날 S&P 500 구성 종목 · 유동성 필터)에 적용. "
        "일봉 EMA20/50 · 주봉 10/40 · 월봉 3/10 이평선의 골든크로스 임박(−5% ~ 0) 또는 직후(0 ~ +5%)",
        f"- 클러스터매수: 그날 포함 {CLUSTER_DAYS}일 안에 서로 다른 내부자 {CLUSTER_MIN}명 이상이 장내 매수를 신고 (SEC Form 4, 메일과 같은 기준)",
        "- 매출 증가: 그날까지 공시된 최근 분기 매출이 전년 같은 분기보다 큼 (SEC 10-Q·10-K 처음 제출 값, 메일 헬스체크의 '매출+N%')",
        "- PT-1 계좌: $5,000 · 3슬롯 · 하루 1건, 조건 맞는 종목 중 메일 순서 첫 종목을 사고 PT-1 매도 규칙으로 판다 (없으면 안 삼)",
        f"- 신호 품질: 조건에 새로 든 날 다음 시가부터 {HOLD}거래일 보유 (종목당 ≤ 5%, 나머지 SPY) → SPY 와 비교",
        "- 확인: BASE 재현 · 래퍼 투명성 통과",
        f"- 채택 후보 = 기준 대비 ΔSharpe 95% 하한 > 0 · 3구간 중 2개 이상 개선 · MDD 2%p 이내 · p ≤ {P_BONF:.3f}"
        " (신호 품질은 이벤트 판정 '후보'도)",
        "", "## 얼마나 자주 나오나 (2016-01 ~ 2025-09, 그날 S&P 500)", "",
        "| 조건 | 목록에 있던 종목-일 | 그런 날 | 새로 든 사건 | " + " | ".join(str(y) for y in range(2016, 2026)) + " |",
        "|---|---|---|---|" + "---|" * 10,
    ]
    for key in ("GC", "GCC", "GCR", "GCCR"):
        r = c.loc[key]
        lines.append(f"| {r['이름']} | {int(r['종목일']):,} | {int(r['있던 날']):,} | {int(r['새로 든 사건']):,} | "
                     + " | ".join(str(int(r[str(y)])) for y in range(2016, 2026)) + " |")
    lines += ["", "## 판정", "",
              "| 검정 | Sharpe | CAGR | MDD | 거래·사건 | 기준 | 기준 대비 ΔSharpe [95%] | p | 구간 | 이벤트 | 판정 |",
              "|---|---|---|---|---|---|---|---|---|---|---|",
              f"| BASE PT-1 그대로 | {b['Sharpe']:.2f} | {b['CAGR']:+.1%} | {b['MDD']:.1%} | {int(b['거래수'])} | | | | | | 기준 |"]
    for tid in TESTS:
        x = t.loc[tid]
        bench = f"PT-1 {b['Sharpe']:.2f}" if x["기준"] == "PT-1 BASE" else f"SPY {x['SPY_Sharpe']:.2f}"
        lines.append(f"| {tid} {x['이름']} | {x['Sharpe']:.2f} | {x['CAGR']:+.1%} | {x['MDD']:.1%} | {int(x['거래수'])} | "
                     f"{bench} | {iv._f(x['ΔSharpe'], '+.2f')} [{iv._f(x['하한'], '+.2f')}, {iv._f(x['상한'], '+.2f')}] | "
                     f"{iv._f(x['p'], '.4f')} | {x['구간']} | {x.get('이벤트_판정', '') if isinstance(x.get('이벤트_판정'), str) else ''} | "
                     f"**{x['판정']}** |")
    if pd.notna(ev.get("ev_mean_ar", np.nan)):
        lines += ["", f"- GCCR-ev 이벤트: {int(ev['ev_n_events'])}건 · 63거래일 평균 초과수익 {ev['ev_mean_ar']:+.2%} "
                  f"(비용 뺀 {ev['ev_mean_net']:+.2%}) · 중앙 {ev['ev_median_ar']:+.2%} · 이긴 비율 {ev['ev_hit']:.0%} · "
                  f"NW t {ev['ev_t_nw']:.2f} · 플러스였던 해 {int(ev['ev_years_pos'])}/{int(ev['ev_n_years'])}",
                  f"- GCCR-ev 포트폴리오: 평균 보유 {ev['평균 보유 종목']:.1f}종목 · 종목 비중 평균 {ev['종목 비중 평균']:.0%} (나머지 SPY)"]
    lines += [f"- GCCR-pt1: 거래 {int(p1['거래수'])}건 · 승률 {iv._f(p1['승률'], '.0%')} · 주식을 들고 있던 날 "
              f"{iv._f(p1['주식 보유일 비율'], '.0%')} · 청산 {p1['청산이유']} · SPY 대비 ΔSharpe {iv._f(p1['ΔSharpe_SPY'], '+.2f')}",
              "", "## 참고: 조건을 하나씩 (판정 없음)", "",
              "| 변형 | Sharpe | CAGR | MDD | 거래·사건 | 기준 대비 ΔSharpe [95%] | 이벤트 평균 초과수익 · t |",
              "|---|---|---|---|---|---|---|"]
    for tid in [f"{v}-pt1" for v in VARIANTS] + [f"{v}-ev" for v in VARIANTS] + [f"{v}-ev{HOLD_REF}" for v in VARIANTS]:
        if tid in TESTS or tid not in t.index:
            continue
        x = t.loc[tid]
        evs = (f"{x['ev_mean_ar']:+.2%} · {x['ev_t_nw']:.2f}" if pd.notna(x.get("ev_mean_ar", np.nan)) else "")
        lines.append(f"| {x['이름']} | {x['Sharpe']:.2f} | {x['CAGR']:+.1%} | {x['MDD']:.1%} | {int(x['거래수'])} | "
                     f"{x['기준']} {iv._f(x['ΔSharpe'], '+.2f')} [{iv._f(x['하한'], '+.2f')}, {iv._f(x['상한'], '+.2f')}] | {evs} |")
    yr = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in series.items()})
    lines += ["", "## 연도별 수익", "", "| 연도 | " + " | ".join(yr.columns) + " |", "|---" * (len(yr.columns) + 1) + "|"]
    lines += [f"| {y} | " + " | ".join(iv._f(v, "+.1%") for v in row) + " |" for y, row in yr.iterrows()]
    recent = lists[(lists["buyers"] >= CLUSTER_MIN) & (np.nan_to_num(lists["yoy"].to_numpy(dtype=float), nan=-1) > 0)]
    recent = recent[pd.to_datetime(recent["date"]) <= DEV_END].sort_values("date").drop_duplicates("ticker", keep="last").tail(10)
    if len(recent):
        lines += ["", "## 참고: 조건을 모두 만족했던 최근 종목 (종목별 마지막 날)", "", "| 날짜 | 종목 | 매수 신고자 | 매출 YoY |",
                  "|---|---|---|---|"]
        lines += [f"| {pd.Timestamp(r.date):%Y-%m-%d} | {r.ticker} | {int(r.buyers)}명 | {r.yoy:+.0%} |" for r in recent.itertuples()]
    lines += ["", "## 한계", "",
              "- 라이브 메일 목록엔 NASDAQ 종목도 있지만, 여기선 그날 S&P 500 구성 종목만 봤다 (생존 편향 없는 자료가 그것뿐). "
              "상장폐지 종목은 가격이 없어 빠진다.",
              "- 클러스터매수는 SEC 원본 Form 4 로 다시 셌다. openinsider 와 신고자 구분·정정 처리 방식이 조금 다를 수 있다.",
              "- 매출 증가율은 처음 공시된 분기 값으로 계산했다. yfinance 는 나중에 정정된 값을 쓸 수 있다.",
              "- 개발 구간은 지금까지 많은 시도에 쓰였다. 홀드아웃은 쓰지 않았다. 결과가 좋아도 라이브 반영은 사용자 결정이다.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"리포트: {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lists", action="store_true", help="목록·조건 건수만 (성과 계산 없음)")
    g.add_argument("--register", action="store_true", help="결과 계산 전 사전 등록 (TRIAL_LOG)")
    g.add_argument("--run", action="store_true", help="확인 → 엔진 변형 4개 → 달력시간 → 판정 · 보고서")
    args = ap.parse_args()
    if args.lists:
        lists_only()
    elif args.register:
        register()
    else:
        run()


if __name__ == "__main__":
    main()
