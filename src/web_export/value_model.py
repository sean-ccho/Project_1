"""예측 모델 — 가치 체크리스트 스크린 (2026-11-11 · 표시 전용).

사이트 내계좌 › 예측 모델 탭의 '우리 커버리지' 데이터. Baoro Research 블로그 로직(로컬 문서 docs/블로그주식분석.md 9절 — 저장소에는 안 올림)을
S&P 500 + 저자 종목에 밤마다 적용한다. 저자 종목의 출처는 두 곳 (2026-11-12): 사이트 Coverage 페이지(AUTHOR_COVERAGE)와
Ideas(블로그)의 심층 리서치 글(AUTHOR_RESEARCH — Fiserv 처럼 커버리지에 없는 종목도). A-Series 스크린 목록(AUTHOR_SCREEN)은 따로.

- 품질 관문 5개 (연간 재무제표 4~5년, yfinance)
  ROE 평균 ≥ 15% · CV ≤ 0.3 / ROIC 평균 ≥ 12% · CV ≤ 0.3 · 평균 ROIC > WACC / Altman Z ≥ 2.99 /
  Beneish M ≤ −2.22 / FCF 모든 해 플러스 · FCF 합 > 순차입 합
- 가격 관문: 3년 평균 FCF 수익률이 5년 동안 매출 성장률(−5 ~ +10%로 자름), 그 뒤 2.5% 로 자란다고 보고 낸
  기대수익률 − 요구수익률(10년 국채 + 베타 × 5.36%) ≥ 5%p. 매수 기준가 = 그 조건이 딱 맞는 주가.
- 앞으로 기록: 처음 통과한 날 · 가격(우리)과 저자 커버리지(처음 실행한 날)를 snapshot/value_log.json 에 남겨
  지금까지 수익률을 SPY(캐나다는 XIU)와 비교한다.

- v2 (같은 데이터로 따로 계산해 v1 과 같은 날부터 기록 · 비교): v1 통과 + 하방 관문(FCF −30% · 성장 0 이하에서도
  알파 ≥ 0), 우선순위 = 괴리(실제 매출 성장 − 시장이 가정하는 영구 성장) 큰 순, 반증 조건(품질 탈락 · 매출 감소(최근
  분기 · 연간) · 시장 가정 성장 > 실제 · 적정가 도달)에 걸리면 기록 종료, 업종 집중 · 베타 민감은 표시만.

재무는 snapshot/value_fund.json.gz 에 쌓아 MAX_AGE_DAYS 마다 다시 받는다 (하룻밤 시간 한도 안에서 조금씩).
주가 · 베타 · 하락폭 · 가격 관문은 매일 다시 계산한다.
점수 · 필터 · PT-1 · 시트와 무관 — universe-scan.yml 이 universe-data 브랜치에 같이 올린다.

사용: PYTHONPATH=.:src python -m web_export.value_model --out-dir DIR [--prev-dir DIR] [--tickers A,B] [--no-fetch]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT_NAME = "value_model.json"
CACHE_NAME = "value_fund.json.gz"
LOG_NAME = "value_log.json"
SCHEMA = 1

# ── 모델 가정 (docs/블로그주식분석.md 9.3절)
ERP = 0.0536            # 저자가 쓴 장기 중앙값 ERP
TERM_G = 0.025          # 5년 뒤 영구 성장률 (저자는 국채금리 — 저베타주 과대평가라 2.5%로 바꿈)
ALPHA_MIN = 0.05        # 기대수익률 − 요구수익률 ≥ 5%p
STAGE1_YEARS = 5
G1_RANGE = (-0.05, 0.10)  # 첫 5년 성장률 = 최근 매출 연평균 성장률, 이 범위로 자름
BETA_RANGE = (0.3, 2.5)
RF_FALLBACK = 0.045

# ── 품질 관문 기준
ROE_MIN, ROIC_MIN, CV_MAX = 0.15, 0.12, 0.30
ALTMAN_MIN, BENEISH_MAX = 2.99, -2.22
MIN_YEARS = 3           # 평균 · CV 를 내려면 이 해 수 이상

# ── 받기
MAX_AGE_DAYS = 30       # 연간 재무는 한 달에 한 번꼴로 다시
WORKERS = 2             # 4개면 Yahoo 429 가 났다 (2026-11-05 시험)
MAX_FAIL_STREAK = 25
BUDGET_MIN = 20.0

# ── v2 (2026-11-11) — 시니어식 보강. v1 규칙은 그대로 두고 같은 데이터로 따로 계산해 같은 날부터 기록 · 비교한다.
#    규칙은 시작일에 고정 — 초반 성적을 보고 고치지 않는다 (바꾸고 싶으면 v3 로 새로 시작).
BEAR_FCF_CUT = 0.30     # 하방 관문: FCF 30% 감소 · 첫 5년 성장 0 이하에서도 알파 ≥ 0
SENSITIVE_BETA = 0.6    # 베타 민감 배지: 베타 하한을 0.6 으로 올리면 가격 관문에서 탈락하는지
TOP_N = 10              # 우선순위 상위 묶음 (v1 필요 하락 작은 순 · v2 괴리 큰 순) — 매달 첫 실행에 다시 고른다
THEME_TOP = 20          # 업종 집중 경고를 볼 v2 상위 범위
THEME_WARN = 3          # 그 안에 같은 업종이 이만큼 이상이면 경고
DETAIL_NAME = "detail.json.gz"  # 분기 매출 — web_export/stock_detail.py 캐시 (같은 밤 전체분석이 먼저 쓴다)
GROUPS = ("ours", "v2", "v1_top", "v2_top", "author")  # 기록 묶음 — ours = v1 통과
MODEL_SCHEMA = 2

# 은행 · 보험 · 운용 · 증권 · 리츠는 FCF · Altman Z 가 의미 없어 뺀다. 금융데이터 · 거래소 · 결제망 · 보험중개는 남긴다.
EXCLUDE_SECTORS = frozenset({"Real Estate"})
KEEP_FIN_INDUSTRIES = frozenset({"Financial Data & Stock Exchanges", "Credit Services", "Insurance Brokers"})

# Damodaran 대형 비금융: 이자보상배율 상한 → 부도 스프레드 (Fiserv 글 표와 같음)
SPREADS: tuple[tuple[float, float], ...] = (
    (0.2, 0.19), (0.65, 0.16), (0.8, 0.1261), (1.25, 0.0885), (1.5, 0.0509), (1.75, 0.0321), (2.0, 0.0275),
    (2.25, 0.0184), (2.5, 0.0138), (3.0, 0.0111), (4.25, 0.0089), (5.5, 0.0078), (6.5, 0.0070), (8.5, 0.0055),
    (math.inf, 0.0040),
)

# ── 저자(Baoro Research) 커버리지 — 사이트 Coverage 페이지 2026-11-05. 바뀌면 여기를 고친다.
AUTHOR_AS_OF = "2026-11-05"
AUTHOR_COVERAGE: tuple[dict[str, Any], ...] = (
    {"ticker": "RHI", "target": 57.90, "ccy": "USD"},
    {"ticker": "IT", "target": 396.0, "ccy": "USD"},
    {"ticker": "FDS", "target": 423.0, "ccy": "USD"},
    {"ticker": "NICE", "target": 161.0, "ccy": "USD"},
    {"ticker": "SJ.TO", "target": 269.0, "ccy": "CAD"},
    {"ticker": "EFOR", "target": 126.0, "ccy": "USD"},
    {"ticker": "RCI", "target": 81.50, "ccy": "USD"},  # 저자 표기 'NAS: RCI' — 통화가 안 적혀 있어 미국 상장(달러)으로 본다
    {"ticker": "ENGH.TO", "target": 40.50, "ccy": "CAD"},
    {"ticker": "GIB-A.TO", "target": 306.0, "ccy": "CAD"},
    {"ticker": "PPL.TO", "target": 79.50, "ccy": "CAD"},
    {"ticker": "ACN", "target": 529.0, "ccy": "USD"},
    {"ticker": "III.TO", "target": 18.50, "ccy": "CAD"},
)
# A-Series 스크린 결과 (시황 글, 2025-09 말 · 10월 말) — (티커, 게시일). 멕시코 WALMEX 는 비교가 안 돼 뺀다.
AUTHOR_SCREEN: tuple[tuple[str, str], ...] = (
    ("NVO", "2025-10-01"), ("MRK", "2025-10-01"), ("HON", "2025-10-01"), ("HSY", "2025-10-01"),
    ("CLX", "2025-11-01"), ("GILD", "2025-11-01"), ("KMB", "2025-11-01"), ("MO", "2025-11-01"),
)
# 저자 심층 리서치 글 (Ideas = /blog, 2026-11-12 기준 44개 중) — 회사 하나를 길게 분석한 글: Model C 리서치 · 적정가 업데이트 ·
# 2026 특수상황 장문(Pembina · Imperial Metals). 2024~25 특수상황 짧은 메모 17개 · 시황 글은 뺀다.
# 커버리지와 함께 저자 종목의 두 번째 출처 — 새 리서치 글이 나오면(blog_watch 가 알린다) 여기에 한 줄 더한다.
# view = 저자 결론 (짧게), target = 글에 적힌 목표가 · 기본 시나리오 (없으면 None).
AUTHOR_POST_URL = "https://www.baorocapital.com/post/"
AUTHOR_RESEARCH: tuple[dict[str, Any], ...] = (
    {"slug": "robert-half-inc", "date": "2026-02-27", "ticker": "RHI", "kind": "리서치",
     "view": "Overweight · $57.90", "target": 57.90, "ccy": "USD"},
    {"slug": "factset-research-systems-inc", "date": "2026-02-28", "ticker": "FDS", "kind": "리서치",
     "view": "DCF 세 값 $100 · $290 · $390", "target": None, "ccy": "USD"},
    {"slug": "cgi-inc", "date": "2026-03-05", "ticker": "GIB-A.TO", "kind": "리서치",
     "view": "Strong Buy · C$306", "target": 306.0, "ccy": "CAD"},
    {"slug": "pembina-pipeline-corporation", "date": "2026-07-09", "ticker": "PPL.TO", "kind": "특수상황 장문",
     "view": "발전소 · 송유관 프로젝트가 주당 +C$10.80~14.12", "target": None, "ccy": "CAD"},
    {"slug": "red-chris-block-cave-imperial-metals", "date": "2026-07-13", "ticker": "III.TO", "kind": "특수상황 장문",
     "view": "Red Chris 블록케이브 — FCF 마진 10~40% 시나리오", "target": None, "ccy": "CAD"},
    {"slug": "tsx-engh-fair-value-update", "date": "2026-10-01", "ticker": "ENGH.TO", "kind": "적정가 업데이트",
     "view": "C$40.50 유지", "target": 40.50, "ccy": "CAD"},
    {"slug": "nyse-acn-fair-value-update", "date": "2026-10-01", "ticker": "ACN", "kind": "적정가 업데이트",
     "view": "$529 유지", "target": 529.0, "ccy": "USD"},
    {"slug": "nyse-fds-fair-value-update", "date": "2026-10-01", "ticker": "FDS", "kind": "적정가 업데이트",
     "view": "$423 로 상향", "target": 423.0, "ccy": "USD"},
    {"slug": "fiserv-inc", "date": "2026-10-05", "ticker": "FISV", "kind": "리서치",
     "view": "보수 $68.50 · 기본 $83.75 · 회복 $99", "target": 83.75, "ccy": "USD"},
)


def author_tickers() -> list[str]:
    """저자 종목 — 커버리지 + 리서치 글 (순서대로, 중복 없이). 앞으로 기록의 'author' 묶음."""
    out: list[str] = []
    for t in [c["ticker"] for c in AUTHOR_COVERAGE] + [a["ticker"] for a in AUTHOR_RESEARCH]:
        if t not in out:
            out.append(t)
    return out

# ── 받아 둘 재무 항목 (앞 이름이 빈 해는 뒤 이름으로 채운다 — series)
INC_ROWS = ("Total Revenue", "Operating Revenue", "Net Income Common Stockholders", "Net Income",
            "Net Income Continuous Operations", "EBIT", "Operating Income", "Pretax Income", "Tax Provision",
            "Interest Expense", "Interest Expense Non Operating", "Cost Of Revenue", "Reconciled Cost Of Revenue",
            "Selling General And Administration")
BAL_ROWS = ("Stockholders Equity", "Common Stock Equity", "Total Debt",
            "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "Total Assets",
            "Total Liabilities Net Minority Interest", "Current Assets", "Current Liabilities", "Retained Earnings",
            "Accounts Receivable", "Receivables", "Net PPE", "Long Term Debt And Capital Lease Obligation",
            "Long Term Debt")
CF_ROWS = ("Free Cash Flow", "Operating Cash Flow", "Capital Expenditure", "Net Issuance Payments Of Debt",
           "Net Long Term Debt Issuance", "Cash Dividends Paid", "Common Stock Dividend Paid",
           "Repurchase Of Capital Stock", "Common Stock Payments", "Issuance Of Capital Stock",
           "Common Stock Issuance", "Depreciation And Amortization", "Depreciation Amortization Depletion")
INFO_KEYS = ("marketCap", "currency", "financialCurrency", "sector", "industry", "shortName", "quoteType")

NAN = float("nan")
Fetch = Callable[[str], dict[str, Any]]


# ── 작은 도우미

def _f(v: Any) -> float | None:
    """숫자로 (NaN · inf · 못 바꾸면 None)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def _r(v: Any, digits: int = 4) -> float | None:
    x = _f(v)
    return None if x is None else round(x, digits)


def yf_symbol(ticker: str) -> str:
    """야후 기호 — 미국 종목의 점은 하이픈(BRK.B → BRK-B), 거래소 꼬리(.TO · .V · .MX)는 그대로."""
    _, dot, tail = ticker.rpartition(".")
    if dot and tail in {"TO", "V", "MX", "NE", "CN"}:
        return ticker
    return ticker.replace(".", "-")


def bench_of(ticker: str) -> str:
    """수익률 · 베타 비교 지수 — 캐나다 상장은 XIU(TSX 60), 나머지는 SPY."""
    return "XIU.TO" if ticker.endswith((".TO", ".V")) else "SPY"


# ── 받은 재무제표 → 캐시 모양

def pack_stmt(df: pd.DataFrame | None, rows: Iterable[str]) -> dict[str, dict[str, float]]:
    """yfinance 재무제표 → {항목: {'YYYY-MM-DD': 값}} — 필요한 항목의 숫자만."""
    out: dict[str, dict[str, float]] = {}
    if df is None or df.empty:
        return out
    for name in rows:
        if name not in df.index:
            continue
        s = pd.to_numeric(df.loc[name], errors="coerce")
        if isinstance(s, pd.DataFrame):
            s = s.iloc[0]
        vals = {pd.Timestamp(k).date().isoformat(): float(v) for k, v in s.items() if _f(v) is not None}
        if vals:
            out[name] = vals
    return out


def fetch_one(ticker: str) -> dict[str, Any]:
    """yfinance 로 한 종목의 연간 손익 · 대차 · 현금흐름표 + info 일부. 재무가 하나도 없으면 RuntimeError."""
    import yfinance as yf  # noqa: PLC0415

    tk = yf.Ticker(yf_symbol(ticker))
    ent: dict[str, Any] = {"inc": pack_stmt(tk.income_stmt, INC_ROWS), "bal": pack_stmt(tk.balance_sheet, BAL_ROWS),
                           "cf": pack_stmt(tk.cashflow, CF_ROWS)}
    if not (ent["inc"] or ent["bal"] or ent["cf"]):
        raise RuntimeError(f"{ticker}: 재무제표 없음")
    info = tk.info or {}
    ent["info"] = {k: info[k] for k in INFO_KEYS if info.get(k) is not None}
    return ent


def series(ent: Mapping[str, Any], stmt: str, *names: str) -> pd.Series:
    """캐시 한 종목의 재무 항목 → 연도 오름차순 Series. 앞 이름이 빈 해는 다음 이름 값으로 채운다
    (예: ACN 은 'EBIT' 이 최근 1년만 있고 'Operating Income' 은 4년치)."""
    block = ent.get(stmt) or {}
    out: pd.Series | None = None
    for n in names:
        vals = block.get(n)
        if not vals:
            continue
        s = pd.Series({pd.Timestamp(k): float(v) for k, v in vals.items()}, dtype=float)
        out = s if out is None else out.combine_first(s)
    return out.sort_index() if out is not None else pd.Series(dtype=float)


def _at(s: pd.Series, i: int) -> float:
    try:
        v = float(s.iloc[i])
    except (IndexError, TypeError, ValueError):
        return NAN
    return v if math.isfinite(v) else NAN


# ── 계산 (순수 함수)

def cv(s: pd.Series) -> float:
    """변동계수 = 표준편차 ÷ 평균. 해가 MIN_YEARS 보다 적거나 평균이 0 이하면 NaN."""
    s = s.dropna()
    if len(s) < MIN_YEARS or s.mean() <= 0:
        return NAN
    return float(s.std(ddof=0) / s.mean())


def spread_for(coverage: float) -> float:
    """이자보상배율 → 부도 스프레드 (Damodaran 대형 비금융). 이자비용이 없으면 AAA."""
    if not math.isfinite(coverage):
        return SPREADS[-1][1]
    for upper, sp in SPREADS:
        if coverage < upper:
            return sp
    return SPREADS[-1][1]


def pv_factor(r: float, g1: float, g2: float, years: int = STAGE1_YEARS) -> float:
    """시가총액 대비 현금 수익률 1 이 years 년 동안 g1, 그 뒤 g2 로 자랄 때 할인율 r 에서의 현재가치."""
    v, cf = 0.0, 1.0
    for t in range(1, years + 1):
        cf *= 1 + g1
        v += cf / (1 + r) ** t
    return v + cf * (1 + g2) / (r - g2) / (1 + r) ** years


def irr_two_stage(y0: float, g1: float, g2: float) -> float:
    """현금 수익률 y0 로 '주가 = 현재가치' 가 되는 할인율 (기대수익률). y0 ≤ 0 이면 NaN, 300% 이상은 3.0."""
    if not (math.isfinite(y0) and y0 > 0):
        return NAN
    lo, hi = g2 + 1e-4, 3.0
    if y0 * pv_factor(hi, g1, g2) > 1:
        return hi
    for _ in range(80):
        mid = (lo + hi) / 2
        if y0 * pv_factor(mid, g1, g2) > 1:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def required_yield(ke: float, g1: float) -> float:
    """가격 관문을 딱 통과하는 FCF 수익률 y* (할인율 Ke + 5%p 에서 현재가치 = 1)."""
    return 1 / pv_factor(ke + ALPHA_MIN, g1, TERM_G)


def altman_z(ent: Mapping[str, Any], mcap_fc: float) -> float:
    """Altman Z (최근 해) = 1.2 WC/TA + 1.4 RE/TA + 3.3 EBIT/TA + 0.6 시총/총부채 + 1.0 매출/TA."""
    ta = _at(series(ent, "bal", "Total Assets"), -1)
    tl = _at(series(ent, "bal", "Total Liabilities Net Minority Interest"), -1)
    ca = _at(series(ent, "bal", "Current Assets"), -1)
    cl = _at(series(ent, "bal", "Current Liabilities"), -1)
    re_ = _at(series(ent, "bal", "Retained Earnings"), -1)
    ebit = _at(series(ent, "inc", "EBIT", "Operating Income"), -1)
    rev = _at(series(ent, "inc", "Total Revenue", "Operating Revenue"), -1)
    if not (math.isfinite(ta) and ta > 0 and math.isfinite(tl) and tl > 0):
        return NAN
    return (1.2 * (ca - cl) / ta + 1.4 * re_ / ta + 3.3 * ebit / ta + 0.6 * mcap_fc / tl + 1.0 * rev / ta)


def beneish_m(ent: Mapping[str, Any]) -> float:
    """Beneish M (8변수, 최근 해 vs 직전 해). 빠진 항목은 중립값(1, TATA 는 0)."""
    rev = series(ent, "inc", "Total Revenue", "Operating Revenue").dropna()
    if len(rev) < 2:
        return NAN

    def pair(s: pd.Series) -> tuple[float, float]:
        s = s.reindex(rev.index)
        return _at(s, -1), _at(s, -2)

    def ratio(a: float, b: float) -> float:
        return a / b if (math.isfinite(a) and math.isfinite(b) and b != 0) else NAN

    def neutral(x: float, default: float = 1.0) -> float:
        return x if (math.isfinite(x) and 0 < x < 10) else default

    r_t, r_p = pair(rev)
    ar_t, ar_p = pair(series(ent, "bal", "Accounts Receivable", "Receivables"))
    cogs_t, cogs_p = pair(series(ent, "inc", "Cost Of Revenue", "Reconciled Cost Of Revenue"))
    ca_t, ca_p = pair(series(ent, "bal", "Current Assets"))
    ppe_t, ppe_p = pair(series(ent, "bal", "Net PPE"))
    ta_t, ta_p = pair(series(ent, "bal", "Total Assets"))
    dep_t, dep_p = pair(series(ent, "cf", "Depreciation And Amortization", "Depreciation Amortization Depletion"))
    sga_t, sga_p = pair(series(ent, "inc", "Selling General And Administration"))
    ni_t, _ = pair(series(ent, "inc", "Net Income Continuous Operations", "Net Income"))
    cfo_t, _ = pair(series(ent, "cf", "Operating Cash Flow"))
    cl_t, cl_p = pair(series(ent, "bal", "Current Liabilities"))
    ltd_t, ltd_p = pair(series(ent, "bal", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"))

    dsri = neutral(ratio(ratio(ar_t, r_t), ratio(ar_p, r_p)))
    gm_t, gm_p = ratio(r_t - cogs_t, r_t), ratio(r_p - cogs_p, r_p)
    gmi = neutral(ratio(gm_p, gm_t)) if (math.isfinite(gm_t) and gm_t > 0) else 1.0
    aqi = neutral(ratio(1 - ratio(ca_t + ppe_t, ta_t), 1 - ratio(ca_p + ppe_p, ta_p)))
    sgi = neutral(ratio(r_t, r_p))
    depi = neutral(ratio(ratio(dep_p, dep_p + ppe_p), ratio(dep_t, dep_t + ppe_t)))
    sgai = neutral(ratio(ratio(sga_t, r_t), ratio(sga_p, r_p)))
    tata = ratio(ni_t - cfo_t, ta_t)
    tata = tata if math.isfinite(tata) else 0.0
    lvgi = neutral(ratio(ratio(cl_t + (ltd_t if math.isfinite(ltd_t) else 0.0), ta_t),
                         ratio(cl_p + (ltd_p if math.isfinite(ltd_p) else 0.0), ta_p)))
    return (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi
            - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)


def excluded(info: Mapping[str, Any]) -> bool:
    """은행 · 보험 · 운용 · 증권 · 리츠 (금융데이터 · 거래소 · 결제망 · 보험중개는 남긴다)."""
    sector, industry = info.get("sector"), info.get("industry")
    if sector in EXCLUDE_SECTORS:
        return True
    return sector == "Financial Services" and industry not in KEEP_FIN_INDUSTRIES


def analyze(ent: Mapping[str, Any], *, price: float, mcap_fc: float, beta: float, rf: float) -> dict[str, Any]:
    """한 종목의 관문 지표 전부. mcap_fc = 재무제표 통화로 바꾼 오늘 시가총액. 계산 못 한 값은 NaN."""
    rev = series(ent, "inc", "Total Revenue", "Operating Revenue").dropna()
    ni = series(ent, "inc", "Net Income Common Stockholders", "Net Income")
    ebit = series(ent, "inc", "EBIT", "Operating Income")
    eq = series(ent, "bal", "Stockholders Equity", "Common Stock Equity")
    debt = series(ent, "bal", "Total Debt").fillna(0)
    cash = series(ent, "bal", "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents").fillna(0)
    pretax, taxp = series(ent, "inc", "Pretax Income"), series(ent, "inc", "Tax Provision")
    intexp = series(ent, "inc", "Interest Expense", "Interest Expense Non Operating")
    fcf = series(ent, "cf", "Free Cash Flow")
    if fcf.dropna().empty:
        fcf = series(ent, "cf", "Operating Cash Flow") + series(ent, "cf", "Capital Expenditure")
    fcf = fcf.dropna()

    def flow(*names: str) -> pd.Series:  # FCF 해에 맞춰 0으로 채움 (빈 Series 끼리 더하면 전부 NaN)
        return series(ent, "cf", *names).reindex(fcf.index).fillna(0)

    dnet = flow("Net Issuance Payments Of Debt", "Net Long Term Debt Issuance")
    payout = -flow("Cash Dividends Paid", "Common Stock Dividend Paid") \
        - flow("Repurchase Of Capital Stock", "Common Stock Payments") - flow("Issuance Of Capital Stock", "Common Stock Issuance")

    out: dict[str, Any] = {}
    # G1 ROE — 자기자본이 음수인 해가 있으면 판단 불가(통과 처리, MO 등)
    roe = (ni / eq).dropna()
    neg_eq = bool((eq.dropna() <= 0).any())
    out["roe"] = NAN if neg_eq or roe.empty else float(roe.mean())
    out["roe_cv"] = NAN if neg_eq else cv(roe)
    out["g_roe"] = None if neg_eq else bool(out["roe"] >= ROE_MIN and out["roe_cv"] <= CV_MAX)

    # 세율 · ROIC · WACC
    pt = _at(pretax, -1)
    tr = _at(taxp, -1) / pt if math.isfinite(pt) and pt > 0 else NAN
    tr = min(max(tr, 0.0), 0.35) if math.isfinite(tr) else 0.21
    ic = (debt + eq - cash).where(lambda s: s > 0)
    roic = (ebit * (1 - tr) / ic).dropna()
    out["roic"] = float(roic.mean()) if len(roic) else NAN
    out["roic_cv"] = cv(roic)
    ke = rf + beta * ERP
    ie = _at(intexp, -1)
    coverage = _at(ebit, -1) / abs(ie) if math.isfinite(ie) and ie != 0 else math.inf
    kd = rf + spread_for(coverage)
    d_last = _at(debt, -1) if math.isfinite(_at(debt, -1)) else 0.0
    wacc = (mcap_fc * ke + d_last * kd * (1 - tr)) / (mcap_fc + d_last) if mcap_fc > 0 else NAN
    out.update({"beta": beta, "ke": ke, "wacc": wacc})
    out["g_roic"] = bool(out["roic"] >= ROIC_MIN and out["roic_cv"] <= CV_MAX and out["roic"] > wacc)

    # G3 · G4
    out["z"] = altman_z(ent, mcap_fc)
    out["g_z"] = bool(math.isfinite(out["z"]) and out["z"] >= ALTMAN_MIN)
    out["m"] = beneish_m(ent)
    out["g_m"] = bool(math.isfinite(out["m"]) and out["m"] <= BENEISH_MAX)

    # G5 FCF
    out["fcf_years"] = int(len(fcf))
    out["g_fcf"] = bool(len(fcf) >= MIN_YEARS and (fcf > 0).all() and fcf.sum() > dnet.sum())

    # 가격 관문
    n = len(rev)
    g1 = (_at(rev, -1) / _at(rev, 0)) ** (1 / (n - 1)) - 1 if n >= 2 and _at(rev, 0) > 0 else 0.0
    g1 = min(max(g1, G1_RANGE[0]), G1_RANGE[1]) if math.isfinite(g1) else 0.0
    fcf_y = float(fcf.tail(3).mean()) / mcap_fc if mcap_fc > 0 and len(fcf) else NAN
    sh_y = float(payout.tail(3).mean()) / mcap_fc if mcap_fc > 0 and len(payout) else NAN
    irr = irr_two_stage(fcf_y, g1, TERM_G)
    out.update({"g1": g1, "fcf_y": fcf_y, "sh_y": sh_y, "irr": irr, "alpha": irr - ke,
                "alpha_author": irr_two_stage(fcf_y, g1, rf) - ke})  # 저자식: 영구 성장 = 국채금리
    out["g_price"] = bool(math.isfinite(out["alpha"]) and out["alpha"] >= ALPHA_MIN)
    y_star = required_yield(ke, g1)
    out["buy"] = price * fcf_y / y_star if math.isfinite(fcf_y) and fcf_y > 0 and price > 0 else NAN
    out["drop"] = out["buy"] / price - 1 if math.isfinite(out["buy"]) else NAN

    names = (("ROE", "g_roe"), ("ROIC", "g_roic"), ("Z", "g_z"), ("M", "g_m"), ("FCF", "g_fcf"))
    out["fails"] = [label for label, key in names if out[key] is False]
    out["quality"] = not out["fails"]
    out["pass"] = bool(out["quality"] and out["g_price"])

    # v2 — 시장이 가정하는 영구 성장률(현재가 = FCF × (1+g) ÷ (Ke − g)) · 괴리(실제 − 시장 가정) · 하방 관문 · 베타 민감
    out["implied_g"] = (ke - fcf_y) / (1 + fcf_y) if math.isfinite(fcf_y) and fcf_y > 0 else NAN
    out["gap"] = g1 - out["implied_g"] if math.isfinite(out["implied_g"]) else NAN
    out["alpha_bear"] = irr_two_stage(fcf_y * (1 - BEAR_FCF_CUT), min(g1, 0.0), TERM_G) - ke
    out["g_bear"] = bool(math.isfinite(out["alpha_bear"]) and out["alpha_bear"] >= 0)
    out["alpha_b06"] = irr - (rf + max(beta, SENSITIVE_BETA) * ERP)
    out["pass_v2"] = bool(out["pass"] and out["g_bear"])
    out["sensitive"] = bool(out["pass"] and not (math.isfinite(out["alpha_b06"]) and out["alpha_b06"] >= ALPHA_MIN))
    return out


def revenue_falling(quarters: list[Mapping[str, Any]] | None, annual: pd.Series) -> bool | None:
    """매출 감소 신호 — 최근 분기 매출이 1년 전 같은 분기(330~400일 전)보다 적고, 최근 연간 매출도 전년보다 적으면 True.
    yfinance 분기가 보통 5개뿐이라 1년 전과 비교되는 건 최근 분기 하나 — 그래서 연간과 함께 본다.
    분기 비교가 안 되거나(캐나다 등 분기 자료 없음) 연간이 2년 미만이면 None (이 조건만 빠짐)."""
    qs = [(str(q.get("end")), _f(q.get("revenue"))) for q in (quarters or []) if q.get("end")]
    qs = [(e, v) for e, v in qs if v and v > 0]
    a = annual.dropna()
    if not qs or len(a) < 2:
        return None
    end, val = qs[-1]
    d = date.fromisoformat(end)
    prev = [v for e, v in qs if 330 <= (d - date.fromisoformat(e)).days <= 400]
    if not prev:
        return None
    return bool(val < prev[0] and a.iloc[-1] < a.iloc[-2])


def kill_reasons(r: Mapping[str, Any]) -> list[str]:
    """v2 반증 조건 — 걸리면 v2 기록을 끝낸다. 없으면 빈 목록."""
    out: list[str] = []
    if not r.get("quality"):
        out.append("품질 탈락")
    if r.get("rev_down"):
        out.append("매출 감소 (최근 분기 · 연간)")
    gap, alpha = _f(r.get("gap")), _f(r.get("alpha"))
    if gap is not None and gap < 0:
        out.append("시장 가정 성장 > 실제")
    if alpha is not None and alpha < 0:
        out.append("적정가 도달")
    return out


def top_v1(rows: Iterable[Mapping[str, Any]], n: int = TOP_N) -> list[str]:
    """v1 우선순위 — 품질 통과 중 매수 기준가까지 필요한 하락이 작은 순 (통과 종목이 맨 앞)."""
    q = [r for r in rows if r["quality"] and _f(r.get("drop")) is not None]
    return [r["ticker"] for r in sorted(q, key=lambda r: -r["drop"])[:n]]


def top_v2(rows: Iterable[Mapping[str, Any]], n: int = TOP_N) -> list[str]:
    """v2 우선순위 — 품질 통과 · 반증 조건 없음 중 괴리(실제 매출 성장 − 시장이 가정하는 성장)가 큰 순.
    괴리는 '성장이 계속되면 싸다', 알파는 '성장이 5년 뒤 2.5% 로 꺾여도 싸다' 라 엇갈릴 수 있다 — 적정가 도달(알파 < 0) ·
    매출 감소처럼 반증 조건에 걸린 종목은 빼서 '시장이 지나치게 비관적이고 비싸지도 않은' 종목만 남긴다."""
    q = [r for r in rows if r["quality"] and _f(r.get("gap")) is not None and not r.get("kill")]
    return [r["ticker"] for r in sorted(q, key=lambda r: -r["gap"])[:n]]


def theme_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """v2 상위 THEME_TOP 안 업종별 종목 수 (많은 순)."""
    by_t = {r["ticker"]: r for r in rows}
    counts: dict[str, int] = {}
    for t in top_v2(by_t.values(), THEME_TOP):
        ind = by_t[t].get("industry") or "—"
        counts[ind] = counts.get(ind, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def monthly_beta(px: pd.DataFrame, ticker: str, bench: str) -> float:
    """최근 60개월 월간 수익률 회귀 베타 (36개월 미만이면 NaN)."""
    if ticker not in px.columns or bench not in px.columns:
        return NAN
    m = px[[ticker, bench]].dropna().resample("ME").last().pct_change().dropna().tail(60)
    if len(m) < 36:
        return NAN
    var = float(np.var(m[bench], ddof=1))
    return float(np.cov(m[ticker], m[bench])[0, 1] / var) if var > 0 else NAN


def price_on(px: pd.DataFrame, ticker: str, day: str | date) -> float:
    """그날(없으면 그 전 마지막 거래일) 종가. 없으면 NaN."""
    if ticker not in px.columns:
        return NAN
    s = px[ticker].dropna()
    s = s[s.index <= pd.Timestamp(day)]
    return float(s.iloc[-1]) if len(s) else NAN


def last_price(px: pd.DataFrame, ticker: str) -> float:
    if ticker not in px.columns:
        return NAN
    s = px[ticker].dropna()
    return float(s.iloc[-1]) if len(s) else NAN


# ── 받기 (시간 한도)

def refresh_order(tickers: Iterable[str], cache: Mapping[str, Mapping[str, Any]], today: date) -> list[str]:
    """다시 받을 종목 — ① 캐시에 없음 ② MAX_AGE_DAYS 일 넘음 (오래된 순)."""
    stale = (today - timedelta(days=MAX_AGE_DAYS)).isoformat()
    tiers: list[tuple[int, str, str]] = []
    for t in tickers:
        fetched = str((cache.get(t) or {}).get("fetched", ""))
        if not fetched:
            tiers.append((0, "", t))
        elif fetched <= stale:
            tiers.append((1, fetched, t))
    return [t for _, _, t in sorted(tiers)]


def refresh(tickers: Iterable[str], cache: dict[str, dict[str, Any]], today: date, *, budget_sec: float,
            fetch: Fetch = fetch_one, workers: int = WORKERS, log: Callable[[str], None] = print) -> dict[str, int]:
    """시간 한도 안에서 cache 를 고쳐 쓴다 → {"want", "ok", "fail", "left"}. 못 받은 종목은 전에 받은 값 그대로."""
    order = refresh_order(tickers, cache, today)
    start = time.monotonic()
    ok = fail = streak = done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        it = iter(order)
        running: dict[Any, str] = {}

        def submit_next() -> bool:
            if time.monotonic() - start > budget_sec or streak >= MAX_FAIL_STREAK:
                return False
            t = next(it, None)
            if t is None:
                return False
            running[pool.submit(fetch, t)] = t
            return True

        for _ in range(workers):
            if not submit_next():
                break
        while running:
            fut = next(as_completed(list(running)))
            t = running.pop(fut)
            done += 1
            try:
                cache[t] = {**fut.result(), "fetched": today.isoformat()}
                ok += 1
                streak = 0
            except Exception:  # noqa: BLE001 — 표시용, 한 종목 실패는 넘어간다
                fail += 1
                streak += 1
            submit_next()
    if streak >= MAX_FAIL_STREAK:
        log(f"[예측 모델] 연속 {streak}번 실패 — 막힌 것 같아 그만 받음 (전에 받은 값 사용)")
    stats = {"want": len(order), "ok": ok, "fail": fail, "left": len(order) - done}
    log(f"[예측 모델] 재무 받을 {stats['want']} · 받음 {ok} · 실패 {fail} · 다음 밤으로 {stats['left']} "
        f"({(time.monotonic() - start) / 60:.1f}분)")
    return stats


def load_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    """전날 재무 캐시 (없거나 못 읽으면 빈 dict)."""
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
        return {str(k): v for k, v in data.get("items", {}).items() if isinstance(v, dict)}
    except (OSError, ValueError) as exc:
        print(f"[예측 모델] 캐시 읽기 실패 (처음부터): {exc}")
        return {}


def save_cache(path: Path, cache: Mapping[str, Mapping[str, Any]], keep: set[str] | None = None) -> int:
    """재무 캐시 저장 — keep 이 있으면 그 종목만. 바이트 수."""
    items = {t: v for t, v in cache.items() if keep is None or t in keep}
    body = json.dumps({"schema": SCHEMA, "items": items}, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    packed = gzip.compress(body.encode("utf-8"), compresslevel=9, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(packed)
    return len(packed)


# ── 앞으로 기록

def load_log(path: Path | None) -> dict[str, Any]:
    """기록 (없거나 못 읽으면 빈 기록)."""
    if path and path.exists():
        try:
            data = json.loads(path.read_text("utf-8"))
            if isinstance(data, dict) and isinstance(data.get("entries"), list):
                return data
        except (OSError, ValueError) as exc:
            print(f"[예측 모델] 기록 읽기 실패 (새로 시작): {exc}")
    return {"schema": SCHEMA, "start": None, "entries": []}


def update_log(log: dict[str, Any], rows: list[Mapping[str, Any]], px: pd.DataFrame, day: str) -> dict[str, Any]:
    """오늘 결과를 기록에 반영한다. 한 번 넣은 줄의 시작일 · 시작가는 바꾸지 않는다.
    - ours(v1 통과) · author(저자 종목 = 커버리지 + 리서치 글): 처음 한 번 넣고 계속 보유.
    - v2: v2 통과하면 넣고, 반증 조건(kill_reasons)에 걸리면 끝낸다(x · xr). 다시 통과하면 새 줄.
    - v1_top · v2_top: 매달 첫 실행에 우선순위 상위 TOP_N 으로 다시 골라, 빠진 종목은 끝내고 새 종목을 넣는다."""
    entries: list[dict[str, Any]] = log.setdefault("entries", [])
    if not log.get("start"):
        log["start"] = day
    baskets: dict[str, str] = log.setdefault("baskets", {})
    by_t = {r["ticker"]: r for r in rows}

    def active(t: str, group: str) -> dict[str, Any] | None:
        return next((e for e in entries if e.get("t") == t and e.get("g") == group and not e.get("x")), None)

    def ever(t: str, group: str) -> bool:
        return any(e.get("t") == t and e.get("g") == group for e in entries)

    def add(t: str, group: str) -> None:
        bench = bench_of(t)
        p, bp = price_on(px, t, day), price_on(px, bench, day)
        if math.isfinite(p) and math.isfinite(bp):
            entries.append({"t": t, "g": group, "d": day, "p": round(p, 4), "bench": bench, "bp": round(bp, 4)})

    def close(e: dict[str, Any], reason: str) -> None:
        e["x"], e["xr"] = day, reason

    for r in rows:
        if r.get("pass") and not ever(r["ticker"], "ours"):
            add(r["ticker"], "ours")
    for e in [e for e in entries if e.get("g") == "v2" and not e.get("x")]:
        r = by_t.get(str(e.get("t")))
        if r is not None and r.get("kill"):
            close(e, " · ".join(r["kill"]))
    for r in rows:
        if r.get("pass_v2") and not active(r["ticker"], "v2"):
            add(r["ticker"], "v2")
    month = day[:7]
    for group, picks in (("v1_top", top_v1(rows)), ("v2_top", top_v2(rows))):
        if baskets.get(group) == month or not picks:
            continue
        keep = set(picks)
        for e in [e for e in entries if e.get("g") == group and not e.get("x")]:
            if e.get("t") not in keep:
                close(e, "순위 밖 (월 교체)")
        for t in picks:
            if not active(t, group):
                add(t, group)
        baskets[group] = month
    for t in author_tickers():  # 커버리지 + 리서치 글 종목 — 처음 보이는 날부터
        if not ever(t, "author"):
            add(t, "author")
    return log


def tracking_rows(log: Mapping[str, Any], px: pd.DataFrame,
                  rows_by: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """기록 줄마다 수익률 (배당 포함 수정주가로 시작일 → 끝낸 날 또는 오늘) · 비교 지수 수익률 · 차이."""
    out: list[dict[str, Any]] = []
    for e in log.get("entries", []):
        t, group, day = e.get("t"), e.get("g"), e.get("d")
        if not t or not day:
            continue
        bench, end = e.get("bench") or bench_of(str(t)), e.get("x")
        p0 = price_on(px, t, day)
        p1 = price_on(px, t, end) if end else last_price(px, t)
        b0 = price_on(px, bench, day)
        b1 = price_on(px, bench, end) if end else last_price(px, bench)
        ret = p1 / p0 - 1 if p0 > 0 else NAN
        bret = b1 / b0 - 1 if b0 > 0 else NAN
        r = rows_by.get(t)
        passing = bool(r and (r.get("pass_v2") if group == "v2" else r.get("pass")))
        out.append({"ticker": t, "group": group, "since": day, "entry": e.get("p"), "bench": bench, "ret": ret,
                    "bench_ret": bret, "excess": ret - bret, "passing": passing, "exit": end, "reason": e.get("xr")})
    return out


def tracking_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """묶음별 개수 · 평균 수익률 · 평균 초과 · 지수를 이긴 수 · 아직 들고 있는 수."""
    out: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        rs = [r for r in rows if r["group"] == group and _f(r["excess"]) is not None]
        if not rs:
            out[group] = {"n": 0}
            continue
        out[group] = {"n": len(rs), "ret": float(np.mean([r["ret"] for r in rs])),
                      "excess": float(np.mean([r["excess"] for r in rs])),
                      "beat": int(sum(1 for r in rs if r["excess"] > 0)),
                      "open": int(sum(1 for r in rs if not r.get("exit")))}
    return out


# ── 주가 · 금리 · 환율

def download_prices(tickers: Iterable[str]) -> pd.DataFrame:
    """5년 일봉 종가 (배당 포함 수정주가) — 열 = 우리 티커."""
    import yfinance as yf  # noqa: PLC0415

    syms = {yf_symbol(t): t for t in tickers}
    raw = yf.download(list(syms), period="5y", interval="1d", auto_adjust=True, progress=False, threads=True,
                      group_by="column")
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].set_axis([next(iter(syms))], axis=1)
    close = close.rename(columns=syms)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.sort_index()


def risk_free() -> tuple[float, str]:
    """10년 국채금리 (^TNX). 못 받으면 RF_FALLBACK."""
    try:
        import yfinance as yf  # noqa: PLC0415

        s = yf.Ticker("^TNX").history(period="10d")["Close"].dropna()
        v = float(s.iloc[-1]) / 100
        return (v / 10 if v > 0.2 else v), "^TNX"
    except Exception as exc:  # noqa: BLE001
        print(f"[예측 모델] 10년 국채 못 받음 — {RF_FALLBACK:.2%} 로 계산: {exc}")
        return RF_FALLBACK, "fallback"


def fx_rates(pairs: Iterable[str]) -> dict[str, float]:
    """'DKKUSD' 같은 통화 쌍 → 1 단위 값 (못 받으면 빠짐)."""
    import yfinance as yf  # noqa: PLC0415

    out: dict[str, float] = {}
    for pair in sorted(set(pairs)):
        try:
            s = yf.Ticker(f"{pair}=X").history(period="10d")["Close"].dropna()
            if len(s):
                out[pair] = float(s.iloc[-1])
        except Exception:  # noqa: BLE001
            continue
    return out


# ── 전체

def screen(tickers: Iterable[str], cache: Mapping[str, Mapping[str, Any]], px: pd.DataFrame, rf: float,
           fx: Mapping[str, float], detail: Mapping[str, Mapping[str, Any]] | None = None,
           ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """종목마다 analyze → (줄 목록, 깔때기 숫자). 시가총액은 받은 날 값 × (오늘 주가 ÷ 받은 날 주가).
    detail = stock_detail 캐시(분기 매출) — v2 반증 조건 '매출 감소 (최근 분기 전년 대비 · 연간 둘 다)'에 쓴다 (없으면 그 조건만 빠짐)."""
    rows: list[dict[str, Any]] = []
    funnel = {"universe": 0, "excluded": 0, "no_data": 0, "computed": 0}
    for t in tickers:
        funnel["universe"] += 1
        ent = cache.get(t)
        info = (ent or {}).get("info") or {}
        if not ent:
            funnel["no_data"] += 1
            continue
        if excluded(info):
            funnel["excluded"] += 1
            continue
        price = last_price(px, t)
        mcap = _f(info.get("marketCap"))
        if not (price > 0 and mcap):
            funnel["no_data"] += 1
            continue
        fetched = ent.get("fetched")
        p_then = price_on(px, t, str(fetched)) if fetched else NAN
        mcap_now = mcap * (price / p_then) if p_then > 0 else mcap
        cur = info.get("currency") or "USD"
        fin = info.get("financialCurrency") or cur
        rate = 1.0 if cur == fin else fx.get(f"{fin}{cur}", NAN)
        if not (math.isfinite(rate) and rate > 0):
            funnel["no_data"] += 1
            continue
        beta = monthly_beta(px, t, bench_of(t))
        beta = min(max(beta, BETA_RANGE[0]), BETA_RANGE[1]) if math.isfinite(beta) else 1.0
        res = analyze(ent, price=price, mcap_fc=mcap_now / rate, beta=beta, rf=rf)
        s = px[t].dropna() if t in px.columns else pd.Series(dtype=float)
        res["dd_3y"] = float(s.iloc[-1] / s.iloc[-756:].max() - 1) if len(s) > 260 else NAN
        res.update({"ticker": t, "name": info.get("shortName"), "sector": info.get("sector"),
                    "industry": info.get("industry"), "ccy": cur, "price": price, "fetched": ent.get("fetched")})
        res["rev_down"] = revenue_falling(((detail or {}).get(t) or {}).get("revenue"),
                                          series(ent, "inc", "Total Revenue", "Operating Revenue"))
        res["kill"] = kill_reasons(res)
        rows.append(res)
        funnel["computed"] += 1
    for key, label in (("g_roe", "roe"), ("g_roic", "roic"), ("g_z", "z"), ("g_m", "m"), ("g_fcf", "fcf"),
                       ("g_price", "price")):
        funnel[label] = sum(1 for r in rows if r[key] is True or (key == "g_roe" and r[key] is None))
    funnel["quality"] = sum(1 for r in rows if r["quality"])
    funnel["pass"] = sum(1 for r in rows if r["pass"])
    funnel["pass_v2"] = sum(1 for r in rows if r["pass_v2"])
    funnel["sensitive"] = sum(1 for r in rows if r["sensitive"])
    return rows, funnel


def author_kind(ticker: str) -> str | None:
    """저자 커버리지('coverage') · 리서치 글에만('research') · A-Series 스크린('screen') · 아님(None)."""
    if any(c["ticker"] == ticker for c in AUTHOR_COVERAGE):
        return "coverage"
    if any(a["ticker"] == ticker for a in AUTHOR_RESEARCH):
        return "research"
    if any(t == ticker for t, _ in AUTHOR_SCREEN):
        return "screen"
    return None


def research_rows(px: pd.DataFrame, rows_by: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """저자 리서치 글마다 게시일 종가 → 오늘 수익률(배당 포함) · 같은 기간 지수(SPY · 캐나다는 XIU) · 목표가까지."""
    out: list[dict[str, Any]] = []
    for a in AUTHOR_RESEARCH:
        t = a["ticker"]
        bench = bench_of(t)
        p0, p1 = price_on(px, t, a["date"]), last_price(px, t)
        b0, b1 = price_on(px, bench, a["date"]), last_price(px, bench)
        ret = p1 / p0 - 1 if p0 > 0 else NAN
        bret = b1 / b0 - 1 if b0 > 0 else NAN
        r = rows_by.get(t)
        price = _f(r.get("price")) if r else None
        if price is None and math.isfinite(p1):
            price = p1
        target = a.get("target")
        out.append({**a, "url": AUTHOR_POST_URL + a["slug"], "bench": bench, "price": _r(price, 2),
                    "ret": _r(ret), "bench_ret": _r(bret), "excess": _r(ret - bret),
                    "upside": _r(target / price - 1) if target and price else None})
    return out


def web_row(r: Mapping[str, Any]) -> dict[str, Any]:
    """사이트에 보낼 한 줄 (숫자는 반올림, NaN 은 null)."""
    keys4 = ("roe", "roe_cv", "roic", "roic_cv", "wacc", "beta", "ke", "g1", "fcf_y", "sh_y", "irr", "alpha",
             "alpha_author", "drop", "dd_3y", "implied_g", "gap", "alpha_bear", "alpha_b06")
    row: dict[str, Any] = {"ticker": r["ticker"], "name": r.get("name"), "sector": r.get("sector"),
                           "industry": r.get("industry"), "ccy": r.get("ccy"), "price": _r(r.get("price"), 2),
                           "buy": _r(r.get("buy"), 2), "z": _r(r.get("z"), 2), "m": _r(r.get("m"), 2),
                           "fcf_years": r.get("fcf_years"), "fetched": r.get("fetched")}
    row.update({k: _r(r.get(k), 4) for k in keys4})
    row["gates"] = {"roe": r["g_roe"], "roic": r["g_roic"], "z": r["g_z"], "m": r["g_m"], "fcf": r["g_fcf"],
                    "price": r["g_price"]}
    row.update({"fails": list(r["fails"]), "quality": r["quality"], "pass": r["pass"],
                "author": author_kind(r["ticker"])})
    row.update({"g_bear": r.get("g_bear"), "pass_v2": r.get("pass_v2"), "sensitive": r.get("sensitive"),
                "rev_down": r.get("rev_down"), "kill": list(r.get("kill") or [])})
    return row


def build_model(rows: list[dict[str, Any]], funnel: Mapping[str, int], *, price_date: str, rf: float, rf_src: str,
                track: list[dict[str, Any]], log_start: str | None, fetch_stats: Mapping[str, int] | None,
                research: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """value_model.json — 관문을 하나라도 의미 있게 지난 종목(품질 통과 · 가격 통과 · 저자 종목)만 싣는다.
    research = research_rows (저자 리서치 글 · 게시일 이후 성과)."""
    keep = [r for r in rows if r["quality"] or r["g_price"] or author_kind(r["ticker"])]
    keep.sort(key=lambda r: (not r["pass"], -(r["alpha"] if math.isfinite(r["alpha"]) else -9)))
    by_t = {r["ticker"]: r for r in rows}
    coverage = []
    for c in AUTHOR_COVERAGE:
        r = by_t.get(c["ticker"])
        coverage.append({**c, "price": _r(r["price"], 2) if r else None,
                         "upside": _r(c["target"] / r["price"] - 1, 4) if r and r["price"] > 0 else None})
    track_web = [{**t, "ret": _r(t["ret"]), "bench_ret": _r(t["bench_ret"]), "excess": _r(t["excess"])} for t in track]
    return {
        "schema": MODEL_SCHEMA,
        "date": price_date,
        "generated": datetime.now(ZoneInfo("America/Toronto")).isoformat(timespec="minutes"),
        "params": {"rf": _r(rf), "rf_src": rf_src, "erp": ERP, "term_g": TERM_G, "alpha_min": ALPHA_MIN,
                   "stage1_years": STAGE1_YEARS, "roe_min": ROE_MIN, "roic_min": ROIC_MIN, "cv_max": CV_MAX,
                   "altman_min": ALTMAN_MIN, "beneish_max": BENEISH_MAX},
        "funnel": dict(funnel),
        "fetch": dict(fetch_stats or {}),
        "rows": [web_row(r) for r in keep],
        "author": {"as_of": AUTHOR_AS_OF, "coverage": coverage,
                   "screen": [{"ticker": t, "date": d} for t, d in AUTHOR_SCREEN],
                   "research": list(research or [])},
        "tracking": {"start": log_start, "rows": track_web, "summary": tracking_summary(track)},
        "v2": {
            "params": {"bear_fcf_cut": BEAR_FCF_CUT, "sensitive_beta": SENSITIVE_BETA, "top_n": TOP_N,
                       "theme_top": THEME_TOP, "theme_warn": THEME_WARN},
            "top": {"v1": top_v1(rows), "v2": top_v2(rows)},
            "theme": theme_counts(rows),
        },
    }


def universe_tickers() -> list[str]:
    """S&P 500 (data.sp500_tickers) + 저자 종목(커버리지 · 리서치 글) · A-Series 종목."""
    from data.sp500_tickers import SP500_TICKERS  # noqa: PLC0415

    extra = author_tickers() + [t for t, _ in AUTHOR_SCREEN]
    return sorted(set(SP500_TICKERS) | set(extra))


def main(argv: list[str] | None = None) -> int:
    """밤 실행 진입점 — 재무 캐시를 조금 받고, 주가로 다시 계산해 value_model.json · 캐시 · 기록을 쓴다."""
    ap = argparse.ArgumentParser(description="예측 모델 (가치 체크리스트 스크린) — 표시 전용")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prev-dir", type=Path, default=None)
    ap.add_argument("--budget-min", type=float, default=BUDGET_MIN, help="재무 받기 시간 한도(분)")
    ap.add_argument("--tickers", default="", help="쉼표로 구분 (시험용 — 없으면 S&P 500 + 저자 종목)")
    ap.add_argument("--no-fetch", action="store_true", help="재무를 새로 받지 않고 캐시만 쓴다")
    ap.add_argument("--detail", type=Path, default=None,
                    help="분기 매출 캐시 (없으면 OUT/snapshot · PREV 의 detail.json.gz)")
    args = ap.parse_args(argv)

    out_dir: Path = args.out_dir
    snap = out_dir / "snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    prev: Path | None = args.prev_dir
    today = datetime.now(ZoneInfo("America/Toronto")).date()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] or universe_tickers()

    cache = load_cache(prev / CACHE_NAME if prev else None)
    log = load_log(prev / LOG_NAME if prev else None)
    fetch_stats: dict[str, int] | None = None
    try:
        if not args.no_fetch:
            fetch_stats = refresh(tickers, cache, today, budget_sec=args.budget_min * 60)
    finally:
        size = save_cache(snap / CACHE_NAME, cache, keep=set(tickers))
        print(f"[예측 모델] 재무 캐시 {len(cache)}종목 · {size / 1e6:.2f}MB")

    benches = sorted({bench_of(t) for t in tickers} | {"SPY"})
    px = download_prices(tickers + benches)
    if px.empty or "SPY" not in px.columns:
        print("[예측 모델] 주가를 못 받음 — 전날 결과 유지")
        (snap / LOG_NAME).write_text(json.dumps(log, ensure_ascii=False, indent=1), "utf-8")
        if prev and (prev / OUT_NAME).exists():
            shutil.copy(prev / OUT_NAME, out_dir / OUT_NAME)
        return 1
    price_date = px["SPY"].dropna().index[-1].date().isoformat()
    rf, rf_src = risk_free()
    pairs = set()
    for t in tickers:
        info = (cache.get(t) or {}).get("info") or {}
        cur, fin = info.get("currency"), info.get("financialCurrency")
        if cur and fin and cur != fin:
            pairs.add(f"{fin}{cur}")
    fx = fx_rates(pairs)

    from web_export.stock_detail import load_cache as load_detail  # noqa: PLC0415 — 같은 밤 전체분석이 쓴 분기 매출

    detail_path = args.detail or next((p for p in (snap / DETAIL_NAME, prev / DETAIL_NAME if prev else None)
                                       if p is not None and p.exists()), None)
    detail = load_detail(detail_path)
    print(f"[예측 모델] 분기 매출 {len(detail)}종목 ({detail_path or '없음'})")

    rows, funnel = screen(tickers, cache, px, rf, fx, detail)
    passing = {r["ticker"] for r in rows if r["pass"]}
    log = update_log(log, rows, px, price_date)
    (snap / LOG_NAME).write_text(json.dumps(log, ensure_ascii=False, indent=1), "utf-8")
    by_ticker = {r["ticker"]: r for r in rows}
    track = tracking_rows(log, px, by_ticker)
    model = build_model(rows, funnel, price_date=price_date, rf=rf, rf_src=rf_src, track=track,
                        log_start=log.get("start"), fetch_stats=fetch_stats, research=research_rows(px, by_ticker))
    body = json.dumps(model, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (out_dir / OUT_NAME).write_text(body, "utf-8")
    print(f"[예측 모델] {price_date} · 계산 {funnel['computed']} · 품질 통과 {funnel['quality']} · 가격 통과 "
          f"{funnel['price']} · 둘 다 {funnel['pass']} ({', '.join(sorted(passing)) or '없음'}) · v2 통과 "
          f"{funnel['pass_v2']} · 기록 {len(track)}줄 · {len(body) / 1e3:.0f}KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
