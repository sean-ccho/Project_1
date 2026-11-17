#!/usr/bin/env python3
"""예측 모델 v3 재검증 (2026-11-16, 사전 등록 · 한 번만 실행).

docs/월스트리트_저평가주식_분석_및_v3제안.md 4부(Antigravity · Gemini, 11-14)의 백테스트를 고친 데이터로 다시 잰다.
리뷰에서 찾은 문제 — SPY 배당 빠짐 · 종목 가격 기준 섞임(차트 캐시 vs 배당 포함) · TSX 티커를 앞 글자로 SEC 회사에 연결
(PPL.TO → PPL Corp 등) · 과거 시총 = 오늘 시총 × 가격비(주식 수 · 배당 무시) · sc_to_yf_ratio · 회계연도 단위 시점 확인
· 10년 백테스트 v2 매출 감소 꺼짐 · '오늘 살아 있는 $1B 이상' 유니버스 — 를 고친다.

규칙(v1 · v2 = 지금 value_model.py, v3 = Gemini run_10y_backtest.py 의 evaluate_point_in_time · select_portfolios 그대로)과
판정 기준은 실행 전에 docs/quant_improvement/tier3/TRIAL_LOG.csv 에 등록했다 (value_v3_recheck). 결과를 보고 고치지 않는다.

  fetch-sec     SEC companyfacts (회사 이름 · 연간/분기 재무 · 값마다 공시일) → data/research/value_v3/companyfacts/
  fetch-prices  yfinance 월봉 (분할만 반영 Close · 배당 포함 Adj Close · 분할) + SPY · ^TNX · 환율
  build         companyfacts → 필요한 항목만 (facts.pkl)
  run           매년 5월 말 리밸런싱 → 다음 5월 말 (마지막 2026-10-31), 월간 자본곡선 · 거래비용 · evaluation.py 판정

사용: SEC_USER_AGENT="이름 이메일" PYTHONPATH=.:src python scripts/value_v3_backtest.py fetch-sec
      PYTHONPATH=.:src python scripts/value_v3_backtest.py fetch-prices | build | run
연구 전용 — 점수 · 필터 · PT-1 · 사이트와 무관.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import pickle
import random
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_trading.evaluation import (alpha_beta, cagr, deflated_sharpe, max_drawdown,  # noqa: E402
                                      paired_bootstrap, sharpe, verdict)
from web_export.value_model import (ALPHA_MIN, BEAR_FCF_CUT, BETA_RANGE, G1_RANGE, TERM_G, TOP_N,  # noqa: E402
                                    _at, analyze, excluded, irr_two_stage, kill_reasons, revenue_falling, series)
from web_export.value_model import MIN_YEARS as MIN_YEARS_V  # noqa: E402

NAN_F = float("nan")
# pandas pct_change 기본값(fill_method) 변경 예고 — 계산은 그대로 두고 경고만 끈다 (등록한 실행과 같은 결과)
warnings.filterwarnings("ignore", message="The default fill_method", category=FutureWarning)

OUT = ROOT / "data" / "research" / "value_v3"
CF_DIR = OUT / "companyfacts"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{:010d}.json"
SEC_RATE = 8.0                       # 초당 요청 (SEC 한도 10)
MEMBERSHIP_CSV = ROOT / "data" / "universe" / "sp500_membership.csv"
SECTOR_MAP_CSV = ROOT / "data" / "universe" / "sector_map.csv"
EARN_DATES = ROOT / "data" / "research" / "sec_earnings_dates.parquet"
SCAN_RANKED = ("snapshot/sp500_ranked.parquet", "snapshot/nasdaq_ranked.parquet")
SCAN_BRANCH = "origin/universe-data"
PRICE_START = "2009-01-01"
# 지주회사 재편으로 CIK 가 새로 생긴 큰 회사 — 지금 티커가 이력 없는 새 CIK 로 연결돼 옛 CIK 재무를 합친다 (결과와 무관한 연결 보정)
PREDECESSOR_CIK = {"XOM": 34088, "GOOGL": 1288776, "GOOG": 1288776, "MDT": 64670, "APA": 6769}
# 재무제표 통화 → 'XXXUSD=X' 월봉 (USD 1 단위 = 값). 목록 밖 통화를 쓰는 회사는 빠진다(셈).
FX_CCYS = ("EUR", "GBP", "JPY", "CNY", "HKD", "TWD", "KRW", "INR", "BRL", "MXN", "CAD", "AUD", "CHF", "SEK", "NOK",
           "DKK", "ZAR", "CLP", "COP", "PEN", "ARS", "ILS", "IDR", "PHP", "THB", "SGD", "NZD", "TRY", "RUB", "KZT",
           "PLN", "CZK", "HUF", "MYR", "SAR", "AED", "VND", "EGP", "NGN")


# ── 작은 도우미

def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def norm_ticker(t: str) -> str:
    """BRK.B · brk-b → BRK-B (yfinance · SEC 표기)."""
    return str(t).strip().upper().replace(".", "-")


def git_blob(ref_path: str) -> bytes:
    return subprocess.run(["git", "show", ref_path], cwd=ROOT, capture_output=True, check=True).stdout


def load_scan() -> dict[str, dict[str, Any]]:
    """지금 전체분석 목록(미국 S&P 500 + 나스닥 · NYSE) — 티커 → 업종 · 오늘 시가총액(USD)."""
    out: dict[str, dict[str, Any]] = {}
    for path in SCAN_RANKED:
        df = pd.read_parquet(io.BytesIO(git_blob(f"{SCAN_BRANCH}:{path}")))
        mkt = "sp500" if "sp500" in path else "nasdaq"
        for r in df.itertuples(index=False):
            t = norm_ticker(getattr(r, "티커"))
            mcap = getattr(r, "fund_market_cap")
            out.setdefault(t, {"sector": getattr(r, "fund_sector") or None, "industry": getattr(r, "fund_industry") or None,
                               "mcap_now": float(mcap) if mcap is not None and np.isfinite(mcap) else None, "market": mkt})
    return out


def load_membership(since: str = "2013-01-01") -> pd.DataFrame:
    m = pd.read_csv(MEMBERSHIP_CSV)
    m["date"] = pd.to_datetime(m["date"])
    return m[m["date"] >= pd.Timestamp(since)].sort_values("date").reset_index(drop=True)


def members_on(mem: pd.DataFrame, d: str) -> set[str]:
    sub = mem[mem["date"] <= pd.Timestamp(d)]
    if sub.empty:
        return set()
    return {norm_ticker(t) for t in str(sub.iloc[-1]["tickers"]).split(",") if t.strip()}


# ── SEC 받기

class SecClient:
    """SEC 요청 — User-Agent 에 연락처(환경변수), 초당 SEC_RATE 회 이하, 429 · 5xx 면 쉬고 다시."""

    def __init__(self, ua: str, rate: float = SEC_RATE) -> None:
        import requests
        self._s = requests.Session()
        self._s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self._gap = 1.0 / rate
        self._lock = threading.Lock()
        self._last = 0.0

    def get(self, url: str) -> tuple[int, bytes | None]:
        for attempt in range(5):
            with self._lock:
                wait = self._gap - (time.monotonic() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
            try:
                r = self._s.get(url, timeout=60)
            except Exception:
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code == 404:
                return 404, None
            if r.status_code in (403, 429) or r.status_code >= 500:
                time.sleep(5.0 * (attempt + 1))
                continue
            return r.status_code, r.content
        return -1, None


def build_candidates(sec_rows: Sequence[Sequence[Any]], scan: Mapping[str, Mapping[str, Any]],
                     mem: pd.DataFrame, earn: Mapping[str, int] | None = None) -> dict[str, dict[str, Any]]:
    """후보 = 지금 전체분석 목록(미국 상장 보통주) + 2013년 이후 S&P 500 에 있었던 티커.
    SEC 연결은 SEC 공식 티커 목록(정확히 같은 티커)만 — TSX 티커를 앞 글자로 자르지 않는다(Gemini 버그).
    캐나다 동시 상장 회사는 미국 티커(TU · CNI · CNQ …)로 들어온다. 목록에 없는 옛 S&P 티커는 sec_earnings_dates(옛 연구의
    티커 → CIK)로 잇고, 가격은 그 CIK 의 지금 티커로 받는다(px)."""
    by_t: dict[str, tuple[int, str]] = {}
    by_cik: dict[int, list[tuple[str, str]]] = {}
    for cik, _name, tk, ex in sec_rows:
        if not tk:
            continue
        t = norm_ticker(tk)
        by_t.setdefault(t, (int(cik), str(ex or "")))
        by_cik.setdefault(int(cik), []).append((t, str(ex or "")))
    hist: set[str] = set()
    for s in mem["tickers"].dropna():
        hist.update(norm_ticker(t) for t in str(s).split(",") if t.strip())
    out: dict[str, dict[str, Any]] = {}
    for t in sorted(set(scan) | hist):
        hit = by_t.get(t)
        cik = hit[0] if hit else (earn or {}).get(t)
        if not cik:
            out[t] = {"cik": None, "alt": None, "scan": t in scan, "member": t in hist, "px": None}
            continue
        # 옛 CIK 가 지금 CIK 와 다르면(지주회사 재편: XOM 2115436 ← 34088 등) 이력을 합치려고 같이 받는다
        alt = PREDECESSOR_CIK.get(t) or (earn or {}).get(t)
        alt = int(alt) if alt and int(alt) != int(cik) else None
        px = t if hit else None
        if px is None:  # 옛 티커 — 같은 CIK 의 지금 티커 (전체분석 목록에 있는 것 → 거래소 상장 → 아무거나)
            cur = by_cik.get(int(cik), [])
            pick = [x for x, _ in cur if x in scan] or [x for x, ex in cur if ex in ("NYSE", "Nasdaq")] or [x for x, _ in cur]
            px = pick[0] if pick else None
        out[t] = {"cik": int(cik), "alt": alt, "scan": t in scan, "member": t in hist, "px": px}
    return out


def cmd_fetch_sec(_args: argparse.Namespace) -> int:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        sys.exit("환경변수 SEC_USER_AGENT='이름 이메일' 이 필요합니다 (SEC 규칙)")
    CF_DIR.mkdir(parents=True, exist_ok=True)
    cli = SecClient(ua)
    st, raw = cli.get(SEC_TICKERS_URL)
    if st != 200 or not raw:
        sys.exit(f"SEC 티커 목록을 못 받음 ({st})")
    (OUT / "company_tickers_exchange.json").write_bytes(raw)
    sec = json.loads(raw)
    scan = load_scan()
    mem = load_membership()
    earn: dict[str, int] = {}
    if EARN_DATES.exists():
        ed = pd.read_parquet(EARN_DATES)
        for t, c in zip(ed["티커"], ed["cik"]):
            if pd.notna(t) and pd.notna(c):
                earn.setdefault(norm_ticker(t), int(c))
    cands = build_candidates(sec["data"], scan, mem, earn)
    (OUT / "candidates.json").write_text(json.dumps(cands, ensure_ascii=False, indent=0), "utf-8")
    ciks = sorted({c["cik"] for c in cands.values() if c["cik"]} | {c["alt"] for c in cands.values() if c.get("alt")})
    todo = [c for c in ciks if not (CF_DIR / f"CIK{c:010d}.json.gz").exists()]
    n_scan = sum(1 for c in cands.values() if c["scan"])
    n_mem = sum(1 for c in cands.values() if c["member"])
    log(f"후보 {len(cands)} (전체분석 {n_scan} · 옛 S&P {n_mem}) · CIK 연결 {sum(1 for c in cands.values() if c['cik'])} · "
        f"고유 CIK {len(ciks)} · 받을 것 {len(todo)}")
    t0, done, miss = time.monotonic(), 0, 0

    def one(cik: int) -> tuple[int, int]:
        st, body = cli.get(FACTS_URL.format(cik))
        if st == 200 and body:
            (CF_DIR / f"CIK{cik:010d}.json.gz").write_bytes(gzip.compress(body, 6))
        return cik, st

    with ThreadPoolExecutor(max_workers=6) as ex:
        for fut in as_completed([ex.submit(one, c) for c in todo]):
            _cik, st = fut.result()
            done += 1
            miss += st != 200
            if done % 250 == 0 or done == len(todo):
                log(f"  SEC {done}/{len(todo)} · 없음/실패 {miss} · {time.monotonic() - t0:.0f}s")
    log("SEC 받기 끝")
    return 0


# ── 가격 받기

def _split_frame(df: pd.DataFrame, tickers: Sequence[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            sub = pd.DataFrame({"close": df["Close"][t], "adj": df["Adj Close"][t],
                                "split": df["Stock Splits"][t] if "Stock Splits" in df else 0.0})
        except KeyError:
            continue
        sub = sub.dropna(subset=["close", "adj"])
        if len(sub) >= 2:
            sub.index = pd.to_datetime(sub.index).to_period("M").to_timestamp("M")
            out[t] = sub[~sub.index.duplicated(keep="last")]
    return out


def download_monthly(tickers: Sequence[str], batch: int = 150) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), batch):
        part = tickers[i:i + batch]
        for attempt in range(3):
            try:
                df = yf.download(part, start=PRICE_START, end=(date.today() + pd.Timedelta(days=1)).isoformat(),
                                 interval="1mo", auto_adjust=False, actions=True, progress=False, group_by="column",
                                 threads=True)
                if len(part) == 1 and not isinstance(df.columns, pd.MultiIndex):
                    df.columns = pd.MultiIndex.from_product([df.columns, part])
                got = _split_frame(df, part)
                out.update(got)
                break
            except Exception as e:  # Yahoo 막힘 · 일시 오류
                log(f"  가격 {i}: {type(e).__name__} {e} — 30초 뒤 다시")
                time.sleep(30)
        log(f"  가격 {min(i + batch, len(tickers))}/{len(tickers)} · 받은 종목 {len(out)}")
        time.sleep(1.5)
    return out


def cmd_fetch_prices(_args: argparse.Namespace) -> int:
    cands = json.loads((OUT / "candidates.json").read_text("utf-8"))
    tickers = sorted({c["px"] for c in cands.values() if c.get("px")})
    extra = ["SPY", "^TNX"] + [f"{c}USD=X" for c in FX_CCYS]
    path = OUT / "prices_monthly.pkl"
    px: dict[str, pd.DataFrame] = pickle.loads(path.read_bytes()) if path.exists() else {}
    todo = [t for t in tickers + extra if t not in px]
    log(f"가격: 종목 {len(tickers)} + 지수 · 환율 {len(extra)} · 받을 것 {len(todo)}")
    px.update(download_monthly(todo))
    missing = [t for t in todo if t not in px]
    if missing:
        log(f"한 번 더: 빠진 {len(missing)}")
        time.sleep(20)
        px.update(download_monthly(missing, batch=60))
    path.write_bytes(pickle.dumps(px))
    log(f"가격 저장: {len(px)} 종목 · 끝내 없음 {len([t for t in tickers + extra if t not in px])}")
    return 0


# ── companyfacts → 필요한 항목만 (build)

TAXES = ("us-gaap", "ifrs-full")
# 항목 → XBRL 태그 (앞이 우선). us-gaap 과 ifrs-full 이름을 같이 둔다 — 같은 이름이면 두 분류 모두에서 찾는다.
TAGS: dict[str, tuple[str, ...]] = {
    # 손익 (연간)
    "rev": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet", "SalesRevenueServicesNet",
            "RegulatedAndUnregulatedOperatingRevenue", "Revenue", "RevenueFromContractsWithCustomers",
            "RevenueFromSaleOfGoods", "RevenueFromRenderingOfServices", "RevenueFromRenderingOfTelecommunicationServices"),
    "ni_common": ("NetIncomeLossAvailableToCommonStockholdersBasic", "NetIncomeLoss", "ProfitLossAttributableToOwnersOfParent"),
    "ni": ("NetIncomeLoss", "ProfitLoss", "ProfitLossAttributableToOwnersOfParent"),
    "opinc": ("OperatingIncomeLoss", "ProfitLossFromOperatingActivities"),
    "pretax": ("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
               "ProfitLossBeforeTax"),
    "tax": ("IncomeTaxExpenseBenefit", "IncomeTaxExpenseContinuingOperations"),
    "intexp": ("InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense",
               "FinanceCosts", "InterestExpenseOnBorrowings"),
    "cogs": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices", "CostOfSales"),
    "sga": ("SellingGeneralAndAdministrativeExpense",),
    # 현금흐름 (연간)
    "ocf": ("NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            "CashFlowsFromUsedInOperatingActivities"),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets", "PaymentsForCapitalImprovements",
              "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
              "PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsOtherThanGoodwillInvestmentPropertyAndOtherNoncurrentAssets"),
    "sbc": ("ShareBasedCompensation", "AllocatedShareBasedCompensationExpense", "AdjustmentsForSharebasedPayments",
            "ExpenseFromSharebasedPaymentTransactionsWithEmployees",
            "ExpenseFromSharebasedPaymentTransactionsInWhichGoodsOrServicesReceivedDidNotQualifyForRecognitionAsAssets"),
    "da": ("DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet", "DepreciationAndAmortization",
           "Depreciation", "AdjustmentsForDepreciationAndAmortisationExpense", "DepreciationAndAmortisationExpense"),
    "div": ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends", "PaymentsOfOrdinaryDividends",
            "DividendsPaidClassifiedAsFinancingActivities", "DividendsPaid"),
    "rep": ("PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity", "PaymentsToAcquireOrRedeemEntitysShares"),
    "iss": ("ProceedsFromIssuanceOfCommonStock", "ProceedsFromStockOptionsExercised",
            "ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions",
            "ProceedsFromIssuingShares", "ProceedsFromExerciseOfOptions"),
    "debt_iss": ("ProceedsFromIssuanceOfLongTermDebt", "ProceedsFromIssuanceOfDebt",
                 "ProceedsFromBorrowingsClassifiedAsFinancingActivities", "ProceedsFromNoncurrentBorrowings"),
    "debt_rep": ("RepaymentsOfLongTermDebt", "RepaymentsOfDebt", "RepaymentsOfBorrowingsClassifiedAsFinancingActivities",
                 "RepaymentsOfNoncurrentBorrowings"),
    # 재무상태 (회계연도 끝)
    "eq": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
           "EquityAttributableToOwnersOfParent", "Equity"),
    "eq_nci": ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "Equity", "StockholdersEquity"),
    "assets": ("Assets",),
    "liab": ("Liabilities",),
    "ca": ("AssetsCurrent", "CurrentAssets"),
    "cl": ("LiabilitiesCurrent", "CurrentLiabilities"),
    "re": ("RetainedEarningsAccumulatedDeficit", "RetainedEarnings"),
    "ar": ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent", "TradeAndOtherCurrentReceivables", "CurrentTradeReceivables"),
    "ppe": ("PropertyPlantAndEquipmentNet",
            "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
            "PropertyPlantAndEquipment"),
    "cash_sti": ("CashCashEquivalentsAndShortTermInvestments",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue", "CashAndCashEquivalents", "Cash",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    "sti": ("ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesCurrent",
            "AvailableForSaleSecuritiesDebtSecuritiesCurrent"),
    "debt_total": ("DebtLongtermAndShorttermCombinedAmount", "Borrowings", "DebtAndCapitalLeaseObligations"),
    "ltd_all": ("LongTermDebt",),  # 1년 안 만기분 포함
    "ltd_noncur": ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongtermBorrowings",
                   "NoncurrentPortionOfNoncurrentBorrowings"),
    "ltd_cur": ("LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent", "CurrentPortionOfLongtermBorrowings"),
    "debt_cur": ("DebtCurrent", "CurrentBorrowingsAndCurrentPortionOfNoncurrentBorrowings"),
    "stb": ("ShortTermBorrowings", "ShorttermBorrowings"),
    "cp": ("CommercialPaper",),
    # 주식 수 (연간 가중평균, 단위 shares) — 과거 시가총액의 주식 수 비율에만 쓴다
    "shares": ("WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic",
               "AdjustedWeightedAverageShares", "WeightedAverageShares"),
}
FLOW_ROWS = frozenset({"rev", "ni_common", "ni", "opinc", "pretax", "tax", "intexp", "cogs", "sga", "ocf", "capex", "sbc", "da",
                       "div", "rep", "iss", "debt_iss", "debt_rep"})
DEBT_ROWS = ("debt_total", "ltd_all", "ltd_noncur", "ltd_cur", "debt_cur", "stb", "cp")


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days


def _annual(r: Mapping[str, Any]) -> bool:
    s, e = r.get("start"), r.get("end")
    return bool(s and e and 330 <= _days(s, e) <= 390)


def reporting_ccy(facts: Mapping[str, Any]) -> str | None:
    """재무제표 통화 = 연간(330~390일) 매출 · 영업현금흐름 값이 가장 많은 통화 단위 (SAP 의 USD 편의 환산 같은 건 적어서 밀린다)."""
    counts: dict[str, int] = {}
    for tax in TAXES:
        block = facts.get(tax) or {}
        for tag in TAGS["rev"] + TAGS["ocf"]:
            for unit, rows in ((block.get(tag) or {}).get("units") or {}).items():
                if len(unit) == 3 and unit.isalpha() and unit.isupper():
                    counts[unit] = counts.get(unit, 0) + sum(1 for r in rows if _annual(r))
    return max(counts, key=counts.get) if counts and max(counts.values()) > 0 else None


def extract(facts: Mapping[str, Any], row: str, ccy: str, kind: str) -> list[tuple[str, str, float, int]]:
    """한 항목의 값 후보 (회계연도 끝, 공시일, 값, 태그 순위). 보고서 종류는 가리지 않는다 — 캐나다 CN 레일처럼 재무제표
    XBRL 을 6-K 로 내는 회사가 있다. 기간 길이와 공시일로 고른다.
    kind: 'flow' = 330~390일 기간 값 · 'q' = 80~100일 분기 값 · 'inst' = 시점 값 · 'shares' = 330~390일 주식 수(단위 shares)."""
    out: list[tuple[str, str, float, int]] = []
    unit = "shares" if kind == "shares" else ccy
    for prio, tag in enumerate(TAGS[row]):
        for tax in TAXES:
            node = (facts.get(tax) or {}).get(tag)
            if not node:
                continue
            for r in (node.get("units") or {}).get(unit) or []:
                end, filed, val, start = r.get("end"), r.get("filed"), r.get("val"), r.get("start")
                if not (end and filed and val is not None) or end < "2007-01-01":
                    continue
                if kind == "q":
                    if not start or not 80 <= _days(start, end) <= 100:
                        continue
                elif kind in ("flow", "shares"):
                    if not _annual(r):
                        continue
                elif start:  # inst
                    continue
                out.append((end, filed, float(val), prio))
    return out


def extract_dei_shares(facts: Mapping[str, Any]) -> list[tuple[str, str, float, int]]:
    """표지 발행주식수 (dei:EntityCommonStockSharesOutstanding, 10-K · 10-Q · 20-F 표지, 기준일 = end). 주식 종류가 여럿인
    회사는 companyfacts 에 안 실린다(종류별 값은 차원 값이라)."""
    rows = (((facts.get("dei") or {}).get("EntityCommonStockSharesOutstanding") or {}).get("units") or {}).get("shares") or []
    return [(r["end"], r["filed"], float(r["val"]), 0) for r in rows
            if r.get("end") and r.get("filed") and r.get("val") and float(r["val"]) > 0]


def index_rows(cands: Iterable[tuple[str, str, float, int]]) -> dict[str, list[tuple[str, int, float, str]]]:
    """같은 달(회계연도 끝) → [(공시일, −태그 순위, 값, 끝 날짜)] 공시일 오름차순."""
    by: dict[str, list[tuple[str, int, float, str]]] = {}
    for end, filed, val, prio in cands:
        by.setdefault(end[:7], []).append((filed, -prio, val, end))
    for lst in by.values():
        lst.sort()
    return by


def parse_companyfacts(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    facts = raw.get("facts") or {}
    ccy = reporting_ccy(facts)
    if not ccy:
        return None
    rows: dict[str, dict[str, list[tuple[str, int, float, str]]]] = {}
    for row in TAGS:
        kind = "shares" if row == "shares" else ("flow" if row in FLOW_ROWS else "inst")
        got = extract(facts, row, ccy, kind)
        if got:
            rows[row] = index_rows(got)
    if "rev" not in rows:
        return None
    dei = extract_dei_shares(facts)
    if dei:
        rows["dei"] = index_rows(dei)
    q = index_rows(extract(facts, "rev", ccy, "q"))
    return {"name": raw.get("entityName"), "ccy": ccy, "rows": rows, "q": q}


def merge_facts(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """같은 회사의 두 CIK(지주회사 재편 — ExxonMobil Holdings 2115436 ← Exxon Mobil 34088 등)를 합친다.
    같은 달 값은 공시일 순으로 섞이고, 그날까지의 최근 공시가 이긴다. 통화는 지금 CIK 를 따른다."""
    if not a or not b:
        return a or b
    if a["ccy"] != b["ccy"]:
        return a if len(a["rows"].get("rev", {})) >= len(b["rows"].get("rev", {})) else b
    rows: dict[str, dict[str, list[Any]]] = {}
    for row in set(a["rows"]) | set(b["rows"]):
        ra, rb = a["rows"].get(row, {}), b["rows"].get(row, {})
        rows[row] = {ym: sorted(list(ra.get(ym, [])) + list(rb.get(ym, []))) for ym in set(ra) | set(rb)}
    q = {ym: sorted(list(a["q"].get(ym, [])) + list(b["q"].get(ym, []))) for ym in set(a["q"]) | set(b["q"])}
    return {"name": a.get("name"), "ccy": a["ccy"], "rows": rows, "q": q}


def _parse_file(path: str) -> tuple[int, dict[str, Any] | None]:
    cik = int(Path(path).name[3:13])
    try:
        return cik, parse_companyfacts(json.loads(gzip.decompress(Path(path).read_bytes())))
    except Exception:
        return cik, None


def cmd_build(_args: argparse.Namespace) -> int:
    from concurrent.futures import ProcessPoolExecutor
    files = sorted(str(p) for p in CF_DIR.glob("CIK*.json.gz"))
    log(f"companyfacts {len(files)}개 정리")
    out: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for i, (cik, rec) in enumerate(ex.map(_parse_file, files, chunksize=20), 1):
            if rec:
                out[cik] = rec
            if i % 1000 == 0:
                log(f"  {i}/{len(files)}")
    (OUT / "facts.pkl").write_bytes(pickle.dumps(out))
    ccys: dict[str, int] = {}
    for rec in out.values():
        ccys[rec["ccy"]] = ccys.get(rec["ccy"], 0) + 1
    log(f"정리 끝: {len(out)}개 회사 · 통화 {sorted(ccys.items(), key=lambda kv: -kv[1])[:12]}")
    return 0


# ── 그날 시점의 재무 (value_model.series 가 읽는 모양)

MAX_YEARS = 4          # 운영 중인 v1 · v2 가 yfinance 에서 받는 연간 재무 수 (4~5년)
STALE_MONTHS = 18      # 가장 최근 회계연도 끝이 그날보다 이만큼 넘게 오래되면 뺀다 (Gemini 와 같음)


def pick(by_ym: Mapping[str, Sequence[tuple[str, int, float, str]]] | None, end: str, asof: str) -> float | None:
    """그날(asof)까지 공시된 값 중 그 회계연도 끝(같은 달)의 가장 최근 공시 값. 같은 날 공시면 앞 태그."""
    if not by_ym:
        return None
    best = None
    for item in by_ym.get(end[:7], ()):
        if item[0] > asof:
            break
        if best is None or item[:2] >= best[:2]:
            best = item
    return None if best is None else best[2]


def select_ends(rev: Mapping[str, Sequence[tuple[str, int, float, str]]], asof: str,
                max_years: int = MAX_YEARS) -> list[str]:
    """그날까지 연간 매출이 공시된 회계연도 끝 중 최근 max_years 개 (연속된 해만, 최근 해가 18개월 안)."""
    ends: list[str] = []
    for lst in rev.values():
        ok = [x for x in lst if x[0] <= asof]
        if ok:
            ends.append(max(ok)[3])
    ends = sorted(e for e in ends if e <= asof)
    if not ends or _days(ends[-1], asof) > STALE_MONTHS * 30.5:
        return []
    run = [ends[-1]]
    for e in reversed(ends[:-1]):
        if not 300 <= _days(e, run[0]) <= 430:
            break
        run.insert(0, e)
        if len(run) >= max_years:
            break
    return run if len(run) >= MIN_YEARS_V else []


def total_debt(rows: Mapping[str, Any], end: str, asof: str) -> float | None:
    """총차입금 (리스 부채는 빼고, 태그 조합 순서대로). 차입금 태그가 하나도 없으면 None."""
    v = {r: pick(rows.get(r), end, asof) for r in DEBT_ROWS}
    if v["debt_total"] is not None:
        return v["debt_total"]
    short = (v["stb"] or 0.0) + (v["cp"] or 0.0)
    if v["ltd_all"] is not None:
        return v["ltd_all"] + short
    if any(v[r] is not None for r in ("ltd_noncur", "ltd_cur", "debt_cur", "stb", "cp")):
        cur = v["debt_cur"] if v["debt_cur"] is not None else (v["ltd_cur"] or 0.0) + short
        return (v["ltd_noncur"] or 0.0) + cur
    return None


def build_ent(co: Mapping[str, Any], asof: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """그날 시점의 재무 → value_model.analyze 가 읽는 ent. 두 번째 값은 표시(빠진 항목) 모음."""
    rows = co["rows"]
    ends = select_ends(rows["rev"], asof)
    flags: dict[str, Any] = {}
    if not ends:
        return None, flags

    def val(row: str, e: str) -> float | None:
        x = pick(rows.get(row), e, asof)
        return x if x is not None and math.isfinite(x) else None

    def put(block: dict[str, Any], name: str, fn: Any) -> None:
        d = {}
        for e in ends:
            x = fn(e)
            if x is not None and math.isfinite(x):
                d[e] = float(x)
        if d:
            block[name] = d

    inc: dict[str, Any] = {}
    bal: dict[str, Any] = {}
    cf: dict[str, Any] = {}
    put(inc, "Total Revenue", lambda e: val("rev", e))
    put(inc, "Net Income Common Stockholders", lambda e: val("ni_common", e))
    put(inc, "Net Income", lambda e: val("ni", e))

    def ebit(e: str) -> float | None:
        op = val("opinc", e)
        if op is not None:
            return op
        pt, ie = val("pretax", e), val("intexp", e)
        return None if pt is None else pt + (abs(ie) if ie is not None else 0.0)

    put(inc, "EBIT", ebit)
    put(inc, "Pretax Income", lambda e: val("pretax", e))
    put(inc, "Tax Provision", lambda e: val("tax", e))
    put(inc, "Interest Expense", lambda e: abs(x) if (x := val("intexp", e)) is not None else None)
    put(inc, "Cost Of Revenue", lambda e: val("cogs", e))
    put(inc, "Selling General And Administration", lambda e: val("sga", e))

    put(bal, "Stockholders Equity", lambda e: val("eq", e))
    debt = {e: total_debt(rows, e, asof) for e in ends}
    flags["debt_none"] = all(x is None for x in debt.values())
    bal["Total Debt"] = {e: float(x if x is not None else 0.0) for e, x in debt.items()}

    def cash(e: str) -> float | None:
        c = val("cash_sti", e)
        if c is not None:
            return c
        c = val("cash", e)
        return None if c is None else c + (val("sti", e) or 0.0)

    bal["Cash Cash Equivalents And Short Term Investments"] = {e: float(cash(e) or 0.0) for e in ends}
    put(bal, "Total Assets", lambda e: val("assets", e))

    def liab(e: str) -> float | None:
        x = val("liab", e)
        if x is not None:
            return x
        a, q = val("assets", e), val("eq_nci", e)
        return None if a is None or q is None else a - q

    put(bal, "Total Liabilities Net Minority Interest", liab)
    put(bal, "Current Assets", lambda e: val("ca", e))
    put(bal, "Current Liabilities", lambda e: val("cl", e))
    put(bal, "Retained Earnings", lambda e: val("re", e))
    put(bal, "Accounts Receivable", lambda e: val("ar", e))
    put(bal, "Net PPE", lambda e: val("ppe", e))
    put(bal, "Long Term Debt And Capital Lease Obligation", lambda e: val("ltd_noncur", e))

    put(cf, "Operating Cash Flow", lambda e: val("ocf", e))
    capex = {e: val("capex", e) for e in ends}
    flags["capex_missing"] = all(x is None for x in capex.values())
    put(cf, "Capital Expenditure", lambda e: -abs(capex[e]) if capex[e] is not None else (0.0 if val("ocf", e) is not None else None))
    put(cf, "Free Cash Flow",
        lambda e: (o + (-abs(capex[e]) if capex[e] is not None else 0.0)) if (o := val("ocf", e)) is not None else None)
    put(cf, "Stock Based Compensation", lambda e: abs(x) if (x := val("sbc", e)) is not None else None)
    put(cf, "Depreciation And Amortization", lambda e: abs(x) if (x := val("da", e)) is not None else None)
    put(cf, "Cash Dividends Paid", lambda e: -abs(x) if (x := val("div", e)) is not None else None)
    put(cf, "Repurchase Of Capital Stock", lambda e: -abs(x) if (x := val("rep", e)) is not None else None)
    put(cf, "Issuance Of Capital Stock", lambda e: abs(x) if (x := val("iss", e)) is not None else None)

    def dnet(e: str) -> float | None:
        di, dr = val("debt_iss", e), val("debt_rep", e)
        if di is not None or dr is not None:
            return abs(di or 0.0) - abs(dr or 0.0)
        i = ends.index(e)
        if i > 0 and debt[e] is not None and debt[ends[i - 1]] is not None:
            return debt[e] - debt[ends[i - 1]]
        return None

    put(cf, "Net Issuance Payments Of Debt", dnet)
    return {"info": {}, "inc": inc, "bal": bal, "cf": cf}, flags


def quarters_asof(co: Mapping[str, Any], asof: str) -> list[dict[str, Any]]:
    """그날까지 공시된 분기 매출 [{'end', 'revenue'}] (회계연도 끝 오름차순) — value_model.revenue_falling 모양."""
    out = []
    for lst in co.get("q", {}).values():
        ok = [x for x in lst if x[0] <= asof]
        if ok:
            best = max(ok)
            out.append({"end": best[3], "revenue": best[2]})
    return sorted(out, key=lambda q: q["end"])


# ── 그날 가격 · 시가총액 · 베타

def month_end(d: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(d).to_period("M").to_timestamp("M")


def px_at(df: pd.DataFrame | None, d: pd.Timestamp, col: str = "close") -> float:
    if df is None:
        return NAN_F
    try:
        v = float(df.at[d, col])
    except KeyError:
        return NAN_F
    return v if math.isfinite(v) and v > 0 else NAN_F


def split_factor(df: pd.DataFrame, after: str, upto: str) -> float:
    """(after, upto] 사이 분할 배수 곱 (yfinance 월봉의 분할 칸, 월 단위)."""
    s = df["split"]
    s = s[(s.index > month_end(after)) & (s.index <= month_end(upto)) & (s > 0)]
    return float(np.prod(s.values)) if len(s) else 1.0


def latest_value(by_ym: Mapping[str, Sequence[tuple[str, int, float, str]]], asof: str) -> tuple[str, int, float, str] | None:
    """그날까지 공시된 값 중 기준일(end)이 가장 늦은 것 (같으면 나중 공시)."""
    best = None
    for lst in by_ym.values():
        for item in lst:
            if item[0] > asof:
                break
            if item[2] > 0 and (best is None or (item[3], item[0]) > (best[3], best[0])):
                best = item
    return best


def share_ratio(co: Mapping[str, Any], px: pd.DataFrame, asof: str, now: str) -> tuple[float, str]:
    """그날 발행주식수 ÷ 지금 발행주식수 (지금 분할 기준). 표지 주식수(dei, 10-Q · 10-K 표지라 더 최근)를 먼저, 없으면
    연간 가중평균 희석주식수. 분할은 SEC 숫자에서도 실제로 튄 경우만 반영한다: (공시값 비율 R) 과 (R × 그 사이 yfinance
    분할 배수) 중 1 에 가까운 쪽 — ADR 비율 변경처럼 원주 수가 안 변한 '분할'을 걸러낸다. 주식 수가 없거나 결과가 1/3 ~ 3
    밖이면 1 (= 오늘 주식 수, Gemini 방식) 로 두고 표시한다."""
    for src in ("dei", "shares"):
        sh = co["rows"].get(src)
        if not sh:
            continue
        then, cur = latest_value(sh, asof), latest_value(sh, now)
        if not then or not cur or _days(then[3], asof) > 460:
            continue
        r = then[2] / cur[2]
        f = split_factor(px, then[0], cur[0])
        best = min((r, r * f), key=lambda x: abs(math.log(x)))
        if not 1 / 3 <= best <= 3:
            return 1.0, "share_jump"
        return best, src + ("_split" if best != r else "")
    return 1.0, "no_shares"


def monthly_beta_tr(stock: pd.DataFrame, spy: pd.DataFrame, d: pd.Timestamp) -> float:
    """그날까지 60개월 배당 포함 월간 수익률 회귀 베타 (36개월 미만이면 1.0, 0.3~2.5 로 자름) — value_model.monthly_beta 와 같은 규칙."""
    a = stock["adj"][stock.index <= d].pct_change()
    b = spy["adj"][spy.index <= d].pct_change()
    m = pd.concat([a.rename("s"), b.rename("b")], axis=1).dropna().tail(60)
    if len(m) < 36:
        return 1.0
    var = float(np.var(m["b"], ddof=1))
    beta = float(np.cov(m["s"], m["b"])[0, 1] / var) if var > 0 else float("nan")
    return min(max(beta, BETA_RANGE[0]), BETA_RANGE[1]) if math.isfinite(beta) else 1.0


# ── 규칙 — v1 · v2 = value_model.analyze (지금 운영 규칙), v3 = Gemini run_10y_backtest.evaluate_point_in_time 그대로

V3_CAP = 3      # 업종 상한 (Gemini 10년 백테스트 · 문서 4.5 표와 같음)


def cagr_series(s: pd.Series) -> float:
    """Gemini run_10y_backtest.cagr_series 그대로."""
    s = s.dropna()
    n = len(s)
    if n >= 2 and _at(s, 0) > 0 and _at(s, -1) > 0:
        return float((_at(s, -1) / _at(s, 0)) ** (1.0 / (n - 1)) - 1.0)
    if n >= 2 and _at(s, 0) > 0 and _at(s, -1) <= 0:
        return -0.50
    return float("nan")


def v3_metrics(res: Mapping[str, Any], ent: Mapping[str, Any], mcap_fc: float, use_ev: bool) -> dict[str, Any]:
    """v3 (문서 4.6 = Gemini 코드). use_ev=False 면 분모만 시가총액 (변형 B — 식 오류 제거)."""
    rev = series(ent, "inc", "Total Revenue", "Operating Revenue").dropna()
    ni = series(ent, "inc", "Net Income Common Stockholders", "Net Income").dropna()
    ebit = series(ent, "inc", "EBIT", "Operating Income").dropna()
    eq = series(ent, "bal", "Stockholders Equity", "Common Stock Equity").dropna()
    debt = series(ent, "bal", "Total Debt").fillna(0)
    cash = series(ent, "bal", "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents").fillna(0)
    fcf = series(ent, "cf", "Free Cash Flow").dropna()
    sbc = series(ent, "cf", "Stock Based Compensation").reindex(fcf.index).fillna(0)

    fcf_sub_sbc = (fcf - sbc).dropna()
    real_ok = bool(len(fcf_sub_sbc) >= MIN_YEARS_V and (fcf_sub_sbc.tail(3) > 0).all())
    fcf_use = fcf_sub_sbc if real_ok else fcf
    neg_eq = bool((eq <= 0).any())
    sh_y = res["sh_y"]
    if neg_eq:
        g_roe = bool(len(ni) >= MIN_YEARS_V and (ni > 0).all() and math.isfinite(sh_y) and sh_y > 0)
    else:
        g_roe = bool(res["g_roe"])
    quality = bool(g_roe and res["g_roic"] and res["g_z"] and res["g_m"] and res["g_fcf"] and real_ok)
    out: dict[str, Any] = {"quality_v3": quality, "pass_v3": False, "kill_v3": ["quality_v3"]}
    if not quality:
        return out
    d_last = max(_at(debt, -1), 0.0) if math.isfinite(_at(debt, -1)) else 0.0
    c_last = max(_at(cash, -1), 0.0) if math.isfinite(_at(cash, -1)) else 0.0
    denom = max(mcap_fc + d_last - c_last, mcap_fc * 0.25) if use_ev else mcap_fc
    fcf_mean = float(fcf_use.tail(3).mean())
    fcf_y = fcf_mean / denom if denom > 0 else float("nan")
    g_c = [g for g in (cagr_series(rev), cagr_series(ebit), cagr_series(fcf_use)) if math.isfinite(g)]
    g1 = min(max(min(g_c) if g_c else 0.0, G1_RANGE[0]), G1_RANGE[1])
    ke = res["ke"]
    alpha = irr_two_stage(fcf_y, g1, TERM_G) - ke
    alpha_bear = irr_two_stage(fcf_y * (1.0 - BEAR_FCF_CUT), min(g1, 0.0), TERM_G) - ke
    g_price = bool(math.isfinite(alpha) and alpha >= ALPHA_MIN)
    g_bear = bool(math.isfinite(alpha_bear) and alpha_bear >= 0)
    g_sh = bool(not math.isfinite(sh_y) or sh_y >= 0.0)
    rev_g = cagr_series(rev)
    rev_down = bool(len(rev) >= 2 and _at(rev, -1) < _at(rev, -2) and (rev_g < 0 or _at(rev, -1) < _at(rev, -2) * 0.90))
    fcf_ok = bool(_at(fcf_use, -1) >= fcf_mean * 0.80)
    g_rev = bool(not rev_down and fcf_ok)
    kill = []
    if not g_rev:
        kill.append("rev_or_fcf_down")
    if not g_sh:
        kill.append("sh_dilution")
    if math.isfinite(alpha) and alpha < 0:
        kill.append("alpha_neg")
    out.update({"fcf_y_v3": fcf_y, "g1_v3": g1, "alpha_v3": alpha, "alpha_bear_v3": alpha_bear, "g_sh_v3": g_sh,
                "g_rev_v3": g_rev, "pass_v3": bool(g_price and g_bear and g_sh and g_rev), "kill_v3": kill})
    return out


def evaluate(ent: Mapping[str, Any], quarters: list[dict[str, Any]], *, price: float, mcap_fc: float, beta: float,
             rf: float) -> dict[str, Any] | None:
    """한 회사 · 한 날짜. v1 품질을 통과한 회사만 돌려준다 (Gemini 와 같음 — v2 · v3 · 플라시보 후보도 이 안에서)."""
    fcf_d = (ent.get("cf") or {}).get("Free Cash Flow") or {}
    rev_d = (ent.get("inc") or {}).get("Total Revenue") or {}
    if len(fcf_d) < MIN_YEARS_V or any(v <= 0 for v in fcf_d.values()):
        return None
    if len(rev_d) < MIN_YEARS_V or any(v <= 0 for v in rev_d.values()):
        return None
    res = analyze(ent, price=price, mcap_fc=mcap_fc, beta=beta, rf=rf)
    if not res["quality"]:
        return None
    res["rev_down"] = revenue_falling(quarters, series(ent, "inc", "Total Revenue", "Operating Revenue"))
    res["kill"] = kill_reasons(res)
    for tag, use_ev in (("A", True), ("B", False)):
        for k, v in v3_metrics(res, ent, mcap_fc, use_ev).items():
            res[k.replace("_v3", f"_v3{tag}")] = v
    return res


def select_portfolios(rows: Sequence[Mapping[str, Any]], cap: int = V3_CAP) -> dict[str, list[Mapping[str, Any]]]:
    """Pass · Top 10 묶음. top_v1 · top_v2 = value_model 과 같은 정렬, top_v3 = Gemini select_portfolios (업종 상한 cap)."""
    fin = lambda x: x is not None and math.isfinite(x)  # noqa: E731
    out: dict[str, list[Mapping[str, Any]]] = {
        "pass_v1": [r for r in rows if r["pass"]],
        "pass_v2": [r for r in rows if r["pass_v2"]],
        "top_v1": sorted([r for r in rows if fin(r["drop"])], key=lambda r: -r["drop"])[:TOP_N],
        "top_v2": sorted([r for r in rows if not r["kill"] and fin(r["gap"])], key=lambda r: -r["gap"])[:TOP_N],
    }
    for v in ("A", "B"):
        out[f"pass_v3{v}"] = [r for r in rows if r.get(f"pass_v3{v}")]
        q = [r for r in rows if r.get(f"quality_v3{v}") and r.get(f"g_rev_v3{v}") and r.get(f"g_sh_v3{v}")
             and fin(r.get(f"alpha_v3{v}")) and fin(r.get(f"alpha_bear_v3{v}"))]
        q.sort(key=lambda r: (0 if r[f"pass_v3{v}"] else (1 if not r[f"kill_v3{v}"] else 2),
                              -(0.5 * r[f"alpha_v3{v}"] + 0.5 * r[f"alpha_bear_v3{v}"])))
        picked: list[Mapping[str, Any]] = []
        counts: dict[str, int] = {}
        for r in q:
            ind = r.get("industry") or "—"
            if counts.get(ind, 0) >= cap:
                continue
            picked.append(r)
            counts[ind] = counts.get(ind, 0) + 1
            if len(picked) >= TOP_N:
                break
        out[f"top_v3{v}"] = picked
    return out


STRATS = ("pass_v1", "pass_v2", "pass_v3A", "pass_v3B", "top_v1", "top_v2", "top_v3A", "top_v3B")
PRIMARY = ("pass_v3A", "pass_v3B", "top_v3A", "top_v3B")
V2_OF = {"pass_v3A": "pass_v2", "pass_v3B": "pass_v2", "top_v3A": "top_v2", "top_v3B": "top_v2"}


# ── 수익률 (월간 자본곡선)

COST = 0.001       # 편도 거래비용 (매매 금액의 0.1%)


def period_paths(px: Mapping[str, pd.DataFrame], tickers: Iterable[str], d0: pd.Timestamp,
                 months: Sequence[pd.Timestamp]) -> dict[str, np.ndarray]:
    """종목별 d0 대비 누적 배당 포함 가격 비율 (월말마다). 중간에 가격이 끊기면 마지막 값으로 둔다(현금화, 셈)."""
    out: dict[str, np.ndarray] = {}
    for t in tickers:
        df = px.get(t)
        p0 = px_at(df, d0, "adj")
        if not math.isfinite(p0):
            continue
        vals, last = [], 1.0
        for m in months:
            p = px_at(df, m, "adj")
            if math.isfinite(p):
                last = p / p0
            vals.append(last)
        out[t] = np.array(vals)
    return out


def run_series(baskets: Sequence[Sequence[str]], paths: Sequence[Mapping[str, np.ndarray]],
               cost: float = COST) -> tuple[list[float], list[float]]:
    """해마다 같은 비중으로 사서 1년 보유(비중은 흘러감), 리밸런싱 때 매매 금액 × cost. → (월간 수익률, 기간 수익률)."""
    monthly: list[float] = []
    period: list[float] = []
    prev_w: dict[str, float] = {}
    for names, pth in zip(baskets, paths):
        names = [t for t in names if t in pth]
        if not names:
            raise ValueError("빈 묶음 — SPY 로 채운 뒤 부를 것")
        w0 = 1.0 / len(names)
        new_w = {t: w0 for t in names}
        turnover = sum(abs(new_w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in set(new_w) | set(prev_w))
        g = np.mean([pth[t] for t in names], axis=0)           # 포트폴리오 가치 (시작 1)
        rets = np.diff(np.concatenate([[1.0], g])) / np.concatenate([[1.0], g[:-1]])
        rets[0] = (1 + rets[0]) * (1 - cost * turnover) - 1
        monthly.extend(rets.tolist())
        period.append(float(g[-1] * (1 - cost * turnover) - 1))
        last = {t: pth[t][-1] for t in names}
        tot = sum(last.values())
        prev_w = {t: v / tot for t, v in last.items()} if tot > 0 else {}
    return monthly, period


def stats_block(r: Sequence[float], spy: Sequence[float]) -> dict[str, float]:
    ab = alpha_beta(list(r), {"SPY": list(spy)}, periods=12)
    return {"cagr": cagr(list(r), 12), "sharpe": sharpe(list(r), 12), "mdd": max_drawdown(list(r)),
            "alpha": ab.get("알파_연", float("nan")), "alpha_t": ab.get("알파_t", float("nan")),
            "beta": ab.get("베타_SPY", float("nan"))}


def boot(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """paired_bootstrap (6개월 블록, 2,000번). evaluation.sharpe 가 일간(252) 연율화라 ΔSharpe 를 월간(12) 기준으로 되돌린다."""
    st = paired_bootstrap(list(a), list(b), block=6, n_boot=2000, seed=42, n_folds=3)
    if not st:
        return {}
    k = math.sqrt(12 / 252)
    out = dict(st)
    for key in ("ΔSharpe", "ΔSharpe_하한", "ΔSharpe_상한"):
        out[key] = st[key] * k
    out["판정"] = verdict(st)
    return out


# ── 실행

REBAL = [f"{y}-05-31" for y in range(2014, 2026)] + ["2026-10-31"]
N_PLACEBO = 1000


def load_inputs() -> dict[str, Any]:
    facts = pickle.loads((OUT / "facts.pkl").read_bytes())
    px = pickle.loads((OUT / "prices_monthly.pkl").read_bytes())
    cands = json.loads((OUT / "candidates.json").read_text("utf-8"))
    scan = load_scan()
    sm: dict[str, tuple[str | None, str | None]] = {}
    if SECTOR_MAP_CSV.exists():
        df = pd.read_csv(SECTOR_MAP_CSV)
        cols = {c.lower(): c for c in df.columns}
        tcol = cols.get("ticker") or cols.get("티커")
        gics = {"Financials": "Financial Services"}  # sector_map 은 GICS 이름 — excluded() 는 yfinance 이름을 본다
        for _, r in df.iterrows():
            sec = r.get(cols.get("sector", ""), None)
            sm[norm_ticker(r[tcol])] = (gics.get(sec, sec), r.get(cols.get("industry", ""), None))
    return {"facts": facts, "px": px, "cands": cands, "scan": scan, "sector_map": sm, "mem": load_membership()}


def run_backtest(inp: Mapping[str, Any]) -> dict[str, Any]:
    facts, px, cands, scan, sm, mem = (inp[k] for k in ("facts", "px", "cands", "scan", "sector_map", "mem"))
    spy, tnx = px["SPY"], px["^TNX"]
    now = date.today().isoformat()
    # CIK 마다 대표 티커 하나 — 전체분석 목록(지금 상장) 중 시가총액 큰 쪽
    by_cik: dict[int, dict[str, Any]] = {}
    for t, c in cands.items():
        if not c.get("cik") or not c.get("px") or c["px"] not in px:
            continue
        if c["cik"] not in facts and (c.get("alt") or 0) not in facts:
            continue
        cur = by_cik.get(c["cik"])
        mc = (scan.get(c["px"]) or {}).get("mcap_now") or 0.0
        if cur is None or (c["scan"] and mc > cur["mcap_now"]):
            by_cik[c["cik"]] = {"px": c["px"], "mcap_now": mc, "scan": c["scan"], "alt": c.get("alt") or (cur or {}).get("alt")}
    cos = {cik: merge_facts(facts.get(cik), facts.get(m["alt"]) if m.get("alt") else None) for cik, m in by_cik.items()}
    tick2cik = {t: c["cik"] for t, c in cands.items() if c.get("cik")}
    log(f"가격 · SEC 둘 다 있는 회사 {len(by_cik)}")

    years: list[dict[str, Any]] = []
    flags_total: dict[str, int] = {}
    for d0s, d1s in zip(REBAL[:-1], REBAL[1:]):
        d0, d1 = month_end(d0s), month_end(d1s)
        months = [m for m in spy.index if d0 < m <= d1]
        rf = px_at(tnx, d0)
        rf = min(max(rf / 100.0, 0.015), 0.06) if math.isfinite(rf) else 0.04
        members = members_on(mem, d0s)
        a_ciks = {tick2cik[t] for t in members if t in tick2cik}
        cov = {"members": len(members), "members_cik": len(a_ciks)}
        rows_by_cik: dict[int, dict[str, Any]] = {}
        b_ciks: set[int] = set()
        a_eval: set[int] = set()
        for cik, meta in by_cik.items():
            in_a = cik in a_ciks
            if not (in_a or meta["scan"]):
                continue
            t = meta["px"]
            df = px[t]
            p0 = px_at(df, d0)
            if not math.isfinite(p0):
                continue
            sec = (scan.get(t) or {}).get("sector") or sm.get(t, (None, None))[0]
            ind = (scan.get(t) or {}).get("industry") or sm.get(t, (None, None))[1]
            if not sec or excluded({"sector": sec, "industry": ind}):
                continue
            co = cos[cik]
            mcap_now = meta["mcap_now"]
            p_now = float(df["close"].iloc[-1])
            if not (mcap_now and math.isfinite(p_now) and p_now > 0):
                continue
            ratio, how = share_ratio(co, df, d0s, now)
            flags_total[how] = flags_total.get(how, 0) + 1
            mcap_usd = mcap_now * (p0 / p_now) * ratio
            in_b = meta["scan"] and mcap_usd >= 1e9
            if not (in_a or in_b):
                continue
            ccy = co["ccy"]
            fx = 1.0 if ccy == "USD" else px_at(px.get(f"{ccy}USD=X"), d0)
            if not math.isfinite(fx):
                flags_total["fx_missing"] = flags_total.get("fx_missing", 0) + 1
                continue
            ent, fl = build_ent(co, d0s)
            if ent is None:
                continue
            for k, v in fl.items():
                if v:
                    flags_total[k] = flags_total.get(k, 0) + 1
            if in_a:
                a_eval.add(cik)
            beta = monthly_beta_tr(df, spy, d0)
            res = evaluate(ent, quarters_asof(co, d0s), price=p0, mcap_fc=mcap_usd / fx, beta=beta, rf=rf)
            if in_b:
                b_ciks.add(cik)
            if res is None:
                continue
            res.update({"ticker": t, "cik": cik, "industry": ind, "sector": sec, "mcap_usd": mcap_usd, "share_how": how})
            rows_by_cik[cik] = res
        uni = {"A": a_ciks, "B": b_ciks}
        cov.update({"A_with_data": len(a_eval), "B_companies": len(b_ciks)})
        ports: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
        pools: dict[str, list[str]] = {}
        for u, ciks in uni.items():
            rows = sorted((r for c, r in rows_by_cik.items() if c in ciks), key=lambda r: r["ticker"])
            ports[u] = select_portfolios(rows)
            pools[u] = [r["ticker"] for r in rows]
            cov[f"{u}_quality"] = len(rows)
        all_t = {r["ticker"] for r in rows_by_cik.values()} | {"SPY"}
        paths = period_paths(px, all_t, d0, months)
        years.append({"d0": d0s, "d1": d1s, "rf": rf, "cov": cov, "ports": ports, "pools": pools, "paths": paths,
                      "months": months})
        log(f"{d0s}: A 구성 {cov['members']} → 자료 {cov['A_with_data']} · 품질 {cov['A_quality']} | "
            f"B ≥$1B {cov['B_companies']} · 품질 {cov['B_quality']} | v3A 통과 A {len(ports['A']['pass_v3A'])} "
            f"B {len(ports['B']['pass_v3A'])}")
    return {"years": years, "flags": flags_total}


def summarize(bt: Mapping[str, Any], seed: int = 7) -> dict[str, Any]:
    years = bt["years"]
    paths = [y["paths"] for y in years]
    spy_m, spy_p = run_series([["SPY"]] * len(years), paths, cost=0.0)
    months = [m for y in years for m in y["months"]]
    windows = {"full": (0, len(years)), "from2016": (2, len(years)), "to2025": (0, len(years) - 1)}
    out: dict[str, Any] = {"months": [str(m.date()) for m in months], "spy": {"monthly": spy_m, "period": spy_p}, "U": {}}
    rng = random.Random(seed)
    for u in ("A", "B"):
        res_u: dict[str, Any] = {}
        series_m: dict[str, list[float]] = {}
        for s in STRATS:
            baskets = []
            picks: list[tuple[str, str, float]] = []
            spy_years = 0
            for y in years:
                names = [r["ticker"] for r in y["ports"][u][s] if r["ticker"] in y["paths"]]
                if not names:
                    names, spy_years = ["SPY"], spy_years + 1
                else:
                    for t in names:
                        picks.append((y["d0"], t, float(y["paths"][t][-1] - 1)))
                baskets.append(names)
            m, p = run_series(baskets, paths)
            series_m[s] = m
            blk: dict[str, Any] = {"baskets": baskets, "period": p, "spy_years": spy_years,
                                   "beat_years": sum(1 for a, b in zip(p, spy_p) if a > b),
                                   "picks_n": len(picks),
                                   "win": float(np.mean([x[2] > 0 for x in picks])) if picks else float("nan"),
                                   "crash": float(np.mean([x[2] <= -0.25 for x in picks])) if picks else float("nan"),
                                   "picks": picks}
            for w, (i0, i1) in windows.items():
                mi0 = sum(len(y["months"]) for y in years[:i0])
                mi1 = sum(len(y["months"]) for y in years[:i1])
                blk[w] = stats_block(m[mi0:mi1], spy_m[mi0:mi1])
                blk[w]["vs_spy"] = boot(m[mi0:mi1], spy_m[mi0:mi1])
            res_u[s] = blk
        for s in STRATS:  # v2 대비 · 플라시보
            blk = res_u[s]
            if s in V2_OF:
                blk["full"]["vs_v2"] = boot(series_m[s], series_m[V2_OF[s]])
            sims = []
            for _ in range(N_PLACEBO):
                bk = []
                for y, real in zip(years, blk["baskets"]):
                    pool = [t for t in y["pools"][u] if t in y["paths"]]
                    n = 0 if real == ["SPY"] else len(real)
                    bk.append(rng.sample(pool, min(n, len(pool))) if n and pool else ["SPY"])
                sims.append(sharpe(run_series(bk, paths)[0], 12))
            sims.sort()
            act = blk["full"]["sharpe"]
            blk["placebo"] = {"pct": 100.0 * sum(1 for x in sims if x < act) / len(sims),
                              "p50": sims[len(sims) // 2], "p95": sims[int(0.95 * len(sims)) - 1]}
        trial_sr = [sharpe(series_m[s], 1) for s in STRATS]
        for s in STRATS:
            r = series_m[s]
            sk = float(pd.Series(r).skew())
            ku = float(pd.Series(r).kurt()) + 3.0
            res_u[s]["dsr"] = deflated_sharpe(sharpe(r, 1), trial_sr, len(r), sk, ku)
        out["U"][u] = res_u
    spy_full = stats_block(spy_m, spy_m)
    out["spy"].update({"full": spy_full,
                       "from2016": stats_block(spy_m[sum(len(y["months"]) for y in years[:2]):],
                                               spy_m[sum(len(y["months"]) for y in years[:2]):]),
                       "to2025": stats_block(spy_m[:-len(years[-1]["months"])], spy_m[:-len(years[-1]["months"])])})
    # 판정 (사전 등록): 두 유니버스 모두 vs SPY 하한 > 0 + 플라시보 95 백분위 이상
    verdicts = {}
    for s in PRIMARY:
        ok = []
        for u in ("A", "B"):
            b = out["U"][u][s]
            vs = b["full"]["vs_spy"]
            ok.append(bool(vs and vs["ΔSharpe_하한"] > 0 and b["placebo"]["pct"] >= 95.0))
        rej = all((out["U"][u][s]["full"]["vs_spy"] or {}).get("ΔSharpe_상한", 1.0) < 0 for u in ("A", "B"))
        verdicts[s] = "후보" if all(ok) else ("기각" if rej else "운과 구분 안 됨")
    out["verdicts"] = verdicts
    out["coverage"] = [{"d0": y["d0"], **y["cov"]} for y in years]
    out["flags"] = bt["flags"]
    return out


def pct(x: float, d: int = 1) -> str:
    return "—" if x is None or not math.isfinite(x) else f"{x * 100:+.{d}f}%"


def report_md(res: Mapping[str, Any]) -> str:
    """사람이 읽는 결과 (문서 4.10절에 옮겨 쓴다)."""
    names = {"pass_v1": "v1 통과", "pass_v2": "v2 통과", "pass_v3A": "v3 통과 (A: EV)", "pass_v3B": "v3 통과 (B: 시총)",
             "top_v1": "v1 Top 10", "top_v2": "v2 Top 10", "top_v3A": "v3 Top 10 (A: EV)", "top_v3B": "v3 Top 10 (B: 시총)"}
    L = []
    sp = res["spy"]["full"]
    L.append(f"SPY(배당 포함) 2014-06~2026-10: CAGR {pct(sp['cagr'])} · 샤프 {sp['sharpe']:.2f} · MDD {pct(sp['mdd'])}\n")
    for u, title in (("A", "방식 A — 그날 S&P 500 구성종목"), ("B", "방식 B′ — 그날 SEC 공시 · 시가총액 US$10억 이상")):
        L.append(f"\n### {title}\n")
        L.append("| 묶음 | CAGR | 샤프 | MDD(월) | 알파(연) · t | SPY 이긴 해 | SPY 보유 해 | 종목 승률 | 폭락(≤−25%) | 샤프 차이 vs SPY [95%] | 플라시보 백분위 |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
        for s in STRATS:
            b = res["U"][u][s]
            f = b["full"]
            vs = f["vs_spy"] or {}
            ci = f"{vs.get('ΔSharpe', float('nan')):+.2f} [{vs.get('ΔSharpe_하한', float('nan')):+.2f}, {vs.get('ΔSharpe_상한', float('nan')):+.2f}]" if vs else "—"
            L.append(f"| {names[s]} | {pct(f['cagr'])} | {f['sharpe']:.2f} | {pct(f['mdd'])} | {pct(f['alpha'])} · {f['alpha_t']:.2f} | "
                     f"{b['beat_years']}/{len(b['period'])} | {b['spy_years']} | {pct(b['win'], 0)} | {pct(b['crash'], 0)} | {ci} | "
                     f"{b['placebo']['pct']:.0f} |")
    L.append("\n### 사전 등록 판정\n")
    for s, v in res["verdicts"].items():
        L.append(f"- {names[s]}: **{v}**")
    return "\n".join(L)


def cmd_run(_args: argparse.Namespace) -> int:
    inp = load_inputs()
    bt = run_backtest(inp)
    res = summarize(bt)
    slim = json.loads(json.dumps(res, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o)))
    (OUT / "results.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1), "utf-8")
    md = report_md(res)
    (OUT / "report.md").write_text(md, "utf-8")
    print(md)
    log(f"표시: {res['flags']}")
    return 0


# ── 명령

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch-sec", "fetch-prices", "build", "run"):
        sub.add_parser(name)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    return {"fetch-sec": cmd_fetch_sec, "fetch-prices": cmd_fetch_prices, "build": cmd_build, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
