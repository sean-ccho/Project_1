"""yfinance 기반의 기초 재무 지표 스냅샷 수집."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

# 펀더멘탈 디스크 캐시 설정
_FUND_CACHE_DIR = Path("data/cache")
_FUND_CACHE_MAX_AGE_HOURS = 24


def _fund_cache_path(tickers: List[str]) -> Path:
    key = ",".join(sorted(tickers))
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return _FUND_CACHE_DIR / f"fundamentals_{h}.parquet"


def _fund_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
    return age_hours < _FUND_CACHE_MAX_AGE_HOURS


def _safe_float(value) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _to_datetime(value) -> Optional[pd.Timestamp]:
    if value in (None, "", 0):
        return None
    try:
        ts = pd.to_datetime(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if isinstance(ts, pd.Series):
        ts = ts.dropna()
        if ts.empty:
            return None
        ts = ts.iloc[0]
    if isinstance(ts, (list, tuple)):
        ts = pd.Series(ts).dropna()
        if ts.empty:
            return None
        ts = pd.to_datetime(ts.iloc[0])
    if isinstance(ts, pd.Timestamp):
        return ts.tz_localize(None) if ts.tzinfo else ts
    if isinstance(ts, datetime):
        return pd.Timestamp(ts)
    return None


_MARKET_TZ = ZoneInfo("America/New_York")
# yfinance 1.x 는 실적일을 epoch 초(earningsTimestamp*)로 준다. 옛 키(earningsDate)는 예전 버전 호환용
_EARNINGS_KEYS = ("earningsTimestampStart", "earningsTimestamp", "earningsDate", "nextEarningsDate")


def _to_market_date(value) -> Optional[date]:
    """yfinance 실적일 값(epoch 초 · 날짜 문자열 · 리스트) → 미국 동부 날짜. 해석 못 하면 None."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value) or value <= 0:
            return None
        return pd.Timestamp(float(value), unit="s", tz="UTC").tz_convert(_MARKET_TZ).date()
    ts = _to_datetime(value)
    return ts.date() if ts is not None else None


def _extract_earnings_date(info: Dict, today: Optional[date] = None) -> tuple[str, float]:
    """다음 실적 발표일과 남은 달력일 (미국 동부 날짜 기준).

    후보 키 중 오늘 이후 날짜의 가장 이른 값을 쓴다. 이미 지난 날짜뿐이면 ('', nan) — 다음 일정 미정.
    실행 시각(20:45 ET = 다음날 00:45 UTC)에 UTC 날짜를 쓰면 하루가 밀리므로 ET 날짜로 센다.
    """
    today = today or datetime.now(_MARKET_TZ).date()
    upcoming = [d for d in (_to_market_date(info.get(k)) for k in _EARNINGS_KEYS) if d is not None and d >= today]
    if not upcoming:
        return "", float("nan")
    nxt = min(upcoming)
    return nxt.strftime("%Y-%m-%d"), float((nxt - today).days)


# 웹 종목 상세 전용 숫자 칸 (2026-11-03, web_export.market_summary) — 밸류에이션 배수 · 애널리스트 목표주가 · 의견
_WEB_INFO_NUMS = {
    "fund_pe": "trailingPE",
    "fund_forward_pe": "forwardPE",
    "fund_peg": "trailingPegRatio",
    "fund_ps": "priceToSalesTrailing12Months",
    "fund_pb": "priceToBook",
    "fund_ev_ebitda": "enterpriseToEbitda",
    "fund_target_mean": "targetMeanPrice",
    "fund_target_median": "targetMedianPrice",
    "fund_target_high": "targetHighPrice",
    "fund_target_low": "targetLowPrice",
    "fund_analyst_n": "numberOfAnalystOpinions",
    "fund_rec_mean": "recommendationMean",
}


def fetch_fundamental_snapshots(tickers: List[str], use_cache: bool = False) -> pd.DataFrame:
    """ROE, 부채비율 등 간단한 재무 상태 지표를 내려받아 표 형식으로 반환한다.

    use_cache=True 일 때 24시간 이내 디스크 캐시가 있으면 API 호출을 건너뛴다.
    백테스트에서는 run_paper_trading_backtest()가 시뮬레이션 시작 전에 한 번만 호출하므로
    use_cache=False(기본값)로도 반복 호출이 발생하지 않는다.
    """
    if use_cache:
        cache_path = _fund_cache_path(tickers)
        if _fund_cache_valid(cache_path):
            print(f"[백테스트] Fundamentals 캐시 로드: {cache_path.name}")
            return pd.read_parquet(cache_path)

    records = []
    for raw_symbol in tickers:
        symbol = str(raw_symbol).upper().strip()
        if not symbol:
            continue
        try:
            info = yf.Ticker(symbol).get_info()
        except Exception:
            info = {}

        next_earnings_date, days_to_next_earnings = _extract_earnings_date(info)

        # Short interest 관련 데이터
        short_interest = _safe_float(info.get("sharesShort"))
        float_shares = _safe_float(info.get("floatShares"))
        short_pct_float = _safe_float(info.get("shortPercentOfFloat"))
        # yfinance가 shortPercentOfFloat를 제공하지 않을 경우 직접 계산
        if np.isnan(short_pct_float) and not np.isnan(short_interest) and not np.isnan(float_shares) and float_shares > 0:
            short_pct_float = short_interest / float_shares

        # 기관 보유 비율
        inst_holders_pct = _safe_float(info.get("heldPercentInstitutions"))

        record = {
            "티커": symbol,
            "fund_sector": info.get("sector", "Unknown"),  # 섹터 정보 추가
            # 웹 표시 전용 (2026-10-28) — 세부 업종 · 회사 소개(영어). 점수·필터에 안 쓴다
            "fund_industry": info.get("industry") or None,
            "fund_summary": info.get("longBusinessSummary") or None,
            # Yahoo 정기 배당(1년, 달러/주) — 0 = 정기 배당 없음(특별배당만·안 줌), NaN = 정보를 못 받음
            "fund_dividend_rate": (_safe_float(info.get("dividendRate")) if info.get("dividendRate") else 0.0) if info else np.nan,
            "fund_roe": _safe_float(info.get("returnOnEquity")),
            "fund_debt_to_equity": _safe_float(info.get("debtToEquity")),
            "fund_revenue_growth": _safe_float(info.get("revenueGrowth")),
            "fund_profit_margin": _safe_float(info.get("profitMargins")),
            "fund_gross_margin": _safe_float(info.get("grossMargins")),
            "fund_earnings_growth": _safe_float(info.get("earningsGrowth")),
            "fund_current_ratio": _safe_float(info.get("currentRatio")),
            "fund_quick_ratio": _safe_float(info.get("quickRatio")),
            "fund_free_cashflow": _safe_float(info.get("freeCashflow")),
            "fund_operating_cashflow": _safe_float(info.get("operatingCashflow")),
            "fund_market_cap": _safe_float(info.get("marketCap")),
            "fund_float_shares": float_shares,
            # Phase 3: 추가 데이터 소스
            "fund_short_interest": short_interest,
            "fund_short_pct_float": short_pct_float,
            "fund_institutional_holders_pct": inst_holders_pct,
            "next_earnings_date": next_earnings_date,
            "days_to_next_earnings": days_to_next_earnings,
            # 웹 표시 전용 (2026-11-03) — 종목 상세 밸류에이션 · 애널리스트 카드. 점수 · 필터에 안 쓴다 (같은 info 호출, 추가 요청 없음)
            **{k: _safe_float(info.get(src)) for k, src in _WEB_INFO_NUMS.items()},
            "fund_rec_key": info.get("recommendationKey") or None,
            # 웹 표시 전용 (2026-11-18) — 통화 코드. 현금흐름 · 매출은 회사 보고 통화(financialCurrency: Sony 는 JPY)라
            # 거래 통화(currency: ADR 은 USD, 캐나다 상장은 대개 CAD)와 다를 수 있다. 점수 · 필터에 안 쓴다 (같은 info 호출)
            "fund_fin_ccy": info.get("financialCurrency") or None,
            "fund_ccy": info.get("currency") or None,
        }

        records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    numeric_cols = [
        "fund_roe",
        "fund_debt_to_equity",
        "fund_revenue_growth",
        "fund_profit_margin",
        "fund_gross_margin",
        "fund_earnings_growth",
        "fund_current_ratio",
        "fund_quick_ratio",
        "fund_free_cashflow",
        "fund_operating_cashflow",
        "fund_market_cap",
        "fund_float_shares",
        "fund_short_interest",
        "fund_short_pct_float",
        "fund_institutional_holders_pct",
        "fund_dividend_rate",
        "days_to_next_earnings",
        *_WEB_INFO_NUMS,
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 캐시 저장 (use_cache=True일 때만)
    if use_cache:
        try:
            cache_path = _fund_cache_path(tickers)
            _FUND_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            print(f"[백테스트] Fundamentals 캐시 저장: {cache_path.name}")
        except Exception as e:
            print(f"[백테스트] Fundamentals 캐시 저장 실패 (무시): {e}")

    return df


# ── 표시용 글자 (헬스체크 근거 줄 · 시가총액) — 점수 계산은 screener.health_score ──────


def _format_pct(val) -> str:
    """수치를 +35% / -10% / ? 로 포맷."""
    if val is None:
        return "?"
    try:
        x = float(val)
    except (TypeError, ValueError):
        return "?"
    if np.isnan(x):
        return "?"
    sign = "+" if x > 0 else ""
    return f"{sign}{x*100:.0f}%"


def format_growth_indicators(row) -> str:
    """매출/마진/이익 성장 한 줄 — 예: '매출+35% 마진+38% 이익+25%'."""
    g = row.get if hasattr(row, "get") else (lambda k: None)
    return (
        f"매출{_format_pct(g('fund_revenue_growth'))} "
        f"마진{_format_pct(g('fund_profit_margin'))} "
        f"이익{_format_pct(g('fund_earnings_growth'))}"
    )


def format_market_cap(row) -> str:
    """시가총액 — $5.2T / $8.3B / $850M / ?."""
    v = row.get("fund_market_cap") if hasattr(row, "get") else None
    if v is None:
        return "?"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "?"
    if np.isnan(x) or x <= 0:
        return "?"
    if x >= 1e12: return f"${x/1e12:.1f}T"
    if x >= 1e9:  return f"${x/1e9:.1f}B"
    if x >= 1e6:  return f"${x/1e6:.0f}M"
    return f"${x:.0f}"
