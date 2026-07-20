"""yfinance 기반의 기초 재무 지표 스냅샷 수집."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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


def _extract_earnings_date(info: Dict) -> tuple[str, float]:
    raw = info.get("earningsDate") or info.get("nextEarningsDate")
    timestamp = None
    if isinstance(raw, (list, tuple)) and raw:
        timestamp = _to_datetime(raw[0])
    else:
        timestamp = _to_datetime(raw)

    if not timestamp:
        return "", float("nan")

    days_to = (timestamp.date() - datetime.utcnow().date()).days
    return timestamp.strftime("%Y-%m-%d"), float(days_to)


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
        "days_to_next_earnings",
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


# ── 회사 건전성 점수 (display-only) ────────────────────────────


def _health_grade(score: int) -> str:
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"


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


def calculate_health_score(row) -> tuple[int | None, str]:
    """회사 건전성 점수 (0-100) + 표시 문자열.

    8개 펀더멘털 항목을 가중치 합산. 결측은 미가산.
    평가 가능 항목이 4개 미만이면 N/A.

    Returns:
        (score, display)
        - score: 0-100 또는 None
        - display: "85 (A) · 매출+35% 마진+38% 이익+25%" 또는 "N/A"
          (시트는 그대로 한 줄 사용, 이메일은 ' · '로 split해서 두 줄 렌더)
    """
    from screener.config import (
        FUND_HEALTH_ROE_MIN,
        FUND_HEALTH_DEBT_TO_EQUITY_MAX,
        FUND_HEALTH_REVENUE_GROWTH_MIN,
        FUND_HEALTH_PROFIT_MARGIN_MIN,
        FUND_HEALTH_EARNINGS_GROWTH_MIN,
        FUND_HEALTH_CURRENT_RATIO_MIN,
    )

    # (컬럼명, 임계값, 점수, "gt"/"lt")
    items = [
        ("fund_roe",                       FUND_HEALTH_ROE_MIN,             20, "gt"),
        ("fund_debt_to_equity",            FUND_HEALTH_DEBT_TO_EQUITY_MAX,  15, "lt"),
        ("fund_revenue_growth",            FUND_HEALTH_REVENUE_GROWTH_MIN,  15, "gt"),
        ("fund_profit_margin",             FUND_HEALTH_PROFIT_MARGIN_MIN,   15, "gt"),
        ("fund_earnings_growth",           FUND_HEALTH_EARNINGS_GROWTH_MIN, 10, "gt"),
        ("fund_current_ratio",             FUND_HEALTH_CURRENT_RATIO_MIN,   10, "gt"),
        ("fund_free_cashflow",             0,                               10, "gt"),
        ("fund_institutional_holders_pct", 0.60,                             5, "gt"),
    ]

    total = 0
    evaluated = 0
    for col, thr, pts, op in items:
        val = row.get(col) if hasattr(row, "get") else None
        if val is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if np.isnan(v):
            continue
        evaluated += 1
        if (op == "gt" and v > thr) or (op == "lt" and v < thr):
            total += pts

    if evaluated < 4:
        return None, "N/A"

    grade = _health_grade(total)
    return total, f"{total} ({grade}) · {format_growth_indicators(row)}"
