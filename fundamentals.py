"""yfinance 기반의 기초 재무 지표 스냅샷 수집."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


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


def fetch_fundamental_snapshots(tickers: List[str]) -> pd.DataFrame:
    """ROE, 부채비율 등 간단한 재무 상태 지표를 내려받아 표 형식으로 반환한다."""

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

        record = {
            "티커": symbol,
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
            "fund_float_shares": _safe_float(info.get("floatShares")),
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
        "days_to_next_earnings",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
