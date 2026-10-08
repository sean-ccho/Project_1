"""섹터 로테이션 로직: 강한 섹터 판별 및 필터링."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from screener.config import (
    SECTOR_ETFS,
    SECTOR_STRENGTH_LOOKBACK,
    SECTOR_STRENGTH_THRESHOLD,
)

# yfinance 섹터명 → GICS 섹터명 (SECTOR_ETFS 키). 펀더멘털을 켜면 `섹터`가 yfinance 이름으로 바뀌어
# 강한 섹터(GICS 이름)와 비교가 안 맞던 문제(Technology·Healthcare 등 6개 섹터 buy_signal 상시 차단) 수정용.
SECTOR_ALIASES: Dict[str, str] = {
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
}


def to_gics(sector: object) -> str:
    """섹터명을 GICS 이름으로 (모르는 값·결측은 그대로 / 'Unknown')."""
    if sector is None or (isinstance(sector, float) and np.isnan(sector)) or str(sector).strip() in ("", "nan"):
        return "Unknown"
    s = str(sector).strip()
    return SECTOR_ALIASES.get(s, s)


_SECTOR_CSV = Path(__file__).resolve().parents[2] / "data" / "universe" / "sector_map.csv"
_SECTOR_CSV_MAP: Dict[str, str] | None = None


def _sector_csv_map() -> Dict[str, str]:
    """data/universe/sector_map.csv (현재 S&P 500 GICS 섹터). 없으면 빈 dict."""
    global _SECTOR_CSV_MAP
    if _SECTOR_CSV_MAP is None:
        try:
            df = pd.read_csv(_SECTOR_CSV)
            _SECTOR_CSV_MAP = dict(zip(df["ticker"].astype(str), df["sector"].astype(str)))
        except (OSError, KeyError, ValueError):
            _SECTOR_CSV_MAP = {}
    return _SECTOR_CSV_MAP


def fill_unknown_sectors(sectors: pd.Series, tickers: pd.Series) -> pd.Series:
    """Unknown·결측 섹터를 sector_map.csv(GICS)로 채우고 GICS 이름으로 통일한다.

    펀더멘털 없는 백테스트는 config.SECTOR_MAP 에 없는 종목이 전부 Unknown(S&P 500의 ~78%)이라
    "Unknown은 통과" 규칙 때문에 섹터 필터가 사실상 꺼져 실거래와 달랐다.
    """
    smap = _sector_csv_map()
    g = sectors.map(to_gics)
    fill = tickers.astype(str).map(smap)
    return g.where(g != "Unknown", fill).fillna("Unknown").map(to_gics)


def mark_strong_sectors(sectors: pd.Series, strong_sectors: Set[str]) -> pd.Series:
    """섹터 열 → 강한 섹터 여부 (실거래 규칙: Unknown 은 통과)."""
    return sectors.map(lambda s: is_strong_or_unknown(s, strong_sectors))


def is_strong_or_unknown(sector: object, strong_sectors: Set[str]) -> bool:
    """실거래 규칙: 강한 섹터(GICS 이름 비교)이거나 섹터를 모르면 True."""
    g = to_gics(sector)
    return g == "Unknown" or g in strong_sectors


def compute_sector_strength(
    etf_data: Dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
) -> Dict[str, float]:
    """각 섹터 ETF의 상대 강도(SPY 대비 초과 수익률)를 계산한다.
    
    Args:
        etf_data: 섹터 ETF 티커 -> OHLCV DataFrame 매핑
        spy_data: SPY의 OHLCV DataFrame
        
    Returns:
        섹터명 -> 상대 강도(%) 매핑
    """
    lookback = SECTOR_STRENGTH_LOOKBACK
    
    # SPY 수익률 계산
    spy_close = spy_data["Close"].dropna()
    if len(spy_close) < lookback + 1:
        return {}
    
    spy_return = (spy_close.iloc[-1] / spy_close.iloc[-lookback - 1]) - 1.0
    
    sector_strength: Dict[str, float] = {}
    
    for sector_name, etf_ticker in SECTOR_ETFS.items():
        if etf_ticker not in etf_data:
            continue
        
        etf_close = etf_data[etf_ticker]["Close"].dropna()
        if len(etf_close) < lookback + 1:
            continue
            
        etf_return = (etf_close.iloc[-1] / etf_close.iloc[-lookback - 1]) - 1.0
        relative_strength = etf_return - spy_return
        sector_strength[sector_name] = float(relative_strength)
    
    return sector_strength


def get_strong_sectors(
    etf_data: Dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
    threshold: float | None = None,
) -> Set[str]:
    """SPY 대비 강한 섹터 목록을 반환한다.
    
    Args:
        etf_data: 섹터 ETF 티커 -> OHLCV DataFrame 매핑
        spy_data: SPY의 OHLCV DataFrame
        threshold: 상대 강도 임계값 (기본값: config에서 가져옴)
        
    Returns:
        강한 섹터 이름 집합
    """
    if threshold is None:
        threshold = SECTOR_STRENGTH_THRESHOLD
        
    strength = compute_sector_strength(etf_data, spy_data)
    
    strong = {
        sector for sector, rel_str in strength.items()
        if rel_str >= threshold
    }
    
    return strong


def filter_by_strong_sectors(
    df: pd.DataFrame,
    strong_sectors: Set[str],
) -> pd.Series:
    """종목이 강한 섹터에 속하는지 여부를 반환한다.
    
    Args:
        df: 종목 데이터프레임 (섹터 컬럼 필요)
        strong_sectors: 강한 섹터 집합
        
    Returns:
        강한 섹터 소속 여부 불리언 시리즈
    """
    if "섹터" not in df.columns:
        # 섹터 정보가 없으면 모두 통과
        return pd.Series(True, index=df.index)
    
    return df["섹터"].isin(strong_sectors) | (df["섹터"] == "Unknown")
