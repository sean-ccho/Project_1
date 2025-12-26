"""섹터 로테이션 로직: 강한 섹터 판별 및 필터링."""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
import pandas as pd

from config import (
    SECTOR_ETFS,
    SECTOR_STRENGTH_LOOKBACK,
    SECTOR_STRENGTH_THRESHOLD,
)


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
