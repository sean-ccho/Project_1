"""시세 수집 유틸리티."""

from typing import List

import pandas as pd
import yfinance as yf

MAX_DOWNLOAD_RETRIES = 3
RETRY_DELAY_SECONDS = 3


def fetch_ohlcv(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """지정한 티커들의 일별 OHLCV 데이터를 내려받아 (티커, 필드) 멀티인덱스로 반환한다."""

    # auto_adjust=True 설정으로 분할/배당 효과가 반영된 수정주가를 수집한다.
    last_exception: Exception | None = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            data = yf.download(
                tickers,
                period=period,
                interval="1d",
                auto_adjust=True,
                threads=True,
                progress=False,
                actions=True,
            )
            if not data.empty:
                break
        except Exception as exc:  # yfinance가 다양한 사용자 정의 예외를 던짐
            last_exception = exc
            if attempt < MAX_DOWNLOAD_RETRIES:
                import time

                time.sleep(RETRY_DELAY_SECONDS)
                continue
            raise
    else:
        if last_exception:
            raise last_exception
        raise RuntimeError("yfinance download returned empty data after retries")

    if isinstance(data.columns, pd.MultiIndex):
        # yfinance 기본 반환 형태는 (필드, 티커)이므로 축을 뒤집어 (티커, 필드)로 맞춘다.
        if "Close" in data.columns.get_level_values(0):
            return data.swaplevel(0, 1, axis=1).sort_index(axis=1)
        return data

    # 단일 티커 요청 시에도 downstream 로직이 동일하게 동작하도록 멀티인덱스로 승격한다.
    return pd.concat({tickers[0]: data}, axis=1)
