"""시세 및 부가 정보 수집 유틸리티."""

from typing import Dict, List

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

MAX_DOWNLOAD_RETRIES = 3
RETRY_DELAY_SECONDS = 3
COMPANY_NAME_CACHE: Dict[str, str] = {}
COMPANY_NAME_CACHE_FILE = Path("data/company_names_cache.json")

if COMPANY_NAME_CACHE_FILE.exists():
    try:
        COMPANY_NAME_CACHE.update(
            json.loads(COMPANY_NAME_CACHE_FILE.read_text())
        )
    except Exception:
        COMPANY_NAME_CACHE.clear()


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


def fetch_company_names(tickers: List[str]) -> Dict[str, str]:
    """yfinance Ticker API를 활용해 종목명을 가져온다."""

    names: Dict[str, str] = {}
    if not tickers:
        return names

    import time

    updated = False
    for raw_symbol in tickers:
        symbol = str(raw_symbol).upper().strip()
        if not symbol:
            continue
        if symbol in COMPANY_NAME_CACHE:
            names[symbol] = COMPANY_NAME_CACHE[symbol]
            continue

        retrieved = False
        last_exc: Exception | None = None
        for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
            try:
                info = yf.Ticker(symbol).get_info()
                name = (
                    info.get("shortName")
                    or info.get("longName")
                    or info.get("displayName")
                    or ""
                )
                if name:
                    COMPANY_NAME_CACHE[symbol] = name
                    names[symbol] = name
                    updated = True
                retrieved = True
                break
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_DOWNLOAD_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
            time.sleep(0.1)
        if not retrieved and last_exc:
            print(f"[fetch_company_names] {symbol}: {last_exc}")

    if updated:
        try:
            COMPANY_NAME_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            COMPANY_NAME_CACHE_FILE.write_text(
                json.dumps(COMPANY_NAME_CACHE, ensure_ascii=False)
            )
        except Exception:
            pass

    return names


def fetch_latest_prices(tickers: List[str]) -> Dict[str, float]:
    """현재/최근 종가를 딕셔너리로 반환한다."""

    prices: Dict[str, float] = {}
    if not tickers:
        return prices

    for raw_symbol in tickers:
        symbol = str(raw_symbol).upper().strip()
        if not symbol:
            continue

        last_exc: Exception | None = None
        for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
            try:
                info = yf.Ticker(symbol)
                close = info.fast_info.get("lastPrice") or info.fast_info.get(
                    "previousClose"
                )
                if close is None:
                    hist = info.history(period="1d")
                    if not hist.empty and "Close" in hist.columns:
                        close = hist["Close"].iloc[-1]
                if close is not None:
                    prices[symbol] = float(close)
                break
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_DOWNLOAD_RETRIES:
                    import time

                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
        else:
            if last_exc:
                print(f"[fetch_latest_prices] {symbol}: {last_exc}")

    return prices
