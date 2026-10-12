"""시세 및 부가 정보 수집 유틸리티."""

import logging
import os
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_RETRIES = 4
RETRY_DELAY_SECONDS = 10

# OHLCV 배치 다운로드 설정
_OHLCV_BATCH_SIZE = 50         # 배치당 티커 수 (Yahoo rate limit 대응: 150→50)
_OHLCV_BATCH_DELAY = 3.0       # 배치 간 대기 시간(초) (1.5→3)

# OHLCV 디스크 캐시 설정 (백테스트용)
_OHLCV_CACHE_DIR = Path("data/cache")
_OHLCV_CACHE_MAX_AGE_HOURS = 24
_IS_CI = bool(os.environ.get("CI"))


def _ohlcv_cache_path(tickers: List[str], period: str) -> Path:
    key = ",".join(sorted(tickers)) + "|" + period
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return _OHLCV_CACHE_DIR / f"ohlcv_{h}.parquet"


def _ohlcv_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
    return age_hours < _OHLCV_CACHE_MAX_AGE_HOURS
COMPANY_NAME_CACHE: Dict[str, str] = {}
COMPANY_NAME_CACHE_FILE = Path("src/data/company_names_cache.json")

if COMPANY_NAME_CACHE_FILE.exists():
    try:
        COMPANY_NAME_CACHE.update(
            json.loads(COMPANY_NAME_CACHE_FILE.read_text())
        )
    except Exception:
        COMPANY_NAME_CACHE.clear()


_YF_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}


def _yf_range(period: str) -> dict[str, str]:
    """yfinance 기간 인자. yfinance 는 "10y" 까지만 받으므로 "13y" 같은 값은 시작 날짜로 바꾼다."""
    if period in _YF_PERIODS or not (period.endswith("y") and period[:-1].isdigit()):
        return {"period": period}
    start = datetime.now() - timedelta(days=int(round(365.25 * int(period[:-1]))))
    return {"start": start.strftime("%Y-%m-%d")}


def _download_ohlcv_batch(
    tickers: List[str],
    period: str,
) -> pd.DataFrame | None:
    """단일 배치의 OHLCV 데이터를 다운로드하고 (티커, 필드) 멀티인덱스로 정규화한다.

    실패 시 None을 반환한다.
    """
    import time as _time

    last_exception: Exception | None = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            data = yf.download(
                tickers,
                **_yf_range(period),
                interval="1d",
                auto_adjust=True,
                threads=True,
                progress=False,
                actions=True,
            )
            if not data.empty:
                break
        except Exception as exc:
            last_exception = exc
            if attempt < MAX_DOWNLOAD_RETRIES:
                _time.sleep(RETRY_DELAY_SECONDS)
                continue
            logger.warning(f"[fetch_ohlcv] 배치 다운로드 실패: {exc}")
            return None
    else:
        if last_exception:
            logger.warning(f"[fetch_ohlcv] 배치 다운로드 실패: {last_exception}")
        return None

    # 멀티인덱스 정규화: (필드, 티커) → (티커, 필드)
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            data = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
    else:
        # 단일 티커 요청 시에도 멀티인덱스로 승격
        data = pd.concat({tickers[0]: data}, axis=1)

    return data


def fetch_ohlcv(tickers: List[str], period: str = "1y", force_download: bool = False) -> pd.DataFrame:
    """지정한 티커들의 일별 OHLCV 데이터를 내려받아 (티커, 필드) 멀티인덱스로 반환한다.

    force_download=False 일 때 24시간 이내의 디스크 캐시가 있으면 캐시를 사용한다.
    티커 수가 _OHLCV_BATCH_SIZE 초과 시 배치로 나눠서 다운로드한다.
    """
    import time as _time

    cache_path = _ohlcv_cache_path(tickers, period)

    if not _IS_CI and not force_download and _ohlcv_cache_valid(cache_path):
        print(f"[백테스트] OHLCV 캐시 로드: {cache_path.name}")
        return pd.read_parquet(cache_path)

    # --- 배치 다운로드 ---
    if len(tickers) <= _OHLCV_BATCH_SIZE:
        data = _download_ohlcv_batch(tickers, period)
        if data is None or data.empty:
            raise RuntimeError("yfinance download returned empty data after retries")
    else:
        frames: list[pd.DataFrame] = []
        total_batches = (len(tickers) + _OHLCV_BATCH_SIZE - 1) // _OHLCV_BATCH_SIZE
        skipped = 0

        for i in range(0, len(tickers), _OHLCV_BATCH_SIZE):
            batch = tickers[i : i + _OHLCV_BATCH_SIZE]
            batch_num = i // _OHLCV_BATCH_SIZE + 1
            print(
                f"[OHLCV] 배치 다운로드 {batch_num}/{total_batches} "
                f"({len(batch)}개 종목, period={period})..."
            )

            batch_data = _download_ohlcv_batch(batch, period)
            if batch_data is not None and not batch_data.empty:
                frames.append(batch_data)
            else:
                skipped += len(batch)

            if i + _OHLCV_BATCH_SIZE < len(tickers):
                _time.sleep(_OHLCV_BATCH_DELAY)

        if not frames:
            raise RuntimeError("yfinance download returned empty data for all batches")

        data = pd.concat(frames, axis=1)

        if skipped > 0:
            print(f"[OHLCV] 경고: {skipped}개 종목 다운로드 실패 (건너뜀)")

    # 데이터 품질 검증 (정보용) — 종류별로 한 줄씩
    try:
        from data.validators import summarize_warnings, validate_ohlcv
        for line in summarize_warnings(validate_ohlcv(data)):
            print(line)
    except Exception as e:
        logger.debug(f"데이터 검증 스킵: {e}")

    # 디스크 캐시 저장
    if not _IS_CI:
        try:
            _OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            data.to_parquet(cache_path)
            print(f"[백테스트] OHLCV 캐시 저장: {cache_path.name}")
        except Exception as e:
            print(f"[백테스트] OHLCV 캐시 저장 실패 (무시): {e}")

    return data


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
    import time

    prices: Dict[str, float] = {}
    if not tickers:
        return prices

    for raw_symbol in tickers:
        symbol = str(raw_symbol).upper().strip()
        if not symbol:
            continue

        fetched = False
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
                    fetched = True
                    break
                # close가 None이면 rate limit 또는 데이터 없음 — 재시도
                if attempt < MAX_DOWNLOAD_RETRIES:
                    print(f"[fetch_latest_prices] {symbol}: 가격 없음 (시도 {attempt}/{MAX_DOWNLOAD_RETRIES}), {RETRY_DELAY_SECONDS}초 후 재시도")
                    time.sleep(RETRY_DELAY_SECONDS)
            except Exception as exc:
                if attempt < MAX_DOWNLOAD_RETRIES:
                    print(f"[fetch_latest_prices] {symbol}: 오류 (시도 {attempt}/{MAX_DOWNLOAD_RETRIES}) — {exc}")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    print(f"[fetch_latest_prices] {symbol}: 최종 실패 — {exc}")

        if not fetched:
            print(f"[fetch_latest_prices] {symbol}: 가격 조회 실패, nan으로 처리됨")

    return prices
