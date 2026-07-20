"""시세 및 부가 정보 수집 유틸리티."""

import logging
import os
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_RETRIES = 3
RETRY_DELAY_SECONDS = 3

# OHLCV 배치 다운로드 설정
_OHLCV_BATCH_SIZE = 150        # 배치당 티커 수 (1단계/2단계와 동일)
_OHLCV_BATCH_DELAY = 1.5       # 배치 간 대기 시간(초)

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


def _translate_text(payload: str) -> str:
    """뉴스 텍스트를 그대로 반환한다."""

    return payload


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
                period=period,
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

    # 데이터 품질 검증
    try:
        from data.validators import validate_ohlcv
        warnings = validate_ohlcv(data)
        if warnings:
            print(f"[DataValidator] {len(warnings)}건의 데이터 품질 경고 발생")
            for w in warnings:
                print(f"Warning:  {w.ticker}: {w.message}")
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


def _prepare_news_entry(entry: dict) -> dict | None:
    """Normalize raw news payload into display-friendly fields."""

    def _clean(value: object) -> str:
        return str(value).strip() if value is not None else ""

    payload = entry.get("content") or entry

    title = _clean(
        payload.get("title")
        or entry.get("title")
        or payload.get("headline")
        or entry.get("headline")
    )

    link = _clean(
        entry.get("link")
        or payload.get("link")
        or (payload.get("canonicalUrl") or {}).get("url")
        or (payload.get("clickThroughUrl") or {}).get("url")
        or entry.get("url")
    )

    publisher = _clean(
        entry.get("publisher")
        or payload.get("publisher")
        or (payload.get("provider") or {}).get("displayName")
        or (entry.get("provider") or {}).get("displayName")
        or (payload.get("source") or {}).get("name")
    )

    published_ts = (
        payload.get("providerPublishTime")
        or entry.get("providerPublishTime")
        or payload.get("pubDate")
        or entry.get("pubDate")
        or payload.get("date")
    )

    published_dt: datetime | None = None
    if published_ts:
        try:
            published_dt = datetime.fromtimestamp(int(published_ts), tz=timezone.utc)
        except Exception:
            try:
                normalized = str(published_ts).replace("Z", "+00:00")
                published_dt = datetime.fromisoformat(normalized)
                if published_dt.tzinfo is None:
                    published_dt = published_dt.replace(tzinfo=timezone.utc)
            except Exception:
                published_dt = None

    published_str = (
        published_dt.strftime("%Y-%m-%d %H:%M UTC")
        if published_dt is not None
        else ""
    )

    display_parts = [part for part in [title, publisher, published_str] if part]
    display_text = " | ".join(display_parts) if display_parts else (title or link)
    display_text = display_text or "제목 없음"

    return {
        "text": display_text,
        "link": link,
        "timestamp": published_dt,
    }


def _escape_formula_text(text: str) -> str:
    """Escape double quotes for safe insertion into spreadsheet formulas."""

    return text.replace("\"", "\"\"")


def _format_news_entries(entries: list[dict]) -> str:
    """Convert prepared entries into a newline-delimited, clickable string."""

    if not entries:
        return "최근 2일 뉴스 없음"

    tokens: list[str] = []
    plain_lines: list[str] = []
    for entry in entries:
        text = entry.get("text", "제목 없음") or "제목 없음"
        link = entry.get("link", "").strip()
        plain_lines.append(text)
        if link:
            safe_link = link.replace('"', "")
            safe_text = _escape_formula_text(text)
            tokens.append(f'HYPERLINK("{safe_link}", "{safe_text}")')
        else:
            tokens.append("")

    if all(token for token in tokens):
        if len(tokens) == 1:
            return f'={tokens[0]}'
        formula = " & CHAR(10) & ".join(tokens)
        return f"={formula}"

    return "\n".join(plain_lines)


def fetch_analyst_data(tickers: List[str]) -> Dict[str, dict]:
    """yfinance를 이용해 애널리스트 컨센서스 데이터를 가져온다.

    반환 구조:
        {
            "VLO": {
                "target_mean": 248.0,
                "current_price": 226.0,
                "upside_pct": 0.097,
                "num_analysts": 18,
                "recommendation": "Buy",   # "Buy" / "Hold" / "Sell" / "N/A"
            },
            ...
        }
    실패한 티커는 결과에서 생략한다.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    result: Dict[str, dict] = {}
    if not tickers:
        return result

    def _fetch_one(symbol: str) -> tuple[str, dict | None]:
        for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
            try:
                ticker_obj = yf.Ticker(symbol)

                # 현재가
                fast = ticker_obj.fast_info
                current_price: float | None = (
                    fast.get("lastPrice") or fast.get("previousClose")
                )

                # 애널리스트 목표가
                apt = ticker_obj.analyst_price_targets
                target_mean: float | None = None
                num_analysts: int = 0
                if apt is not None and not apt.empty:
                    row = apt.iloc[-1] if hasattr(apt, "iloc") else apt
                    target_mean = float(row.get("mean") or row.get("targetMeanPrice") or 0) or None
                    num_analysts = int(row.get("numberOfAnalystOpinions") or row.get("numAnalysts") or 0)

                # 컨센서스 추천
                rec_summary = ticker_obj.recommendations_summary
                recommendation = "N/A"
                if rec_summary is not None and not rec_summary.empty:
                    latest = rec_summary.iloc[0]
                    buy_cnt = int(latest.get("strongBuy", 0) or 0) + int(latest.get("buy", 0) or 0)
                    hold_cnt = int(latest.get("hold", 0) or 0)
                    sell_cnt = int(latest.get("sell", 0) or 0) + int(latest.get("strongSell", 0) or 0)
                    total = buy_cnt + hold_cnt + sell_cnt
                    if total > 0:
                        if buy_cnt / total >= 0.5:
                            recommendation = "Buy"
                        elif sell_cnt / total >= 0.4:
                            recommendation = "Sell"
                        else:
                            recommendation = "Hold"
                        if num_analysts == 0:
                            num_analysts = total

                upside_pct: float | None = None
                if target_mean and current_price and current_price > 0:
                    upside_pct = (target_mean - current_price) / current_price

                return symbol, {
                    "target_mean": target_mean,
                    "current_price": current_price,
                    "upside_pct": upside_pct,
                    "num_analysts": num_analysts,
                    "recommendation": recommendation,
                }
            except Exception as exc:
                if attempt < MAX_DOWNLOAD_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                logger.warning(f"[fetch_analyst_data] {symbol}: {exc}")
                return symbol, None
        return symbol, None

    unique = list(dict.fromkeys(str(t).upper().strip() for t in tickers if str(t).strip()))
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fetch_one, s): s for s in unique}
        for fut in as_completed(futures):
            symbol, data = fut.result()
            if data is not None:
                result[symbol] = data

    return result


def fetch_latest_news(tickers: List[str], max_items: int = 1) -> Dict[str, str]:
    """Return the most recent news headline/link for each ticker."""

    news_map: Dict[str, str] = {}
    if not tickers:
        return news_map

    import time

    seen: set[str] = set()
    for raw_symbol in tickers:
        symbol = str(raw_symbol).upper().strip()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)

        last_exc: Exception | None = None
        for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
            try:
                ticker = yf.Ticker(symbol)
                items = getattr(ticker, "news", None)
                if callable(items):  # 일부 버전에서는 메서드로 제공됨
                    items = items()
                if not items:
                    items = ticker.get_news() if hasattr(ticker, "get_news") else []
                prepared = [
                    entry
                    for item in items
                    if isinstance(item, dict)
                    and (entry := _prepare_news_entry(item)) is not None
                ]

                cutoff = datetime.now(timezone.utc) - timedelta(days=2)
                recent = [
                    entry
                    for entry in prepared
                    if entry["timestamp"] is not None
                    and entry["timestamp"] >= cutoff
                ]

                if not recent:
                    # fallback to latest entries even if older than cutoff
                    recent = prepared[:1]

                recent.sort(
                    key=lambda entry: entry["timestamp"]
                    or datetime(1970, 1, 1, tzinfo=timezone.utc),
                    reverse=True,
                )

                top_entries = recent[:max_items]
                for entry in top_entries:
                    entry["text"] = _translate_text(entry.get("text", ""))

                news_map[symbol] = _format_news_entries(top_entries)
                break
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_DOWNLOAD_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
        else:
            if last_exc:
                print(f"[fetch_latest_news] {symbol}: {last_exc}")
            news_map[symbol] = "뉴스 조회 실패"

    return news_map
