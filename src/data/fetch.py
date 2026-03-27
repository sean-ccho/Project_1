"""시세 및 부가 정보 수집 유틸리티."""

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

MAX_DOWNLOAD_RETRIES = 3
RETRY_DELAY_SECONDS = 3
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


def fetch_latest_news(tickers: List[str]) -> Dict[str, str]:
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

                top_entries = recent[:1]
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
