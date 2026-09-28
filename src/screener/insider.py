"""openinsider.com 기반 내부자 거래(매수/매도) 요약 수집.

표시용 전용 모듈. 스코어링·매매 로직에는 영향을 주지 않는다.
스크리너가 골라낸 후보 티커에 대해서만 openinsider를 조회해 한 줄 요약
(`내부자(90일)`)을 만들어 이메일/차트분석 시트에 표시한다.

정적 HTML이라 Playwright 없이 urllib + BeautifulSoup(html.parser)로 충분하다.
"""

from __future__ import annotations

import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup

from . import config

# 내부자 데이터 디스크 캐시 (티커 단위, 24시간)
_INSIDER_CACHE_DIR = Path("data/cache")
_INSIDER_CACHE_MAX_AGE_HOURS = 24

_DISPLAY_COLUMN = "내부자(90일)"
_USER_AGENT = "Mozilla/5.0 (compatible; StockScreener/1.0)"

# fetch_insider_snapshots가 반환하는 컬럼
_RESULT_COLUMNS = [
    "티커",
    "insider_buy_count",
    "insider_sell_count",
    "insider_net_value",
    "insider_last_trade_date",
    _DISPLAY_COLUMN,
]


def _cache_path(ticker: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", ticker.upper())
    return _INSIDER_CACHE_DIR / f"insider_{safe}.parquet"


def _cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
    return age_hours < _INSIDER_CACHE_MAX_AGE_HOURS


def _parse_value(text: str) -> float:
    """'$1,234,567' / '-$1,234' 형태를 float로 변환. 실패 시 0.0."""
    cleaned = re.sub(r"[^0-9.\-]", "", str(text).replace("+", ""))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _format_money(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    amount = abs(value)
    if amount >= 1e6:
        return f"{sign}${amount / 1e6:.1f}M"
    if amount >= 1e3:
        return f"{sign}${amount / 1e3:.0f}K"
    return f"{sign}${amount:.0f}"


def _fetch_rows(ticker: str) -> List[List[str]]:
    """openinsider screener 페이지에서 거래 행(셀 리스트)들을 파싱해 반환."""
    lookback = getattr(config, "INSIDER_LOOKBACK_DAYS", 90)
    url = (
        f"http://openinsider.com/screener?s={ticker}"
        f"&fd={lookback}&xp=1&xs=1&cnt=100&sortcol=0&page=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", "ignore")

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="tinytable")
    rows: List[List[str]] = []
    if table is None or table.find("tbody") is None:
        return rows
    for tr in table.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) >= 12:
            rows.append(cells)
    return rows


def _summarize(ticker: str) -> Dict[str, object]:
    """단일 티커의 내부자 거래를 집계해 요약 dict를 만든다."""
    cluster_min = getattr(config, "INSIDER_CLUSTER_MIN_BUYERS", 2)

    rows = _fetch_rows(ticker)

    buy_count = sell_count = 0
    net_value = 0.0
    buyers: set[str] = set()
    last_trade_date = ""

    for cells in rows:
        # 컬럼: [X, Filing, Trade Date(2), Ticker, Insider(4), Title, Type(6), Price, Qty, Owned, ΔOwn, Value(11), ...]
        trade_date = cells[2]
        insider = cells[4]
        trade_type = cells[6].strip().upper()
        value = _parse_value(cells[11])

        if trade_type.startswith("P"):
            buy_count += 1
            net_value += abs(value)
            buyers.add(insider)
        elif trade_type.startswith("S"):
            sell_count += 1
            net_value -= abs(value)

        if trade_date > last_trade_date:
            last_trade_date = trade_date

    return {
        "티커": ticker,
        "insider_buy_count": buy_count,
        "insider_sell_count": sell_count,
        "insider_net_value": net_value,
        "insider_last_trade_date": last_trade_date,
        _DISPLAY_COLUMN: _build_summary_text(
            buy_count, sell_count, net_value, len(buyers), last_trade_date, cluster_min
        ),
    }


def _build_summary_text(
    buy_count: int,
    sell_count: int,
    net_value: float,
    distinct_buyers: int,
    last_trade_date: str,
    cluster_min: int,
) -> str:
    """'🟢 클러스터매수 · 매수 5/매도 0 · 순 +$860K (~05-21)' 형태의 한 줄 요약."""
    if buy_count == 0 and sell_count == 0:
        return "—"

    if distinct_buyers >= cluster_min:
        signal = "🟢 클러스터매수"
    elif buy_count > 0:
        signal = "🟢 매수"
    else:
        signal = "🔴 매도"

    last = last_trade_date[5:] if len(last_trade_date) >= 10 else (last_trade_date or "?")
    return (
        f"{signal} · 매수 {buy_count}/매도 {sell_count} · "
        f"순 {_format_money(net_value)} (~{last})"
    )


def _empty_record(ticker: str) -> Dict[str, object]:
    return {
        "티커": ticker,
        "insider_buy_count": 0,
        "insider_sell_count": 0,
        "insider_net_value": 0.0,
        "insider_last_trade_date": "",
        _DISPLAY_COLUMN: "—",
    }


def fetch_insider_snapshots(tickers: List[str], use_cache: bool = True) -> pd.DataFrame:
    """openinsider.com에서 티커별 내부자 매수/매도를 수집·집계해 반환한다.

    티커 단위 24시간 디스크 캐시를 사용하므로(use_cache=True) 이메일·SP500·NASDAQ
    여러 경로에서 겹치는 티커는 재조회하지 않는다. 개별 티커 조회가 실패하면 해당
    티커는 빈 레코드(—)로 채우고 계속 진행한다.
    """
    delay = getattr(config, "INSIDER_REQUEST_DELAY_SEC", 0.4)
    seen: set[str] = set()
    records: List[Dict[str, object]] = []

    for raw in tickers:
        ticker = str(raw).upper().strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)

        cache_path = _cache_path(ticker)
        if use_cache and _cache_valid(cache_path):
            try:
                cached = pd.read_parquet(cache_path)
                if not cached.empty:
                    records.append(cached.iloc[0].to_dict())
                    continue
            except Exception:
                pass  # 캐시 손상 시 새로 조회

        try:
            record = _summarize(ticker)
            time.sleep(delay)
        except Exception as exc:  # 네트워크/파싱 실패는 무시하고 빈 레코드
            print(f"[Insider] {ticker} 조회 실패 (무시): {exc}")
            record = _empty_record(ticker)

        records.append(record)

        if use_cache:
            try:
                _INSIDER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([record]).to_parquet(cache_path, index=False)
            except Exception:
                pass

    if not records:
        return pd.DataFrame(columns=_RESULT_COLUMNS)
    return pd.DataFrame(records, columns=_RESULT_COLUMNS)


def attach_insider_summary(df: pd.DataFrame) -> pd.DataFrame:
    """df['티커']에 내부자 요약 컬럼('내부자(90일)')을 머지해 반환한다.

    INSIDER_ENABLED=False면 조회 없이 빈 컬럼만 추가한다. 표시용 컬럼만 머지하므로
    기존 점수/매매 컬럼에는 영향이 없다.
    """
    if df is None or df.empty or "티커" not in df.columns:
        return df

    if not getattr(config, "INSIDER_ENABLED", True):
        if _DISPLAY_COLUMN not in df.columns:
            df = df.copy()
            df[_DISPLAY_COLUMN] = "—"
        return df

    tickers = df["티커"].dropna().astype(str).str.upper().str.strip().unique().tolist()
    insider = fetch_insider_snapshots(tickers)
    if insider.empty:
        out = df.copy()
        out[_DISPLAY_COLUMN] = "—"
        return out

    summary = insider[["티커", _DISPLAY_COLUMN]].copy()
    out = df.copy()
    # 머지 키 정규화
    out["_insider_key"] = out["티커"].astype(str).str.upper().str.strip()
    summary["_insider_key"] = summary["티커"].astype(str).str.upper().str.strip()
    summary = summary.drop(columns=["티커"])

    out = out.merge(summary, on="_insider_key", how="left")
    out = out.drop(columns=["_insider_key"])
    out[_DISPLAY_COLUMN] = out[_DISPLAY_COLUMN].fillna("—")
    return out
