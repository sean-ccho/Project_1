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


def has_fresh_cache(ticker: str) -> bool:
    """24시간 안에 받아 둔 캐시가 있는지 (네트워크 없이 확인)."""
    return _cache_valid(_cache_path(str(ticker).strip()))


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
    return _parse_rows(_fetch_html(url))


def _fetch_html(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "ignore")


def _parse_rows(html: str) -> List[List[str]]:
    """screener 표(tinytable)의 거래 행(셀 리스트)들 — 종목별 표 칸 순서 [X, Filing, Trade Date, Ticker, Insider, ...].

    티커 없이 받은 시장 전체 표는 Ticker 뒤에 'Company Name' 칸이 하나 더 있어서, 머리글에서 찾아 빼고 같은 순서로 맞춘다.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="tinytable")
    rows: List[List[str]] = []
    if table is None or table.find("tbody") is None:
        return rows
    header = [th.get_text(strip=True).replace("\xa0", " ") for th in table.find_all("th")]
    company = next((i for i, h in enumerate(header) if h.startswith("Company")), None)
    for tr in table.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if company is not None and len(cells) > company:
            del cells[company]
        if len(cells) >= 12:
            rows.append(cells)
    return rows


def _summarize(ticker: str) -> Dict[str, object]:
    """단일 티커의 내부자 거래를 집계해 요약 dict를 만든다."""
    return _aggregate(ticker, _fetch_rows(ticker))


def _aggregate(ticker: str, rows: List[List[str]]) -> Dict[str, object]:
    """거래 행들 → 요약 dict (종목별 조회·시장 전체 조회 공용)."""
    cluster_min = getattr(config, "INSIDER_CLUSTER_MIN_BUYERS", 2)

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


def fetch_insider_market(
    days: Optional[int] = None,
    *,
    page_size: int = 5000,
    max_pages: int = 20,
    fetch_html=None,
) -> pd.DataFrame:
    """시장 전체 최근 N일 내부자 매수·매도를 한 번에 받아 티커별로 집계한다 (fetch_insider_snapshots 와 같은 컬럼).

    종목마다 한 페이지씩 받으면 5천 종목에 2시간쯤 걸려서, 전체 분석(run_universe_scan.py)은 티커 없이 받는다
    (5,000건 한 페이지 ≈ 5초). 종목별 조회는 거래 100건까지만 받아서, 그보다 많은 종목은 숫자가 조금 다를 수 있다.
    티커는 야후 형식(BRK.B → BRK-B). 한 페이지라도 실패하면 빈 표 — 일부만 집계하면 숫자가 틀리므로.
    """
    lookback = days or getattr(config, "INSIDER_LOOKBACK_DAYS", 90)
    delay = getattr(config, "INSIDER_REQUEST_DELAY_SEC", 0.4)
    fetch_html = fetch_html or _fetch_html
    by_ticker: Dict[str, List[List[str]]] = {}
    try:
        for page in range(1, max_pages + 1):
            url = (
                f"http://openinsider.com/screener?s=&fd={lookback}&xp=1&xs=1"
                f"&cnt={page_size}&sortcol=0&page={page}"
            )
            rows = _parse_rows(fetch_html(url, timeout=90))
            for cells in rows:
                ticker = cells[3].strip().upper().replace(".", "-")
                if ticker:
                    by_ticker.setdefault(ticker, []).append(cells)
            if len(rows) < page_size:
                break
            time.sleep(delay)
        else:
            print(f"[Insider] 시장 전체 {max_pages}페이지 한도 — 그 뒤 거래는 빠짐")
    except Exception as exc:  # noqa: BLE001 — 표시용
        print(f"[Insider] 시장 전체 조회 실패 (내부자 칸 비움): {exc}")
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    records = [_aggregate(t, rows) for t, rows in by_ticker.items()]
    n_rows = sum(len(r) for r in by_ticker.values())
    print(f"[Insider] 시장 전체 {lookback}일: 거래 {n_rows:,}건 · {len(records):,}종목")
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
