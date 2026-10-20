"""전체 분석(src/run_universe_scan.py) → 웹 JSON. 표시 전용 — 매매 판단·상태 파일과 무관하다.

웹사이트 '전체분석' 페이지는 이 파일을 서버가 아니라 브라우저가 GitHub raw(universe-data 브랜치)에서 직접 받는다.
나스닥·NYSE 전체는 종목당 약 2KB × 5천 = 약 10MB 라 Vercel 함수 응답 한도(4.5MB)를 넘고, raw 는 gzip(약 7배)·CORS 를 준다.
- {key}.json         시장 목록 — 종목 정보는 dashboard.json 의 stocks 와 같은 모양(TickerInfo) + tv(TradingView 심볼)
- stocks/{티커}.json  종목 상세 페이지용 한 종목 (dashboard.json 에 없는 종목일 때 사이트가 읽는다)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from web_export.dashboard import _num, _text, ticker_info

SCHEMA_VERSION = 1
ET = ZoneInfo("America/Toronto")
INDEX_TICKERS = ("SPY", "QQQ", "IWM")  # 시장 레짐 판단용으로 분석은 하지만 목록에는 안 보인다 (전체차트와 같다)

ISB_NAMES = {"rev": "BUY·REV", "bottom": "◇BOTTOM", "top": "◆TOP", "exit": "EXIT"}
SQZ_DOTS = {"black": "●", "gray": "○", "blue": "·"}
SQZ_COLORS = {"lime": "연두", "green": "초록", "red": "빨강", "maroon": "적갈"}
TF_KO = {"D": "일", "W": "주", "M": "월"}


def isb_text(rec: dict[str, Any] | None) -> str:
    """ISB 최근 라벨 한 줄 (시트) — '일 BUY·REV 10/14 · 주 ◇BOTTOM 9/26'. 봉마다 가장 최근 라벨 하나."""
    parts = []
    for tf in ("D", "W"):
        labels = (rec or {}).get(tf) or {}
        hits = [(v.get("ago", 10**6), name, v.get("date", "")) for name, v in labels.items() if isinstance(v, dict)]
        if hits:
            _ago, name, date = min(hits)
            md = f"{int(date[5:7])}/{int(date[8:10])}" if len(date) >= 10 else date
            parts.append(f"{TF_KO[tf]} {ISB_NAMES.get(name, name)} {md}")
    return " · ".join(parts)


def sqz_text(rec: dict[str, Any] | None) -> str:
    """SQZ 모멘텀 한 줄 (시트) — '일 ●5 연두 · 주 ○ 초록 · 월 · 빨강'. ● = 스퀴즈 중(n봉째), ○ = 풀림, · = 둘 다 아님."""
    parts = []
    for tf in ("D", "W", "M"):
        st = (rec or {}).get(tf)
        if not st:
            continue
        dot = SQZ_DOTS.get(st.get("dot", ""), "?")
        if st.get("dot") == "black" and st.get("run"):
            dot += str(st["run"])
        color = SQZ_COLORS.get(st.get("color", ""), "")
        parts.append(" ".join(p for p in (TF_KO[tf], dot, color) if p))
    return " · ".join(parts)


def item_info(row: pd.Series, *, tv: str | None, insider: dict[str, Any] | None, isb: dict[str, Any] | None,
              sqz: dict[str, Any] | None, golden: dict[str, Any] | None = None) -> dict[str, Any]:
    """분석 한 줄 → 웹 TickerInfo (ticker_info 와 같은 모양) + tv · 신규 상장 칸."""
    info = ticker_info(row)
    info.update({"tv": tv, "insider": insider, "isb": isb, "sqzmom": sqz, "golden": golden})
    first = _text(row.get("첫거래일"))
    if first:
        info["first_date"] = first
        days = _num(row.get("상장후거래일"))
        info["listed_days"] = int(days) if days is not None else None
        info["ret_since_listing"] = _num(row.get("상장후수익률"), 4)
    return info


def market_payload(key: str, label: str, items: dict[str, dict[str, Any]], *, bar_date: str | None,
                   counts: dict[str, int], golden: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """시장 한 개 JSON. items 는 순서 그대로 (매수적합도 순 · 신규 상장은 최근 상장 순)."""
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "bar_date": bar_date,
        "key": key,
        "label": label,
        "counts": counts,
        "golden": golden or [],
        "items": {t: v for t, v in items.items() if t not in INDEX_TICKERS},
    }


def write_payloads(out_dir: Path, payloads: list[dict[str, Any]]) -> dict[str, int]:
    """{key}.json 과 stocks/{티커}.json 을 쓴다 → {파일: 바이트}. 같은 티커가 두 시장에 있으면 앞의 것."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stocks_dir = out_dir / "stocks"
    stocks_dir.mkdir(exist_ok=True)
    sizes: dict[str, int] = {}
    written: set[str] = set()
    for p in payloads:
        body = json.dumps(p, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        (out_dir / f"{p['key']}.json").write_text(body, encoding="utf-8")
        sizes[f"{p['key']}.json"] = len(body.encode())
        for ticker, info in p["items"].items():
            if ticker in written:
                continue
            written.add(ticker)
            one = {"schema": SCHEMA_VERSION, "bar_date": p["bar_date"], "market": p["key"], "ticker": ticker, "info": info}
            (stocks_dir / f"{ticker}.json").write_text(
                json.dumps(one, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    sizes["stocks/"] = len(written)
    return sizes
