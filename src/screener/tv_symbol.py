"""야후 티커 → TradingView 심볼.

TradingView 는 'BRK-B' 를 런던(LSE) 종목으로 연다 — 2026-10-16 캡처에서 BRK-B 차트가 '1,440' 짜리 런던 종목이었다.
종류주는 TradingView 에서 점으로 쓴다 (BRK.B · BF.B). 그 밖의 예외는 config.TRADINGVIEW_TICKER_MAP 에 둔다.
"""

from __future__ import annotations

import re

_SHARE_CLASS = re.compile(r"^[A-Z]+-[A-Z]$")


def tv_symbol(ticker: str, ticker_map: dict[str, str] | None = None) -> str:
    """지도에 있으면 그 값, 'BRK-B' 같은 종류주는 'BRK.B', 나머지는 그대로."""
    if ticker_map and ticker in ticker_map:
        return ticker_map[ticker]
    if _SHARE_CLASS.match(ticker):
        return ticker.replace("-", ".")
    return ticker
