#!/usr/bin/env python3
"""Yahoo 미국 일봉 확인 — 야간 작업을 보내기 전에 '가장 최근 거래일 종가'가 다 들어왔는지 본다 (2026-11-17).

왜: 평일 밤 20:00쯤 ~ 21:10~21:35 ET 에 Yahoo 는 그날 미국 일봉을 '거래량만 있고 가격은 빈 칸'으로 준다
(11-12 · 11-16 · 11-17 확인 — 캐나다 TSX 는 멀쩡). 이때 파이프라인이 돌면 그날 일봉으로 계산한 점수가 망가지거나
(PT-1 이 그 날을 '처리함'으로 남김) 하루 늦은 가격을 쓴다 (PT-SPY 건너뜀 · 내계좌 알림 전날 가격).

판정 — 종목마다 Yahoo v8 chart (5일 · 일봉):
  ok   가장 최근 줄에 종가가 있고, 그 줄 날짜가 마지막 정규장 날짜(meta.regularMarketTime)보다 이르지 않음
  bad  받았지만 가장 최근 줄 종가가 빈 칸이거나, 마지막 정규장 날짜의 줄이 아직 없음
  err  받지 못함 (네트워크 · 차단 · 형식 오류)
전체 — ok 가 NEED(12개 중 10개) 이상이면 OK(exit 0), bad 가 BAD_WAIT(3개) 이상이면 WAIT(exit 1, 아직), 그 밖은 ERR(exit 2, 확인 못 함).
주말 · 휴일은 마지막 정규장 날짜가 금요일 · 전날이라 그대로 OK 다.

표준 라이브러리만 쓴다 — Mac LaunchAgent 는 ~/Documents 의 가상환경을 못 읽어서 /usr/bin/python3(3.9)로 돈다.
원본: scripts/launchagent/ (install.sh 가 ~/account-watch 로 복사). nightly_dispatch.sh 와 GitHub 예비 실행 gate 가 같이 쓴다.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

SYMBOLS: tuple[str, ...] = ("SPY", "QQQ", "DIA", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "XOM", "BRK-B")
NEED = 10      # 이만큼 ok 면 정상 — 한두 종목의 이상한 일봉 때문에 밤 작업이 막히지 않게
BAD_WAIT = 3   # 이만큼 bad 면 '아직 안 들어옴' (빈 칸 시간에는 미국 종목 전부가 bad)
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=5d&interval=1d"
USER_AGENT = "Mozilla/5.0"
DEFAULT_TZ = "America/New_York"
EXIT_CODE = {"OK": 0, "WAIT": 1, "ERR": 2}


def judge(payload: Mapping[str, Any]) -> tuple[str, str]:
    """chart 응답 하나 → ('ok' | 'bad' | 'err', 짧은 설명)."""
    try:
        res = payload["chart"]["result"][0]
        meta = res.get("meta") or {}
        tz = ZoneInfo(meta.get("exchangeTimezoneName") or DEFAULT_TZ)
        stamps = res.get("timestamp") or []
        closes = ((res.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
        return "err", f"형식 오류 {type(e).__name__}"
    if not stamps or len(closes) != len(stamps):
        return "err", "일봉 없음"
    last_day = datetime.fromtimestamp(stamps[-1], tz).date()
    rmt = meta.get("regularMarketTime")
    market_day = datetime.fromtimestamp(rmt, tz).date() if isinstance(rmt, (int, float)) else None
    if closes[-1] is None:
        return "bad", f"{last_day} 종가 빈 칸"
    if market_day is not None and last_day < market_day:
        return "bad", f"{market_day} 일봉 없음 (마지막 {last_day})"
    return "ok", f"{last_day} {float(closes[-1]):.2f}"


def overall(statuses: Sequence[str], need: int = NEED, bad_wait: int = BAD_WAIT) -> str:
    """종목별 판정 목록 → 'OK' | 'WAIT' | 'ERR'."""
    if statuses.count("ok") >= need:
        return "OK"
    if statuses.count("bad") >= bad_wait:
        return "WAIT"
    return "ERR"


def fetch(sym: str, timeout: float = 15.0) -> Mapping[str, Any]:
    """Yahoo v8 chart 한 종목 (5일 · 일봉) JSON."""
    req = urllib.request.Request(URL.format(sym=sym), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def check(symbols: Sequence[str] = SYMBOLS, pause: float = 0.3) -> tuple[str, list[tuple[str, str, str]]]:
    """표본 종목을 차례로 받아 판정 → (전체 판정, [(종목, 판정, 설명), ...])."""
    rows: list[tuple[str, str, str]] = []
    for i, sym in enumerate(symbols):
        if i and pause:
            time.sleep(pause)
        try:
            status, note = judge(fetch(sym))
        except Exception as e:  # noqa: BLE001 — 어떤 이유로든 못 받으면 err
            status, note = "err", type(e).__name__
        rows.append((sym, status, note))
    return overall([s for _, s, _ in rows]), rows


def summary(verdict: str, rows: Sequence[tuple[str, str, str]]) -> str:
    """로그 한 줄 — 판정 · 정상 수 · 정상 아닌 종목 몇 개."""
    n_ok = sum(1 for _, s, _ in rows if s == "ok")
    odd = [f"{sym} {note}" for sym, s, note in rows if s != "ok"]
    sample = next((note for _, s, note in rows if s == "ok"), "")
    tail = (" · " + " · ".join(odd[:4]) + (f" 외 {len(odd) - 4}" if len(odd) > 4 else "")) if odd else ""
    return f"{verdict} 정상 {n_ok}/{len(rows)}" + (f" (예: {sample})" if sample else "") + tail


def main() -> int:
    verdict, rows = check()
    print(summary(verdict, rows))
    return EXIT_CODE[verdict]


if __name__ == "__main__":
    sys.exit(main())
