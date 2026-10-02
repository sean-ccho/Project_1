"""S&P 500 과거 구성종목 (point-in-time) — 생존 편향 완화.

현재 TICKERS로 과거를 돌리면 "지금까지 살아남은 회사"만 고르게 된다.
그날 실제 구성종목만 후보로 쓰면 이 편향이 줄어든다 (단, 상장폐지 종목은 yfinance에 가격이 없어 여전히 빠짐).

데이터: fja05680/sp500 "S&P 500 Historical Components & Changes" CSV (date,tickers).
scripts/fetch_sp500_membership.py로 받아 config.SP500_MEMBERSHIP_PATH에 둔다.
"""

from __future__ import annotations

import bisect
import csv
from pathlib import Path


def _normalize(ticker: str) -> str:
    """yfinance 표기로 (BRK.B → BRK-B)."""
    return ticker.strip().upper().replace(".", "-")


def load_membership(path: str | Path) -> list[tuple[str, frozenset[str]]]:
    """CSV → [(YYYY-MM-DD, 구성종목 집합)] 날짜 오름차순."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} 없음 → PYTHONPATH=.:src python scripts/fetch_sp500_membership.py 먼저 실행"
        )
    rows: list[tuple[str, frozenset[str]]] = []
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for rec in reader:
            if len(rec) < 2 or not rec[0].strip():
                continue
            tickers = frozenset(_normalize(t) for t in ",".join(rec[1:]).split(",") if t.strip())
            rows.append((rec[0].strip(), tickers))
    rows.sort(key=lambda r: r[0])
    return rows


class Membership:
    """날짜별 구성종목 조회."""

    def __init__(self, rows: list[tuple[str, frozenset[str]]]) -> None:
        self._dates = [d for d, _ in rows]
        self._sets = [s for _, s in rows]

    def on(self, date: str) -> frozenset[str]:
        """date(YYYY-MM-DD) 시점 구성종목. 첫 기록 이전이면 빈 집합."""
        i = bisect.bisect_right(self._dates, date) - 1
        return self._sets[i] if i >= 0 else frozenset()

    def union_between(self, start: str, end: str) -> set[str]:
        """기간 중 한 번이라도 구성종목이었던 티커 (start 시점 구성 포함)."""
        out = set(self.on(start))
        lo = bisect.bisect_right(self._dates, start)
        hi = bisect.bisect_right(self._dates, end)
        for s in self._sets[lo:hi]:
            out |= s
        return out
