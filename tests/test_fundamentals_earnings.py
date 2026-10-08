"""yfinance 실적일 추출 테스트 (네트워크 없음)."""

import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screener.fundamentals import _extract_earnings_date  # noqa: E402

TODAY = date(2026, 10, 8)


def _epoch(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="America/New_York").timestamp())


def test_reads_yfinance_1x_epoch_keys() -> None:
    # yfinance 1.x: earningsTimestamp* (epoch 초). 예전 코드는 earningsDate 만 찾아 항상 빈 값이었다
    info = {"earningsTimestamp": _epoch("2026-10-21 07:30"), "earningsTimestampStart": _epoch("2026-10-21 07:30")}
    assert _extract_earnings_date(info, today=TODAY) == ("2026-10-21", 13.0)


def test_after_close_epoch_uses_et_date_not_utc() -> None:
    # 2026-10-20 16:05 ET = 2026-10-20 20:05 UTC (같은 날) / 2026-10-20 21:00 ET = 10-21 01:00 UTC → ET 날짜 10-20
    info = {"earningsTimestamp": _epoch("2026-10-20 21:00")}
    assert _extract_earnings_date(info, today=TODAY) == ("2026-10-20", 12.0)


def test_picks_earliest_upcoming_and_ignores_past() -> None:
    info = {"earningsTimestamp": _epoch("2026-07-21 07:30"),          # 지난 분기 (갱신 전)
            "earningsTimestampStart": _epoch("2026-10-27 07:30")}
    assert _extract_earnings_date(info, today=TODAY) == ("2026-10-27", 19.0)
    only_past = {"earningsTimestamp": _epoch("2026-07-21 07:30")}
    d, n = _extract_earnings_date(only_past, today=TODAY)
    assert d == "" and math.isnan(n)


def test_today_counts_as_zero_days_and_legacy_keys_still_work() -> None:
    assert _extract_earnings_date({"earningsTimestamp": _epoch("2026-10-08 16:30")}, today=TODAY)[1] == 0.0
    assert _extract_earnings_date({"earningsDate": ["2026-10-10"]}, today=TODAY) == ("2026-10-10", 2.0)


def test_missing_or_garbage_is_nan() -> None:
    for info in ({}, {"earningsTimestamp": None}, {"earningsTimestamp": 0}, {"earningsDate": "not a date"}):
        d, n = _extract_earnings_date(info, today=TODAY)
        assert d == "" and math.isnan(n)
