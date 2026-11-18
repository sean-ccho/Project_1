"""scripts/launchagent/yahoo_bar_ready.py — 야간 작업 전 Yahoo 미국 종가 확인 (2026-11-17)."""
from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "launchagent" / "yahoo_bar_ready.py"
_spec = importlib.util.spec_from_file_location("yahoo_bar_ready", _PATH)
ybr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ybr)

NY = ZoneInfo("America/New_York")


def _ts(day: str, hh: int, mm: int = 0) -> int:
    y, m, d = map(int, day.split("-"))
    return int(datetime(y, m, d, hh, mm, tzinfo=NY).timestamp())


def _payload(days: list[str], closes: list[Any], market_day: str | None) -> dict[str, Any]:
    """Yahoo v8 chart 응답 모양 — 일봉 시각은 09:30, 마지막 정규장 시각은 16:00 (ET)."""
    meta: dict[str, Any] = {"exchangeTimezoneName": "America/New_York"}
    if market_day:
        meta["regularMarketTime"] = _ts(market_day, 16)
    return {"chart": {"result": [{
        "meta": meta,
        "timestamp": [_ts(d, 9, 30) for d in days],
        "indicators": {"quote": [{"close": closes, "volume": [1000] * len(days)}]},
    }]}}


DAYS = ["2026-11-13", "2026-11-16", "2026-11-17"]


def test_judge_ok_when_latest_close_present() -> None:
    status, note = ybr.judge(_payload(DAYS, [754.37, 739.95, 732.44], "2026-11-17"))
    assert status == "ok"
    assert note.startswith("2026-11-17")


def test_judge_bad_when_latest_close_empty() -> None:
    """11-17 20:00~21:30 ET 실제 모양 — 그날 줄은 있는데 종가가 빈 칸."""
    status, note = ybr.judge(_payload(DAYS, [754.37, 739.95, None], "2026-11-17"))
    assert status == "bad"
    assert "빈 칸" in note


def test_judge_bad_when_latest_session_row_missing() -> None:
    """마지막 정규장(11-17) 줄이 아직 없으면 하루 늦은 자료 — 기다린다."""
    status, note = ybr.judge(_payload(DAYS[:2], [754.37, 739.95], "2026-11-17"))
    assert status == "bad"
    assert "2026-11-17" in note


def test_judge_ok_on_weekend_or_holiday() -> None:
    """토요일 밤: 마지막 정규장 = 금요일이면 금요일 줄로 정상."""
    days = ["2026-11-11", "2026-11-12", "2026-11-13"]
    assert ybr.judge(_payload(days, [1.0, 2.0, 3.0], "2026-11-13"))[0] == "ok"


def test_judge_without_market_time_uses_close_only() -> None:
    assert ybr.judge(_payload(DAYS, [1.0, 2.0, 3.0], None))[0] == "ok"
    assert ybr.judge(_payload(DAYS, [1.0, 2.0, None], None))[0] == "bad"


@pytest.mark.parametrize("payload", [
    {},
    {"chart": {"result": []}},
    {"chart": {"result": None}},
    {"chart": {"result": [{"meta": {}, "timestamp": [], "indicators": {"quote": [{"close": []}]}}]}},
    {"chart": {"result": [{"meta": {}, "timestamp": [1, 2], "indicators": {"quote": [{"close": [1.0]}]}}]}},
])
def test_judge_err_on_malformed(payload: dict[str, Any]) -> None:
    assert ybr.judge(payload)[0] == "err"


def test_overall_thresholds() -> None:
    assert ybr.overall(["ok"] * 12) == "OK"
    assert ybr.overall(["ok"] * 10 + ["bad"] * 2) == "OK"       # 한두 종목 이상해도 막히지 않음
    assert ybr.overall(["bad"] * 12) == "WAIT"                  # 빈 칸 시간
    assert ybr.overall(["ok"] * 9 + ["bad"] * 3) == "WAIT"
    assert ybr.overall(["ok"] * 9 + ["bad"] + ["err"] * 2) == "ERR"
    assert ybr.overall(["err"] * 12) == "ERR"                   # 못 받음 → 확인 못 함


def test_check_and_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    good = _payload(DAYS, [1.0, 2.0, 3.0], "2026-11-17")
    empty = _payload(DAYS, [1.0, 2.0, None], "2026-11-17")

    monkeypatch.setattr(ybr.time, "sleep", lambda s: None)
    monkeypatch.setattr(ybr, "fetch", lambda sym, timeout=15.0: good)
    verdict, rows = ybr.check(pause=0)
    assert verdict == "OK" and len(rows) == len(ybr.SYMBOLS)
    assert ybr.main() == 0

    monkeypatch.setattr(ybr, "fetch", lambda sym, timeout=15.0: empty)
    assert ybr.check(pause=0)[0] == "WAIT"

    def boom(sym: str, timeout: float = 15.0) -> dict[str, Any]:
        raise OSError("network down")

    monkeypatch.setattr(ybr, "fetch", boom)
    verdict, rows = ybr.check(pause=0)
    assert verdict == "ERR"
    assert rows[0] == ("SPY", "err", "OSError")


def test_summary_line() -> None:
    rows = [("SPY", "bad", "2026-11-17 종가 빈 칸")] * 6 + [("AAPL", "ok", "2026-11-17 1.00")] * 6
    line = ybr.summary("WAIT", rows)
    assert line.startswith("WAIT 정상 6/12")
    assert "외 2" in line
