"""주요 일정 (data.event_calendar, 2026-11-06) — 규칙 · 파서 · 평소 대비 변동 · 모으기. 네트워크 없이 가짜 자료로."""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import event_calendar as ec  # noqa: E402
from screener.macro_scan import event_rows, sheet_frame  # noqa: E402

FRED_HTML = """
<tr class="odd"><td colspan="2"><span style="font-weight: bold;">Wednesday October 14, 2026</span>
 <span style="float: right; font-weight: bold;">Updated</span></td></tr>
<tr><td nowrap style="width:5%; text-align:right">
          7:30 am      </td><td><a href="/release?rid=10">Consumer Price Index</a></td></tr>
<tr class="odd"><td colspan="2"><span style="font-weight: bold;">Tuesday November 10, 2026</span>     </td></tr>
<tr><td nowrap>   7:30 am   </td><td><a href="/release?rid=10">Consumer Price Index</a></td></tr>
<tr class="odd"><td colspan="2"><span style="font-weight: bold;">Friday January 15, 2027</span>     </td></tr>
<tr><td nowrap>   1:00 pm   </td><td><a href="/release?rid=10">Consumer Price Index</a></td></tr>
"""

STATCAN_JSON = json.dumps([
    {"date": "2026-11-06 00:00:01", "type": "meeting", "title": "Labour Force Survey", "description": "October 2026"},
    {"date": "2026-11-16 00:00:01", "type": "meeting", "title": "Consumer Price Index", "description": "October 2026"},
    {"date": "2026-11-16 00:00:02", "type": "meeting", "title": "Retail trade", "description": "September 2026"},
    {"date": "bad", "type": "meeting", "title": "Consumer Price Index", "description": "x"},
])


# ── 날짜 규칙

def test_third_friday_opex_and_holiday_shift() -> None:
    assert ec.third_friday(2026, 11) == date(2026, 11, 20)
    closed = ec.nyse_closed()
    days = ec.opex_days(date(2026, 3, 1), date(2026, 6, 30), lambda d: d.weekday() < 5 and d not in closed)
    assert days == [(date(2026, 3, 20), True), (date(2026, 4, 17), False), (date(2026, 5, 15), False),
                    (date(2026, 6, 18), True)]          # 6-19 준틴스 휴장 → 목요일
    assert ec.opex_days(date(2027, 3, 1), date(2027, 3, 31), lambda d: d.weekday() < 5 and d not in closed) == [
        (date(2027, 3, 19), True)]


def test_us_election_days() -> None:
    assert ec.us_election_day(2026) == date(2026, 11, 3)
    assert ec.us_election_day(2028) == date(2028, 11, 7) and ec.us_election_day(2030) == date(2030, 11, 5)
    days = ec.us_election_days(1994, 2026)
    assert days[0] == date(1994, 11, 8) and days[-1] == date(2026, 11, 3) and len(days) == 17
    assert ec.us_election_days(2027, 2028) == [date(2028, 11, 7)]


# ── 파서

def test_parse_fred_calendar_central_to_eastern_and_year_filter() -> None:
    got = ec.parse_fred_calendar(FRED_HTML, 2026)
    assert got == [(date(2026, 10, 14), "08:30"), (date(2026, 11, 10), "08:30")]   # 'Updated' 줄도 같은 날짜
    assert ec.parse_fred_calendar(FRED_HTML, 2027) == [(date(2027, 1, 15), "14:00")]
    assert ec.parse_fred_calendar("<html>nothing</html>", 2026) == []


def test_parse_statcan_and_period() -> None:
    sc = ec.parse_statcan(STATCAN_JSON)
    assert sc["ca_cpi"] == [(date(2026, 11, 16), "October 2026")]
    assert sc["ca_jobs"] == [(date(2026, 11, 6), "October 2026")]
    assert ec.period_ko("October 2026") == "2026년 10월분" and ec.period_ko("Q3 2026") == "Q3 2026"
    assert ec.parse_fred_api_dates('{"release_dates":[{"release_id":10,"date":"2025-01-15"},{"date":"2024-12-11"}]}') == [
        date(2024, 12, 11), date(2025, 1, 15)]


# ── 평소 대비 변동

def _close(moves: dict[str, float], n: int = 300, start: str = "2025-01-01") -> pd.Series:
    """평소 ±0.5% 번갈아, moves 의 날짜만 그 크기로 움직이는 종가."""
    idx = pd.bdate_range(start, periods=n)
    r = pd.Series([0.005 if i % 2 else -0.005 for i in range(n)], index=idx)
    for d, m in moves.items():
        r.loc[pd.Timestamp(d)] = m
    return 100 * (1 + r).cumprod()


def test_impact_ratio_next_day_and_min_n() -> None:
    big = {d.date().isoformat(): 0.02 for d in pd.bdate_range("2025-02-03", periods=10, freq="5B")}
    close = _close(big)
    days = [date.fromisoformat(d) for d in big]
    imp = ec.impact(days, close)
    assert imp["n"] == 10 and imp["since"] == 2025 and imp["x"] > 2.5
    # 다음 거래일로 재면 큰 날을 비켜 간다 → 평소와 비슷
    assert ec.impact(days, close, next_day=True)["x"] == pytest.approx(1.0, abs=0.35)
    assert ec.impact(days[:5], close) is None                         # 8번 미만
    assert ec.impact([date(2025, 2, 8)] * 9, close) is None            # 장이 안 연 날(토요일)만
    assert ec.impact(days, None) is None


# ── 모으기

def _prices() -> pd.DataFrame:
    idx = pd.bdate_range("1999-01-04", "2026-11-05")
    rng = np.random.default_rng(0)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    xic = pd.Series(30 * np.exp(np.cumsum(rng.normal(0, 0.008, len(idx)))), index=idx)
    return pd.DataFrame({"SPY": spy, "XIC.TO": xic})


def test_build_events_window_sort_impact_and_errors() -> None:
    today = date(2026, 11, 6)
    fomc = [("2026-10-27", "2026-10-28", False), ("2026-12-08", "2026-12-09", True)]
    boc = [("2026-10-28", True), ("2026-12-09", False)]

    def get(url: str) -> bytes:
        if "fred.stlouisfed.org" in url and "rid=10" in url:
            return FRED_HTML.encode()
        if "statcan" in url:
            return STATCAN_JSON.encode()
        raise RuntimeError("blocked")

    errors: list[str] = []
    ev = ec.build_events(today, fomc, boc, get=get, prices=_prices(), errors=errors)
    dates = [e["date"] for e in ev]
    assert dates == sorted(dates) and dates[0] >= "2026-10-30"           # 7일 전부터
    by_key: dict = {}
    for e in ev:
        by_key.setdefault(e["key"], e)                                    # 키마다 가장 가까운 일정
    assert by_key["cpi"]["time"] == "08:30" and by_key["cpi"]["tier"] == 3 and by_key["cpi"]["impact"] is None  # 키 없음
    assert {e["date"] for e in ev if e["key"] == "cpi"} == {"2026-11-10", "2027-01-15"}
    assert by_key["ca_cpi"]["note_ko"] == "2026년 10월분" and by_key["ca_cpi"]["country"] == "CA"
    fomc_dec = [e for e in ev if e["key"] == "fomc"]
    assert [e["date"] for e in fomc_dec] == ["2026-12-09"] and fomc_dec[0]["note_ko"].startswith("경제 전망")
    assert [e["date"] for e in ev if e["key"] == "fomc_minutes"] == ["2026-11-18", "2026-12-30"]   # 결정 21일 뒤
    assert any(e["key"] == "quad" and e["date"] == "2026-12-18" and "나스닥100" in e["note_ko"] for e in ev)
    assert any(e["key"] == "holiday_us" and e["date"] == "2026-11-26" and "추수감사절" in e["ko"] for e in ev)
    assert any(e["key"] == "funding_us" and e["date"] == "2026-12-11" for e in ev)
    elections = [e for e in ev if e["key"] == "election_us"]
    assert [e["date"] for e in elections][:2] == ["2026-11-03", "2028-11-07"] and "대선" in elections[1]["ko"]
    assert elections[0]["impact"]["n"] == 14 and elections[0]["impact"]["since"] == 2000   # 가짜 가격은 1999년부터
    assert any(e["key"] == "election_ca" and e["date"] == "2029-10-15" for e in ev)
    assert by_key["opex"]["impact"]["n"] > 100                         # 2000년부터 SPY 가 연 셋째 금요일
    # 같은 날은 중요도 높은 것 먼저
    same = [e for e in ev if e["date"] == "2026-12-09"]
    assert [e["key"] for e in same] == ["boc", "fomc"]                 # ★3 둘 — 시각 순 (09:45 → 14:00)
    # 못 받은 FRED 지표는 경고만 남기고 나머지는 그대로
    assert any("FRED 달력 jobs" in x for x in errors) and not any("StatCan" in x for x in errors)
    json.dumps(ev, allow_nan=False)


def test_build_events_without_network_or_prices_still_lists_rules() -> None:
    errors: list[str] = []
    ev = ec.build_events(date(2026, 11, 6), [], [], errors=errors)
    keys = {e["key"] for e in ev}
    assert {"opex", "quad", "holiday_us", "holiday_ca", "election_us", "funding_us"} <= keys
    assert all(e["impact"] is None for e in ev) and errors == []


def test_holiday_list_warns_when_next_year_missing() -> None:
    errors: list[str] = []
    ec.build_events(date(2027, 11, 6), [], [], errors=errors)
    assert any("TSX 휴장 목록에 2028년이 없음" in x for x in errors) and not any("NYSE" in x for x in errors)


# ── 시트 [경제]

def test_event_rows_in_sheet() -> None:
    events = [
        {"date": "2026-11-05", "time": "08:30", "key": "ppi", "kind": "data", "country": "US", "tier": 1, "ko": "어제",
         "note_ko": "", "impact": None},
        {"date": "2026-11-10", "time": "08:30", "key": "cpi", "kind": "data", "country": "US", "tier": 3,
         "ko": "미국 CPI (소비자물가)", "note_ko": "", "impact": {"x": 1.42, "n": 120, "since": 2016}},
        {"date": "2026-12-09", "time": "09:45", "key": "boc", "kind": "rate", "country": "CA", "tier": 3,
         "ko": "BoC 금리 결정", "note_ko": "", "impact": None},
        {"date": "2027-03-01", "time": None, "key": "x", "kind": "policy", "country": "US", "tier": 3, "ko": "먼 일정",
         "note_ko": "", "impact": None},
    ]
    rows = event_rows(events, date(2026, 11, 6))
    assert [r["항목"] for r in rows] == ["미국 CPI (소비자물가)", "BoC 금리 결정"]   # 오늘부터 60일
    assert rows[0]["값"] == "08:30 ET" and rows[0]["비교"] == "★★★" and rows[0]["판정"] == "평소의 1.4배 (120번, 2016년부터)"
    assert rows[1]["설명"] == "캐나다"
    f = sheet_frame({"as_of": "2026-11-06", "indicators": [], "rates": {}, "events": events})
    assert list(f["구분"]) == ["", "주요 일정", "주요 일정"]
