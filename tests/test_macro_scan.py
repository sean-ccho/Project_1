"""경기 상태 · 금리 결정 (screener.macro_scan · data.rate_calendar) — 네트워크 없이 작은 가짜 데이터로."""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.rate_calendar import (BOC_ANNOUNCEMENTS, FOMC_MEETINGS, boc_url, fomc_url, merge_boc,  # noqa: E402
                                merge_fomc, parse_boc_page, parse_fomc_page)
from screener.macro_scan import (_clean, boc_decisions, fomc_decisions, ind_claims, ind_curve, ind_cycdef,  # noqa: E402
                                 ind_hy, ind_sahm, ind_spy200, parse_boc_valet, parse_fred_csv, sheet_frame)

FED_HTML = """
<h4><a id="1">2028 FOMC Meetings</a></h4>
<div class="fomc-meeting__month"><strong>January</strong></div><div class="fomc-meeting__date">25-26</div>
<div class="fomc-meeting__month"><strong>Apr/May</strong></div><div class="fomc-meeting__date">30-1*</div>
<div class="fomc-meeting__month"><strong>August</strong></div><div class="fomc-meeting__date">22 (notation vote)</div>
<h4><a id="2">2027 FOMC Meetings</a></h4>
<div class="fomc-meeting__month"><strong>March</strong></div><div class="fomc-meeting__date">16-17*</div>
"""

BOC_HTML = """<div><h3>Schedule for 2028</h3><table><tr><th>Dates</th><th>Publications</th></tr>
<tr><td>January 26</td><td>Interest rate announcement and Monetary Policy Report</td></tr>
<tr><td>March 8</td><td>Interest rate announcement</td></tr></table>
<p>See Blackout Guidelines</p></div>"""


def test_calendar_parsers_and_merge():
    fed = parse_fomc_page(FED_HTML)
    assert ("2028-01-25", "2028-01-26", False) in fed and ("2028-04-30", "2028-05-01", True) in fed
    assert ("2027-03-16", "2027-03-17", True) in fed and len(fed) == 3          # notation vote 는 뺀다
    merged = merge_fomc(fed)
    assert len(merged) == len(FOMC_MEETINGS) + 2 and merged[-1][1] == "2028-05-01"   # 2027-03-17 은 이미 있음
    boc = parse_boc_page(BOC_HTML)
    assert boc == [("2028-01-26", True), ("2028-03-08", False)]
    assert len(merge_boc(boc)) == len(BOC_ANNOUNCEMENTS) + 2
    assert all(len({d for _s, d, _x in FOMC_MEETINGS if d.startswith(str(y))}) == 8 for y in range(2021, 2028))
    assert all(len([d for d, _m in BOC_ANNOUNCEMENTS if d.startswith(str(y))]) == 8 for y in range(2021, 2028))
    assert fomc_url("2025-10-29").endswith("monetary20251029a.htm")
    assert boc_url("2025-10-29") == "https://www.bankofcanada.ca/2025/10/fad-press-release-2025-10-29/"


def _daily(points: dict[str, float], end: str) -> pd.Series:
    """계단식 금리 — {바뀐 날: 값} → 매일."""
    idx = pd.date_range(min(points), end, freq="D")
    return pd.Series(list(points.values()), index=pd.to_datetime(list(points))).reindex(idx).ffill()


def test_fomc_and_boc_decisions():
    upper = _daily({"2025-01-01": 4.50, "2025-09-18": 4.25}, "2025-11-03")
    lower = upper - 0.25
    meetings = [("2025-07-29", "2025-07-30", False), ("2025-09-16", "2025-09-17", True),
                ("2025-11-03", "2025-11-03", False), ("2025-12-09", "2025-12-10", True)]
    out = fomc_decisions(meetings, upper, lower, today=date(2025, 11, 5))
    got = [(m["decision"], m["change_bp"]) for m in out["meetings"]]
    assert got == [("hold", 0), ("cut", -25), ("pending", None), ("upcoming", None)]   # 11-03 숫자는 아직 없음
    assert out["meetings"][1]["upper"] == 4.25 and out["meetings"][1]["lower"] == 4.0
    assert out["next"]["date"] == "2025-12-10" and out["current"] == {"upper": 4.25, "lower": 4.0, "date": "2025-11-03"}

    rate = pd.Series([2.75, 2.75, 2.50, 2.50], index=pd.to_datetime(["2025-09-16", "2025-09-17", "2025-09-18",
                                                                     "2025-09-19"]))
    b = boc_decisions([("2025-09-17", False), ("2025-10-29", True)], rate, today=date(2025, 9, 20))
    assert [(m["decision"], m["change_bp"], m["rate"]) for m in b["meetings"]] == [("cut", -25, 2.5),
                                                                                   ("upcoming", None, None)]
    assert b["next"]["mpr"] is True


def test_indicators_status():
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    calm = pd.Series([4.0] * 24, index=months)
    assert ind_sahm(calm)["status"] == "good" and ind_sahm(calm)["value"] == 0.0
    rising = pd.Series([3.5] * 18 + [3.8, 4.0, 4.2, 4.3, 4.4, 4.5], index=months)
    s = ind_sahm(rising)
    assert s["status"] == "bad" and s["value"] >= 0.5 and s["extra"]["unrate"] == 4.5

    weeks = pd.date_range("2024-01-06", periods=60, freq="W-SAT")
    claims = pd.Series([200_000] * 56 + [250_000] * 4, index=weeks)
    c = ind_claims(claims)
    assert c["status"] == "bad" and c["extra"]["rise"] == 0.25 and c["value"] == 250_000

    days = pd.date_range("2024-01-01", periods=400, freq="D")
    assert ind_curve(pd.Series(np.r_[np.full(200, -0.3), np.full(200, 0.8)], index=days))["status"] == "good"
    assert ind_curve(pd.Series(np.full(400, -0.1), index=days))["status"] == "bad"
    hy = ind_hy(pd.Series(np.r_[np.full(340, 3.0), np.full(60, 3.6)], index=days))
    assert hy["status"] == "watch" and hy["extra"]["chg3m"] == 0.6            # 수준은 낮아도 3달 새 +0.6%p

    bdays = pd.bdate_range("2023-01-02", periods=400)
    up = pd.Series(np.linspace(100, 150, 400), index=bdays)
    assert ind_spy200(up)["status"] == "good"
    crash = pd.Series(np.r_[np.linspace(100, 150, 350), np.linspace(150, 110, 50)], index=bdays)
    assert ind_spy200(crash)["status"] == "bad"
    px = pd.DataFrame({t: np.linspace(100, 120, 400) for t in ["XLY", "XLI", "XLF", "XLB"]} |
                      {t: np.linspace(100, 101, 400) for t in ["XLP", "XLU", "XLV"]}, index=bdays)
    cd = ind_cycdef(px)
    assert cd["extra"]["lead"] == "cyclical" and cd["status"] == "good" and cd["extra"]["m3"] > 0
    assert cd["history"][0][1] == 100.0


def test_parsers_and_sheet_frame():
    s = parse_fred_csv("observation_date,UNRATE\n2026-08-01,4.1\n2026-09-01,.\n2026-10-01,4.2\n")
    assert list(s.values) == [4.1, 4.2]
    v = parse_boc_valet('{"observations":[{"d":"2026-10-28","V39079":{"v":"2.25"}},{"d":"2026-10-29","V39079":{"v":"2.50"}}]}')
    assert v.iloc[-1] == 2.5

    data = {"as_of": "2025-11-05", "indicators": [
        {"key": "curve", "group": "economy", "date": "2025-11-04", "value": 0.78,
         "extra": {"inverted_days_1y": 0, "last_inverted": None}, "status": "good", "history": []}],
        "rates": {"fomc": {"current": {"upper": 4.25, "lower": 4.0, "date": "2025-11-03"}, "next": None, "meetings": [
            {"date": "2025-09-17", "start": "2025-09-16", "sep": True, "url": "u", "decision": "cut", "change_bp": -25,
             "upper": 4.25, "lower": 4.0},
            {"date": "2025-12-10", "start": "2025-12-09", "sep": True, "url": "u2", "decision": "upcoming",
             "change_bp": None, "upper": None, "lower": None}]}}}
    f = sheet_frame(data)
    assert list(f.columns) == ["구분", "항목", "날짜", "값", "비교", "판정", "설명"]
    assert f.iloc[0]["값"] == "+0.78%p" and f.iloc[0]["판정"] == "좋음"
    fomc = f[f["구분"] == "FOMC"]
    assert list(fomc["항목"]) == ["예정", "인하"]                      # 앞으로의 일정 먼저, 그다음 최근 결정
    assert fomc.iloc[0]["비교"] == "D-35" and fomc.iloc[0]["값"] == "지금 4.00–4.25%"
    assert fomc.iloc[1]["비교"] == "−0.25%p" and fomc.iloc[1]["값"] == "4.00–4.25%" and fomc.iloc[1]["설명"] == "u"
    json.dumps(_clean({"a": np.float64("nan"), "b": [np.int64(3)]}))
