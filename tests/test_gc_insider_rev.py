"""골든크로스 + 클러스터매수 + 매출 증가 테스트 — 신고자 창, 분기 매출 YoY, 새로 든 사건, 선택기 조건·순서."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gc_insider_rev as g  # noqa: E402


def test_cluster_counts_distinct_buyers_in_90_day_window():
    trades = pd.DataFrame({
        "accession": ["a1", "a1", "a2", "a3", "a4"],
        "filing_date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-02-01", "2020-03-15", "2020-04-20"]),
        "티커": ["AAA"] * 5,
    })
    owners = pd.DataFrame({"accession": ["a1", "a2", "a3", "a4"], "owner": ["X", "X", "Y", "Z"]})
    cb = g.ClusterBook(trades, owners, days=90)
    assert cb.buyers("AAA", pd.Timestamp("2020-01-01")) == 0
    assert cb.buyers("AAA", pd.Timestamp("2020-02-01")) == 1           # 같은 사람(X)이 두 번 → 1명
    assert cb.buyers("AAA", pd.Timestamp("2020-03-15")) == 2           # X, Y (제출일 당일 포함)
    assert cb.buyers("AAA", pd.Timestamp("2020-04-20")) == 3           # 1/2 는 90일 밖이지만 2/1 X 는 안 → X, Y, Z
    assert cb.buyers("AAA", pd.Timestamp("2020-05-01")) == 2           # 창 = 2/2 ~ 5/1 → 2/1 X 는 빠짐
    assert cb.buyers("AAA", pd.Timestamp("2020-07-15")) == 1           # 4/20 Z 만 남음
    assert cb.buyers("AAA", pd.Timestamp("2020-07-20")) == 0
    assert cb.buyers("BBB", pd.Timestamp("2020-03-15")) == 0
    missing = g.ClusterBook(trades, owners.iloc[:1], days=90)          # 신고자 모르면 신고서마다 다른 사람으로
    assert missing.buyers("AAA", pd.Timestamp("2020-03-15")) == 3


def _fact(start, end, val, filed, form):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form}


def test_quarterly_revenue_first_filed_and_q4_from_annual():
    tag = "RevenueFromContractWithCustomerExcludingAssessedTax"
    units = [
        _fact("2019-01-01", "2019-03-31", 100, "2019-05-01", "10-Q"),
        _fact("2019-04-01", "2019-06-30", 110, "2019-08-01", "10-Q"),
        _fact("2019-07-01", "2019-09-30", 120, "2019-11-01", "10-Q"),
        _fact("2019-01-01", "2019-09-30", 330, "2019-11-01", "10-Q"),   # 9개월 누적
        _fact("2019-01-01", "2019-12-31", 480, "2020-02-20", "10-K"),   # 연간 → 4분기 150
        _fact("2020-01-01", "2020-03-31", 130, "2020-05-01", "10-Q"),
        _fact("2019-01-01", "2019-03-31", 999, "2020-05-01", "10-Q"),   # 나중 비교 표시(정정) → 무시
    ]
    gaap = {tag: {"units": {"USD": units}}}
    q = g.quarterly_revenue(gaap)
    assert list(q["val"]) == [100, 110, 120, 150, 130]
    assert q.iloc[3]["filed"] == pd.Timestamp("2020-02-20")
    y = g.yoy_table(q)
    assert len(y) == 1 and y.iloc[0]["yoy"] == pytest.approx(0.30) and y.iloc[0]["known"] == pd.Timestamp("2020-05-01")


def test_rev_book_latest_known_quarter_and_staleness():
    table = pd.DataFrame({
        "티커": ["AAA"] * 3,
        "end": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-03-31"]),
        "known": pd.to_datetime(["2020-05-01", "2020-08-01", "2020-09-01"]),   # 셋째: 옛 분기 재공시
        "yoy": [0.10, -0.05, 0.50],
    })
    rb = g.RevBook(table, stale_days=200)
    assert math.isnan(rb.yoy("AAA", pd.Timestamp("2020-04-30")))
    assert rb.yoy("AAA", pd.Timestamp("2020-05-01")) == pytest.approx(0.10)
    assert rb.yoy("AAA", pd.Timestamp("2020-08-01")) == pytest.approx(-0.05)
    assert rb.yoy("AAA", pd.Timestamp("2020-09-15")) == pytest.approx(-0.05)          # 옛 분기가 최신을 덮지 않음
    assert math.isnan(rb.yoy("AAA", pd.Timestamp("2021-02-01")))                     # 분기 말 200일 초과
    assert math.isnan(rb.yoy("ZZZ", pd.Timestamp("2020-08-01")))


def test_qualify_and_new_entries():
    idx = pd.bdate_range("2020-01-01", periods=6)
    cols = pd.Index(["AAA", "BBB"])
    lists = pd.DataFrame({
        "date": [idx[0], idx[1], idx[2], idx[4], idx[1], idx[2]],
        "ticker": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB"],
        "buyers": [2, 2, 1, 3, 2, 2], "yoy": [0.1, 0.1, 0.1, 0.2, np.nan, -0.1],
    })
    q = g.qualify_matrix(lists, True, True, idx, cols)
    assert list(q["AAA"]) == [True, True, False, False, True, False] and not q["BBB"].any()
    ev = g.new_entries(q)
    assert list(ev["signal_date"]) == [idx[0], idx[4]] and list(ev["entry_pos"]) == [1, 5]
    q_all = g.qualify_matrix(lists, False, False, idx, cols)
    assert list(q_all["BBB"]) == [False, True, True, False, False, False]


class _Ctx:
    date = pd.Timestamp("2020-06-01")


def test_selector_filters_in_mail_order(monkeypatch):
    class CB:
        def buyers(self, tk, day):
            return {"AAA": 2, "BBB": 3, "CCC": 0}.get(tk, 0)

    class RB:
        def yoy(self, tk, day):
            return {"AAA": -0.1, "BBB": 0.2, "CCC": 0.3}.get(tk, float("nan"))

    monkeypatch.setattr(g, "books", lambda: (CB(), RB()))
    gc = [{"ticker": t, "sector": "s", "current_price": 1.0, "timeframes": ["일봉"], "tf_gaps": [("일봉", -0.01)],
           "strategy": "모멘텀", "star": "★", "vol_ratio": 1.0, "vol_ma20": 1.0, "tf_count": 1, "has_daily": True}
          for t in ("SPY", "AAA", "BBB", "CCC")]
    monkeypatch.setattr(g, "extract_golden_cross", lambda df: gc)
    monkeypatch.setattr(g.cs, "detect_market_regime", lambda df: "neutral")
    df = pd.DataFrame({"티커": ["SPY", "AAA", "BBB", "CCC"], "close": [1.0, 2.0, 3.0, 4.0]})
    rec: list = []
    picks, debug = g.make_selector("GCCR", _Ctx(), rec)(df, [], k=1)
    assert [p["ticker"] for p in picks] == ["BBB"] and debug["regime"] == "neutral"   # AAA 매출↓, BBB 통과
    assert picks[0]["entry_price"] == 3.0 and picks[0]["strategy"] == "모멘텀"
    assert [r["ticker"] for r in rec] == ["AAA", "BBB", "CCC"]                         # SPY 제외, 메일 순서
    assert [p["ticker"] for p in g.make_selector("GC", _Ctx())(df, [{"ticker": "AAA"}], k=1)[0]] == ["BBB"]
    assert [p["ticker"] for p in g.make_selector("GCR", _Ctx())(df, [], k=1)[0]] == ["BBB"]
    assert g.make_selector(None, _Ctx())(df, [], k=1)[0] == []                         # 목록만 기록


def test_definitions():
    assert g.TESTS == ("GCCR-pt1", "GCCR-ev") and g.P_BONF == pytest.approx(0.025)
    assert g.CLUSTER_DAYS == 90 and g.CLUSTER_MIN == 2 and g.HOLD == 63
    assert g.VARIANTS["GCCR"][1:] == (True, True) and g.VARIANTS["GC"][1:] == (False, False)
    assert "클러스터매수" in g.spec() and "매출 증가" in g.spec()
    st = {"ΔSharpe": 0.3, "ΔSharpe_하한": 0.05, "ΔSharpe_상한": 0.5, "p_개선아님": 0.01, "구간_개선": 3, "구간_수": 3}
    assert g.judge(st, "채택 후보") == "채택 후보" and g.judge(st, "채택 후보", "관찰") == "관찰"
    assert g.judge({**st, "ΔSharpe_하한": -0.1}, "운과 구분 안 됨", "탈락") == "탈락"
