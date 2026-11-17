"""scripts/value_v3_backtest.py — v3 재검증 백테스트의 시점 · 데이터 · 수익률 계산 (2026-11-16)."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import value_v3_backtest as vb  # noqa: E402


def _rows(*items):
    """[(end, filed, val, prio)] → index_rows 모양."""
    return vb.index_rows(items)


# ── 시점 선택

def test_pick_latest_filed_before_asof_and_tag_priority():
    rows = _rows(("2019-12-31", "2020-02-10", 100.0, 1),   # 원래 10-K (뒤 태그)
                 ("2019-12-31", "2020-02-10", 101.0, 0),   # 같은 날 앞 태그 → 이긴다
                 ("2019-12-31", "2021-02-10", 105.0, 0))   # 다음 해 10-K 의 재작성 값
    assert vb.pick(rows, "2019-12-31", "2020-05-31") == 101.0
    assert vb.pick(rows, "2019-12-28", "2021-05-31") == 105.0   # 같은 달이면 같은 회계연도 (52/53주)
    assert vb.pick(rows, "2019-12-31", "2020-01-31") is None     # 아직 공시 전


def test_select_ends_window_staleness_and_gaps():
    rev = _rows(*[(f"{y}-12-31", f"{y + 1}-02-15", 10.0 + y, 0) for y in range(2012, 2020)])
    assert vb.select_ends(rev, "2020-05-31") == ["2016-12-31", "2017-12-31", "2018-12-31", "2019-12-31"]
    assert vb.select_ends(rev, "2020-02-01") == ["2015-12-31", "2016-12-31", "2017-12-31", "2018-12-31"]
    assert vb.select_ends(rev, "2022-05-31") == []                # 최근 해가 18개월 넘게 오래됨
    gap = _rows(("2014-12-31", "2015-02-15", 1.0, 0), ("2016-12-31", "2017-02-15", 1.0, 0),
                ("2017-12-31", "2018-02-15", 1.0, 0), ("2018-12-31", "2019-02-15", 1.0, 0))
    assert vb.select_ends(gap, "2019-05-31") == ["2016-12-31", "2017-12-31", "2018-12-31"]  # 빈 해 앞은 버림
    short = _rows(("2017-12-31", "2018-02-15", 1.0, 0), ("2018-12-31", "2019-02-15", 1.0, 0))
    assert vb.select_ends(short, "2019-05-31") == []              # 3년 미만


def test_total_debt_tag_combinations():
    e, asof = "2019-12-31", "2020-05-31"
    f = "2020-02-15"
    assert vb.total_debt({"debt_total": _rows((e, f, 50.0, 0)), "ltd_all": _rows((e, f, 1.0, 0))}, e, asof) == 50.0
    assert vb.total_debt({"ltd_all": _rows((e, f, 40.0, 0)), "stb": _rows((e, f, 5.0, 0)),
                          "cp": _rows((e, f, 2.0, 0))}, e, asof) == 47.0
    assert vb.total_debt({"ltd_noncur": _rows((e, f, 30.0, 0)), "debt_cur": _rows((e, f, 4.0, 0)),
                          "stb": _rows((e, f, 9.0, 0))}, e, asof) == 34.0   # 단기 합계(debt_cur)가 있으면 그것
    assert vb.total_debt({"ltd_noncur": _rows((e, f, 30.0, 0)), "ltd_cur": _rows((e, f, 3.0, 0)),
                          "stb": _rows((e, f, 1.0, 0))}, e, asof) == 34.0
    assert vb.total_debt({}, e, asof) is None


# ── 시가총액 · 분할

def _px(splits: dict[str, float]) -> pd.DataFrame:
    idx = pd.date_range("2013-01-31", "2026-11-30", freq="ME")
    df = pd.DataFrame({"close": 10.0, "adj": 10.0, "split": 0.0}, index=idx)
    for d, f in splits.items():
        df.loc[vb.month_end(d), "split"] = f
    return df


def test_share_ratio_confirms_real_split_and_ignores_adr_ratio_change():
    # 4:1 분할 (2020-08) — 공시 주식 수도 4배로 뜀 → 분할로 보고 R × 4
    co = {"rows": {"dei": _rows(("2020-04-17", "2020-04-30", 4.3e9, 0), ("2026-10-17", "2026-10-30", 15.0e9, 0))}}
    r, how = vb.share_ratio(co, _px({"2020-08-31": 4.0}), "2020-05-31", "2026-11-16")
    assert how == "dei_split" and r == pytest.approx(4.3 * 4 / 15.0)
    # ADR 비율 변경 (yfinance '분할' 10) — 원주 수는 그대로 → 분할 무시
    co = {"rows": {"shares": _rows(("2014-12-31", "2015-03-30", 1.15e8, 0), ("2025-12-31", "2026-03-30", 1.02e8, 0))}}
    r, how = vb.share_ratio(co, _px({"2015-11-30": 10.0}), "2015-05-31", "2026-11-16")
    assert how == "shares" and r == pytest.approx(1.15 / 1.02)
    assert vb.share_ratio({"rows": {}}, _px({}), "2015-05-31", "2026-11-16") == (1.0, "no_shares")


def test_build_candidates_never_maps_tsx_by_ticker_root_and_links_predecessor():
    sec_rows = [[1, "PPL CORP", "PPL", "NYSE"], [2, "TELUS CORP", "TU", "NYSE"], [3, "ExxonMobil Holdings", "XOM", "NYSE"]]
    scan = {"TU": {}, "XOM": {}}
    mem = pd.DataFrame({"date": pd.to_datetime(["2014-01-01"]), "tickers": ["XOM,PPL.TO"]})
    out = vb.build_candidates(sec_rows, scan, mem)
    assert out["PPL-TO"]["cik"] is None              # Gemini 는 PPL.TO(펨비나) → PPL Corp 로 이었다
    assert out["TU"]["cik"] == 2
    assert out["XOM"]["cik"] == 3 and out["XOM"]["alt"] == 34088


def test_merge_facts_latest_filing_wins_across_ciks():
    old = {"ccy": "USD", "rows": {"rev": _rows(("2014-12-31", "2015-02-15", 66.0, 0))}, "q": {}}
    new = {"ccy": "USD", "rows": {"rev": _rows(("2014-12-31", "2016-02-15", 66.5, 0), ("2015-12-31", "2016-02-15", 75.0, 0))},
           "q": {}}
    m = vb.merge_facts(new, old)
    assert vb.pick(m["rows"]["rev"], "2014-12-31", "2015-05-31") == 66.0
    assert vb.pick(m["rows"]["rev"], "2014-12-31", "2016-05-31") == 66.5
    assert vb.merge_facts(None, old) is old


def test_extract_duration_filters():
    facts = {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2019-01-01", "end": "2019-12-31", "filed": "2020-02-01", "val": 100, "form": "10-K"},
        {"start": "2019-10-01", "end": "2019-12-31", "filed": "2020-02-01", "val": 30, "form": "10-K"},
        {"start": "2019-01-01", "end": "2019-12-31", "filed": "2020-02-01", "val": 101, "form": "6-K"}]}}}}
    assert sorted(v for _, _, v, _ in vb.extract(facts, "rev", "USD", "flow")) == [100.0, 101.0]   # 6-K 도 연간이면 쓴다
    assert [v for _, _, v, _ in vb.extract(facts, "rev", "USD", "q")] == [30.0]
    assert vb.reporting_ccy(facts) == "USD"


# ── 수익률

def test_run_series_buy_and_hold_with_turnover_cost():
    p1 = {"A": np.array([1.1, 1.21]), "B": np.array([1.0, 1.0])}
    p2 = {"A": np.array([1.0, 1.0]), "C": np.array([0.9, 0.9])}
    m, per = vb.run_series([["A", "B"], ["A", "C"]], [p1, p2], cost=0.01)
    # 1기: 처음 매수 비용 1% (매매 금액 = 100%)
    assert m[0] == pytest.approx(1.05 * 0.99 - 1)
    assert per[0] == pytest.approx(1.105 * 0.99 - 1)
    # 2기: 끝 비중 A 1.21/2.21 · B 1/2.21 → A 0.5 · C 0.5, 매매 금액 = |0.5−0.5475| + 0.4525 + 0.5
    w_a = 1.21 / 2.21
    turn = abs(0.5 - w_a) + (1 - w_a) + 0.5
    assert m[2] == pytest.approx(0.95 * (1 - 0.01 * turn) - 1)
    assert len(m) == 4
    with pytest.raises(ValueError):
        vb.run_series([["Z"]], [p1])


# ── v3 규칙 (문서 4.6 = Gemini 코드)

def _ent(cash: float, debt: float, sbc: float = 0.0) -> dict:
    ends = ["2016-12-31", "2017-12-31", "2018-12-31", "2019-12-31"]
    v = lambda x: {e: float(x) for e in ends}  # noqa: E731
    return {"info": {}, "inc": {"Total Revenue": {e: 100.0 + 5 * i for i, e in enumerate(ends)},
                                "Net Income Common Stockholders": v(15), "EBIT": {e: 20.0 + i for i, e in enumerate(ends)},
                                "Pretax Income": v(19), "Tax Provision": v(4)},
            "bal": {"Stockholders Equity": v(80), "Total Debt": v(debt),
                    "Cash Cash Equivalents And Short Term Investments": v(cash)},
            "cf": {"Free Cash Flow": {e: 10.0 + i for i, e in enumerate(ends)}, "Stock Based Compensation": v(sbc)}}


def test_v3_ev_denominator_counts_cash_twice_vs_mcap_variant():
    res = {"sh_y": 0.02, "g_roe": True, "g_roic": True, "g_z": True, "g_m": True, "g_fcf": True, "ke": 0.08}
    a = vb.v3_metrics(res, _ent(cash=60.0, debt=0.0), mcap_fc=100.0, use_ev=True)
    b = vb.v3_metrics(res, _ent(cash=60.0, debt=0.0), mcap_fc=100.0, use_ev=False)
    assert b["fcf_y_v3"] == pytest.approx(12.0 / 100.0)          # 최근 3년 평균 FCF ÷ 시가총액
    assert a["fcf_y_v3"] == pytest.approx(12.0 / 40.0)           # 순현금 60 을 빼서 3배 가까이 '싸 보임'
    s = vb.v3_metrics(res, _ent(cash=0.0, debt=0.0, sbc=2.0), mcap_fc=100.0, use_ev=False)
    assert s["fcf_y_v3"] == pytest.approx(10.0 / 100.0)          # SBC 차감
    neg = dict(res, sh_y=-0.01)
    assert "sh_dilution" in vb.v3_metrics(neg, _ent(0.0, 0.0), 100.0, False)["kill_v3"]


def test_select_portfolios_industry_cap_and_pass_first():
    def row(t, ind, pass3, a, ab):
        return {"ticker": t, "industry": ind, "pass": False, "pass_v2": False, "drop": 0.0, "gap": 0.0, "kill": [],
                "quality_v3A": True, "g_rev_v3A": True, "g_sh_v3A": True, "pass_v3A": pass3, "kill_v3A": [],
                "alpha_v3A": a, "alpha_bear_v3A": ab}
    rows = [row("T1", "Tank", False, 0.20, 0.10), row("T2", "Tank", False, 0.19, 0.10), row("T3", "Tank", False, 0.18, 0.10),
            row("T4", "Tank", False, 0.17, 0.10), row("S1", "Soft", True, 0.06, 0.01)]
    top = [r["ticker"] for r in vb.select_portfolios(rows, cap=3)["top_v3A"]]
    assert top == ["S1", "T1", "T2", "T3"]                       # 통과 종목이 먼저, 같은 업종은 3개까지
