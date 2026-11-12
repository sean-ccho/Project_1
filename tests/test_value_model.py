"""예측 모델(web_export/value_model.py) — 관문 계산 · 기대수익률 · 기록 (네트워크 없이)."""

from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from web_export import value_model as vm

YEARS = ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]


def _rows(values: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {name: dict(zip(YEARS, vals)) for name, vals in values.items()}


def quality_company(fcf: float = 900.0) -> dict:
    """4년 내내 안정적인 우량 회사 (매출 1000 → 1150, ROE 25% 안팎, 빚 적음)."""
    return {
        "fetched": "2026-11-01",
        "info": {"marketCap": 10_000.0, "currency": "USD", "financialCurrency": "USD",
                 "sector": "Technology", "industry": "Information Technology Services", "shortName": "Good Co"},
        "inc": _rows({
            "Total Revenue": [1000, 1050, 1100, 1150],
            "Net Income Common Stockholders": [250, 262, 275, 288],
            "Operating Income": [330, 345, 360, 380],
            "Pretax Income": [320, 336, 352, 368],
            "Tax Provision": [64, 67, 70, 74],
            "Interest Expense": [10, 10, 10, 10],
            "Cost Of Revenue": [400, 420, 440, 460],
            "Selling General And Administration": [200, 210, 220, 230],
        }),
        "bal": _rows({
            "Stockholders Equity": [1000, 1050, 1100, 1150],
            "Total Debt": [200, 200, 200, 200],
            "Cash Cash Equivalents And Short Term Investments": [100, 100, 100, 100],
            "Total Assets": [2000, 2100, 2200, 2300],
            "Total Liabilities Net Minority Interest": [1000, 1050, 1100, 1150],
            "Current Assets": [800, 840, 880, 920],
            "Current Liabilities": [400, 420, 440, 460],
            "Retained Earnings": [900, 950, 1000, 1050],
            "Accounts Receivable": [150, 157, 165, 172],
            "Net PPE": [500, 525, 550, 575],
            "Long Term Debt": [200, 200, 200, 200],
        }),
        "cf": _rows({
            "Free Cash Flow": [fcf, fcf, fcf, fcf],
            "Operating Cash Flow": [300, 315, 330, 345],
            "Net Issuance Payments Of Debt": [0, 0, 0, 0],
            "Cash Dividends Paid": [-50, -50, -50, -50],
            "Repurchase Of Capital Stock": [-100, -100, -100, -100],
            "Depreciation And Amortization": [50, 52, 55, 57],
        }),
    }


def test_yf_symbol_and_bench():
    assert vm.yf_symbol("BRK.B") == "BRK-B"
    assert vm.yf_symbol("GIB-A.TO") == "GIB-A.TO"
    assert vm.yf_symbol("RCI") == "RCI"
    assert vm.bench_of("PPL.TO") == "XIU.TO"
    assert vm.bench_of("ACN") == "SPY"


def test_series_fills_from_next_name():
    # ACN 처럼 'EBIT' 은 최근 1년, 'Operating Income' 은 4년 — 빈 해는 뒤 이름으로 채운다
    ent = {"inc": {"EBIT": {"2025-12-31": 11.0}, "Operating Income": dict(zip(YEARS, [8.0, 9.0, 10.0, 10.5]))}}
    s = vm.series(ent, "inc", "EBIT", "Operating Income")
    assert list(s.values) == [8.0, 9.0, 10.0, 11.0]
    assert vm.series(ent, "inc", "없는 항목").empty


def test_cv():
    assert vm.cv(pd.Series([0.2, 0.2, 0.2])) == pytest.approx(0.0)
    assert math.isnan(vm.cv(pd.Series([0.2, 0.3])))          # 해가 모자람
    assert math.isnan(vm.cv(pd.Series([-0.1, -0.2, -0.3])))  # 평균이 음수


def test_spread_table_edges():
    assert vm.spread_for(2.3) == pytest.approx(0.0138)   # BB+
    assert vm.spread_for(2.5) == pytest.approx(0.0111)   # BBB 는 2.5 부터
    assert vm.spread_for(0.1) == pytest.approx(0.19)
    assert vm.spread_for(math.inf) == pytest.approx(0.0040)


def test_pv_factor_reduces_to_gordon():
    r, g = 0.09, 0.03
    assert vm.pv_factor(r, g, g) == pytest.approx((1 + g) / (r - g), rel=1e-9)


def test_irr_hits_required_yield():
    for ke, g1 in ((0.064, 0.073), (0.08, 0.0), (0.11, -0.05)):
        y = vm.required_yield(ke, g1)
        assert vm.irr_two_stage(y, g1, vm.TERM_G) == pytest.approx(ke + vm.ALPHA_MIN, abs=1e-6)
    assert math.isnan(vm.irr_two_stage(0.0, 0.03, vm.TERM_G))
    assert math.isnan(vm.irr_two_stage(-0.02, 0.03, vm.TERM_G))


def test_lookup_table_matches_doc():
    # docs/블로그주식분석.md 9.3절 조견표: Ke 8%, g1 +3% → 10.0%, Ke 6%, g1 +10% → 6.1%
    assert vm.required_yield(0.08, 0.03) == pytest.approx(0.100, abs=0.0006)
    assert vm.required_yield(0.06, 0.10) == pytest.approx(0.061, abs=0.0006)


def test_beneish_neutral_company():
    # 두 해가 똑같고 순이익 = 영업현금흐름이면 모든 지수가 1, TATA 0 → −2.48
    same = {"2024-12-31": 100.0, "2025-12-31": 100.0}
    ent = {"inc": {"Total Revenue": same, "Cost Of Revenue": {k: 40.0 for k in same},
                   "Selling General And Administration": {k: 20.0 for k in same},
                   "Net Income": {k: 10.0 for k in same}},
           "bal": {"Accounts Receivable": {k: 15.0 for k in same}, "Current Assets": {k: 50.0 for k in same},
                   "Net PPE": {k: 30.0 for k in same}, "Total Assets": {k: 200.0 for k in same},
                   "Current Liabilities": {k: 40.0 for k in same}, "Long Term Debt": {k: 20.0 for k in same}},
           "cf": {"Operating Cash Flow": {k: 10.0 for k in same}, "Depreciation And Amortization": {k: 5.0 for k in same}}}
    assert vm.beneish_m(ent) == pytest.approx(-2.48, abs=1e-9)


def test_altman_formula():
    ent = quality_company()
    z = vm.altman_z(ent, mcap_fc=10_000.0)
    want = 1.2 * (920 - 460) / 2300 + 1.4 * 1050 / 2300 + 3.3 * 380 / 2300 + 0.6 * 10_000 / 1150 + 1.0 * 1150 / 2300
    assert z == pytest.approx(want)


def test_excluded_sectors():
    assert vm.excluded({"sector": "Financial Services", "industry": "Banks - Regional"})
    assert vm.excluded({"sector": "Real Estate", "industry": "REIT - Retail"})
    assert not vm.excluded({"sector": "Financial Services", "industry": "Financial Data & Stock Exchanges"})
    assert not vm.excluded({"sector": "Technology", "industry": "Software - Application"})


def test_analyze_quality_company_passes_and_buy_price_is_threshold():
    ent = quality_company(fcf=900.0)  # FCF 수익률 9% (시총 10,000)
    res = vm.analyze(ent, price=100.0, mcap_fc=10_000.0, beta=0.5, rf=0.0483)
    assert res["quality"], res["fails"]
    assert res["g_roe"] and res["g_roic"] and res["g_z"] and res["g_m"] and res["g_fcf"]
    assert res["fcf_y"] == pytest.approx(0.09)
    # 매수 기준가에서는 알파가 정확히 5%p
    scale = res["buy"] / 100.0
    at_buy = vm.analyze(ent, price=res["buy"], mcap_fc=10_000.0 * scale, beta=0.5, rf=0.0483)
    assert at_buy["alpha"] == pytest.approx(vm.ALPHA_MIN, abs=1e-6)
    assert res["pass"] == (res["alpha"] >= vm.ALPHA_MIN)


def test_analyze_expensive_company_fails_price_only():
    res = vm.analyze(quality_company(fcf=200.0), price=100.0, mcap_fc=10_000.0, beta=1.0, rf=0.0483)
    assert res["quality"]
    assert not res["g_price"] and not res["pass"]
    assert res["drop"] < 0  # 매수 기준가가 지금보다 낮다


def test_negative_equity_roe_is_not_judged():
    ent = quality_company()
    ent["bal"]["Stockholders Equity"] = dict(zip(YEARS, [-100.0, -120.0, -90.0, -80.0]))
    res = vm.analyze(ent, price=100.0, mcap_fc=10_000.0, beta=0.5, rf=0.0483)
    assert res["g_roe"] is None
    assert "ROE" not in res["fails"]


def _px() -> pd.DataFrame:
    idx = pd.bdate_range("2026-11-02", "2026-11-13")
    return pd.DataFrame({"AAA": np.linspace(100, 110, len(idx)), "SPY": np.linspace(500, 505, len(idx)),
                         "XIU.TO": np.linspace(40, 41, len(idx)), "GIB-A.TO": np.linspace(110, 121, len(idx))},
                        index=idx)


def test_update_log_adds_once_and_tracks_returns():
    px = _px()
    log = vm.update_log(vm.load_log(None), {"AAA"}, px, "2026-11-04")
    ours = [e for e in log["entries"] if e["g"] == "ours"]
    assert [e["t"] for e in ours] == ["AAA"] and log["start"] == "2026-11-04"
    # 저자 커버리지 중 주가가 있는 종목(GIB-A.TO)도 들어간다
    assert any(e["t"] == "GIB-A.TO" and e["g"] == "author" for e in log["entries"])
    n = len(log["entries"])
    log = vm.update_log(log, {"AAA"}, px, "2026-11-06")
    assert len(log["entries"]) == n  # 다시 넣지 않는다
    rows = vm.tracking_rows(log, px, {"AAA"})
    aaa = next(r for r in rows if r["ticker"] == "AAA")
    p0 = px.loc["2026-11-04", "AAA"]
    assert aaa["ret"] == pytest.approx(px["AAA"].iloc[-1] / p0 - 1)
    assert aaa["excess"] == pytest.approx(aaa["ret"] - aaa["bench_ret"])
    assert aaa["passing"]
    summ = vm.tracking_summary(rows)
    assert summ["ours"]["n"] == 1 and summ["ours"]["beat"] == 1


def test_refresh_order_and_refresh_with_fake_fetch():
    cache = {"OLD": {"fetched": "2026-09-01"}, "NEW": {"fetched": "2026-11-10"}}
    order = vm.refresh_order(["NEW", "MISSING", "OLD"], cache, date(2026, 11, 11))
    assert order == ["MISSING", "OLD"]

    def fake(t: str) -> dict:
        if t == "OLD":
            raise RuntimeError("막힘")
        return {"inc": {}, "bal": {}, "cf": {}, "info": {}}

    stats = vm.refresh(["NEW", "MISSING", "OLD"], cache, date(2026, 11, 11), budget_sec=60, fetch=fake, workers=1,
                       log=lambda _m: None)
    assert stats == {"want": 2, "ok": 1, "fail": 1, "left": 0}
    assert cache["MISSING"]["fetched"] == "2026-11-11"
    assert cache["OLD"]["fetched"] == "2026-09-01"  # 실패하면 전 값 그대로


def test_screen_and_build_model_are_json_safe():
    idx = pd.bdate_range("2021-11-01", "2026-11-13")
    px = pd.DataFrame({"GOOD": np.linspace(60, 100, len(idx)), "SPY": np.linspace(300, 500, len(idx))}, index=idx)
    cache = {"GOOD": quality_company(fcf=900.0)}
    rows, funnel = vm.screen(["GOOD", "NODATA"], cache, px, 0.0483, {})
    assert funnel["universe"] == 2 and funnel["computed"] == 1 and funnel["no_data"] == 1
    log = vm.update_log(vm.load_log(None), {r["ticker"] for r in rows if r["pass"]}, px, "2026-11-13")
    model = vm.build_model(rows, funnel, price_date="2026-11-13", rf=0.0483, rf_src="test",
                           track=vm.tracking_rows(log, px, set()), log_start=log["start"], fetch_stats=None)
    body = json.dumps(model, allow_nan=False)  # NaN 이 있으면 여기서 실패
    assert '"schema": 1' in body
    assert model["rows"][0]["ticker"] == "GOOD"
    assert len(model["author"]["coverage"]) == len(vm.AUTHOR_COVERAGE)
