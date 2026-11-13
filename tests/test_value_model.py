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


def _row(t: str, **kw) -> dict:
    """update_log · tracking 에 넣을 최소한의 줄."""
    base = {"ticker": t, "pass": False, "pass_v2": False, "quality": True, "kill": [], "drop": -0.3, "gap": 0.01,
            "industry": "Software - Application"}
    return {**base, **kw}


def test_update_log_adds_once_and_tracks_returns():
    px = _px()
    rows = [_row("AAA", **{"pass": True, "pass_v2": True, "drop": 0.05, "gap": 0.08})]
    log = vm.update_log(vm.load_log(None), rows, px, "2026-11-04")
    ours = [e for e in log["entries"] if e["g"] == "ours"]
    assert [e["t"] for e in ours] == ["AAA"] and log["start"] == "2026-11-04"
    assert any(e["t"] == "AAA" and e["g"] == "v2" for e in log["entries"])
    assert {e["g"] for e in log["entries"] if e["t"] == "AAA"} == {"ours", "v2", "v1_top", "v2_top"}
    # 저자 커버리지 중 주가가 있는 종목(GIB-A.TO)도 들어간다
    assert any(e["t"] == "GIB-A.TO" and e["g"] == "author" for e in log["entries"])
    n = len(log["entries"])
    log = vm.update_log(log, rows, px, "2026-11-06")
    assert len(log["entries"]) == n  # 다시 넣지 않는다
    track = vm.tracking_rows(log, px, {r["ticker"]: r for r in rows})
    aaa = next(r for r in track if r["ticker"] == "AAA" and r["group"] == "ours")
    p0 = px.loc["2026-11-04", "AAA"]
    assert aaa["ret"] == pytest.approx(px["AAA"].iloc[-1] / p0 - 1)
    assert aaa["excess"] == pytest.approx(aaa["ret"] - aaa["bench_ret"])
    assert aaa["passing"] and aaa["exit"] is None
    summ = vm.tracking_summary(track)
    assert summ["ours"]["n"] == 1 and summ["ours"]["beat"] == 1 and summ["ours"]["open"] == 1
    assert set(summ) == set(vm.GROUPS)


def test_v2_entry_closes_on_kill_and_freezes_return():
    px = _px()
    log = vm.update_log(vm.load_log(None), [_row("AAA", **{"pass": True, "pass_v2": True})], px, "2026-11-04")
    killed = [_row("AAA", **{"pass": False, "pass_v2": False, "kill": ["적정가 도달"]})]
    log = vm.update_log(log, killed, px, "2026-11-09")
    v2 = next(e for e in log["entries"] if e["g"] == "v2")
    assert v2["x"] == "2026-11-09" and v2["xr"] == "적정가 도달"
    ours = next(e for e in log["entries"] if e["g"] == "ours")
    assert "x" not in ours  # v1 은 끝내지 않는다
    track = vm.tracking_rows(log, px, {"AAA": killed[0]})
    t2 = next(r for r in track if r["group"] == "v2")
    want = px.loc["2026-11-09", "AAA"] / px.loc["2026-11-04", "AAA"] - 1  # 끝낸 날에서 멈춘다
    assert t2["ret"] == pytest.approx(want) and t2["exit"] == "2026-11-09" and not t2["passing"]


def test_top_baskets_rebalance_monthly():
    idx = pd.bdate_range("2026-10-26", "2026-11-13")
    px = pd.DataFrame({t: np.linspace(10, 11, len(idx)) for t in ("AAA", "BBB", "CCC", "SPY")}, index=idx)
    oct_rows = [_row("AAA", drop=-0.1, gap=0.05), _row("BBB", drop=-0.2, gap=0.04)]
    log = vm.update_log(vm.load_log(None), oct_rows, px, "2026-10-30")
    # 같은 달 안에서는 순위가 바뀌어도 그대로
    log = vm.update_log(log, [_row("CCC", drop=-0.05, gap=0.09)] + oct_rows, px, "2026-10-31")
    assert not any(e["t"] == "CCC" and e["g"] == "v1_top" for e in log["entries"])
    # 다음 달 첫 실행에 다시 고른다 — 빠진 종목은 끝, 새 종목은 들어감
    nov = [_row("CCC", drop=-0.05, gap=0.09)]
    log = vm.update_log(log, nov, px, "2026-11-02")
    v1 = {e["t"]: e for e in log["entries"] if e["g"] == "v1_top"}
    assert v1["AAA"]["x"] == "2026-11-02" and "순위 밖" in v1["AAA"]["xr"]
    assert "x" not in v1["CCC"] and log["baskets"]["v1_top"] == "2026-11"


def test_revenue_falling():
    q = lambda e, v: {"end": e, "revenue": v}  # noqa: E731
    five = [q("2025-06-30", 110), q("2025-09-30", 105), q("2025-12-31", 120), q("2026-03-31", 98), q("2026-06-30", 108)]
    down_year = pd.Series([1000.0, 950.0], index=pd.to_datetime(["2024-12-31", "2025-12-31"]))
    up_year = pd.Series([1000.0, 1050.0], index=pd.to_datetime(["2024-12-31", "2025-12-31"]))
    assert vm.revenue_falling(five, down_year) is True     # 최근 분기 110→108 감소 + 연간 감소
    assert vm.revenue_falling(five, up_year) is False      # 연간은 증가
    rising = five[:-1] + [q("2026-06-30", 115)]
    assert vm.revenue_falling(rising, down_year) is False  # 최근 분기 증가
    assert vm.revenue_falling(five[-2:], down_year) is None  # 1년 전 분기가 없음
    assert vm.revenue_falling(None, down_year) is None     # 분기 자료 없음 (캐나다 등)


def test_top_lists():
    rows = [_row("AAA", drop=-0.10, gap=0.03), _row("BBB", drop=-0.30, gap=0.09, kill=["적정가 도달"]),
            _row("CCC", drop=0.05, gap=0.05), _row("DDD", quality=False, drop=0.2, gap=0.2)]
    assert vm.top_v1(rows) == ["CCC", "AAA", "BBB"]       # 필요한 하락 작은 순 (품질 탈락 제외)
    assert vm.top_v2(rows) == ["CCC", "AAA"]              # 괴리 큰 순, 반증 조건 걸린 BBB 제외


def test_kill_reasons():
    ok = {"quality": True, "rev_down": False, "gap": 0.05, "alpha": 0.06}
    assert vm.kill_reasons(ok) == []
    bad = {"quality": False, "rev_down": True, "gap": -0.01, "alpha": -0.02}
    assert vm.kill_reasons(bad) == ["품질 탈락", "매출 감소 (최근 분기 · 연간)", "시장 가정 성장 > 실제", "적정가 도달"]


def test_analyze_v2_fields():
    res = vm.analyze(quality_company(fcf=900.0), price=100.0, mcap_fc=10_000.0, beta=0.5, rf=0.0483)
    ke, y = res["ke"], res["fcf_y"]
    assert res["implied_g"] == pytest.approx((ke - y) / (1 + y))
    assert res["gap"] == pytest.approx(res["g1"] - res["implied_g"])
    assert res["alpha_bear"] < res["alpha"]
    assert res["pass_v2"] == (res["pass"] and res["alpha_bear"] >= 0)
    assert res["alpha_b06"] == pytest.approx(res["irr"] - (0.0483 + 0.6 * vm.ERP))
    assert res["sensitive"] == (res["pass"] and res["alpha_b06"] < vm.ALPHA_MIN)


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
    assert stats == {"want": 2, "ok": 1, "fail": 1, "left": 0, "pauses": 0}
    assert cache["MISSING"]["fetched"] == "2026-11-11"
    assert cache["OLD"]["fetched"] == "2026-09-01"  # 실패하면 전 값 그대로


def test_screen_and_build_model_are_json_safe():
    idx = pd.bdate_range("2021-11-01", "2026-11-13")
    px = pd.DataFrame({"GOOD": np.linspace(60, 100, len(idx)), "SPY": np.linspace(300, 500, len(idx))}, index=idx)
    cache = {"GOOD": quality_company(fcf=900.0)}
    rows, funnel = vm.screen(["GOOD", "NODATA"], cache, px, 0.0483, {})
    assert funnel["universe"] == 2 and funnel["computed"] == 1 and funnel["no_data"] == 1
    log = vm.update_log(vm.load_log(None), rows, px, "2026-11-13")
    model = vm.build_model(rows, funnel, price_date="2026-11-13", rf=0.0483, rf_src="test",
                           track=vm.tracking_rows(log, px, {r["ticker"]: r for r in rows}), log_start=log["start"],
                           fetch_stats=None, research=vm.research_rows(px, {r["ticker"]: r for r in rows}))
    body = json.dumps(model, allow_nan=False)  # NaN 이 있으면 여기서 실패
    assert '"schema": 2' in body
    assert model["rows"][0]["ticker"] == "GOOD"
    assert len(model["author"]["coverage"]) == len(vm.AUTHOR_COVERAGE)
    assert len(model["author"]["research"]) == len(vm.AUTHOR_RESEARCH)
    assert model["v2"]["top"]["v1"] == ["GOOD"] and model["v2"]["top"]["v2"] == ["GOOD"]
    assert {"pass_v2", "sensitive"} <= set(funnel)


# ── 저자 종목 두 번째 출처: Ideas 의 심층 리서치 글 (2026-11-12)

def test_author_research_source():
    assert vm.author_kind("RHI") == "coverage"  # 커버리지 + 리서치 글 둘 다 → 커버리지
    assert vm.author_kind("FISV") == "research"  # 리서치 글에만 (커버리지 페이지에 없음)
    assert vm.author_kind("MRK") == "screen"
    assert vm.author_kind("AAPL") is None
    ts = vm.author_tickers()
    assert ts[: len(vm.AUTHOR_COVERAGE)] == [c["ticker"] for c in vm.AUTHOR_COVERAGE]
    assert "FISV" in ts and len(ts) == len(set(ts))
    assert {"FISV", "RHI", "MRK"} <= set(vm.universe_tickers())
    for a in vm.AUTHOR_RESEARCH:  # 글 주소 · 날짜 · 통화가 빠지지 않게
        assert a["slug"] and len(a["date"]) == 10 and a["ccy"] in {"USD", "CAD"}


def test_update_log_adds_research_only_author():
    px = _px().assign(FISV=lambda d: np.linspace(45, 50, len(d)))
    log = vm.update_log(vm.load_log(None), [], px, "2026-11-04")
    authors = {e["t"] for e in log["entries"] if e["g"] == "author"}
    assert {"FISV", "GIB-A.TO"} <= authors  # 주가가 있는 저자 종목은 커버리지든 리서치든 들어간다


def test_research_rows_since_post():
    idx = pd.bdate_range("2026-09-01", "2026-11-13")
    px = pd.DataFrame({"FISV": np.linspace(40, 60, len(idx)), "SPY": np.linspace(500, 510, len(idx)),
                       "RHI": np.linspace(30, 33, len(idx))}, index=idx)
    out = vm.research_rows(px, {"FISV": {"price": 60.0}})
    fisv = next(r for r in out if r["ticker"] == "FISV")
    p0 = px.loc["2026-10-05", "FISV"]
    b0 = px.loc["2026-10-05", "SPY"]
    assert fisv["url"].endswith("/post/fiserv-inc") and fisv["bench"] == "SPY"
    assert fisv["ret"] == pytest.approx(60 / p0 - 1, abs=1e-4)
    assert fisv["excess"] == pytest.approx((60 / p0 - 1) - (510 / b0 - 1), abs=1e-4)
    assert fisv["upside"] == pytest.approx(83.75 / 60 - 1, abs=1e-4)
    rhi = next(r for r in out if r["ticker"] == "RHI")
    assert rhi["price"] == pytest.approx(33.0) and rhi["ret"] is None  # 게시일(2026-02-27) 주가가 없으면 비움
    gib = next(r for r in out if r["ticker"] == "GIB-A.TO")
    assert gib["price"] is None and gib["upside"] is None and gib["bench"] == "XIU.TO"
    json.dumps(out, allow_nan=False)


# ── 종목 정보 보충 (2026-11-12 밤: Yahoo 정보가 513종목 전부 빈 값 → 계산 0)

def test_fill_info_from_scan_and_currency_hints():
    scan = {"marketCap": 5_000.0, "sector": "Technology", "industry": "Software", "shortName": "Scan Co"}
    got = vm.fill_info("ABC", {}, scan)
    assert got["marketCap"] == 5_000.0 and got["mcap_scan"] is True
    assert (got["sector"], got["industry"], got["shortName"]) == ("Technology", "Software", "Scan Co")
    assert got["currency"] == got["financialCurrency"] == "USD"
    assert vm.fill_info("GIB-A.TO", {}, None)["currency"] == "CAD"
    rci = vm.fill_info("RCI", {}, None)
    assert (rci["currency"], rci["financialCurrency"]) == ("USD", "CAD")
    # 캐시에 정보가 있으면 그대로 (전체분석 값으로 덮지 않음)
    kept = vm.fill_info("ABC", {"marketCap": 9.0, "currency": "USD", "financialCurrency": "USD"}, scan)
    assert kept["marketCap"] == 9.0 and "mcap_scan" not in kept


def test_screen_uses_scan_info_when_yahoo_info_empty():
    idx = pd.bdate_range("2021-11-01", "2026-11-13")
    px = pd.DataFrame({"GOOD": np.linspace(60, 100, len(idx)), "SPY": np.linspace(300, 500, len(idx))}, index=idx)
    ent = quality_company(fcf=900.0)
    ent["info"] = {}  # 11-12 밤처럼 정보만 빈 값
    rows, funnel = vm.screen(["GOOD"], {"GOOD": ent}, px, 0.0483, {})
    assert funnel["computed"] == 0 and funnel["no_data"] == 1  # 보충 없으면 계산 못 함 (11-12 밤 상태)
    scan = {"GOOD": {"marketCap": 10_000.0, "sector": "Technology", "industry": "Information Technology Services",
                     "shortName": "Good Co"}}
    rows, funnel = vm.screen(["GOOD"], {"GOOD": ent}, px, 0.0483, {}, None, scan)
    assert funnel["computed"] == 1 and funnel["info_scan"] == 1
    want, _ = vm.screen(["GOOD"], {"GOOD": quality_company(fcf=900.0) | {"fetched": "2026-11-13"}}, px, 0.0483, {})
    assert rows[0]["name"] == "Good Co" and rows[0]["pass"] == want[0]["pass"]
    assert rows[0]["alpha"] == pytest.approx(want[0]["alpha"])  # 같은 시가총액이면 같은 결과
    assert rows[0]["co"] == "good" and rows[0]["own"] is True
    rows, _ = vm.screen(["GOOD"], {"GOOD": ent}, px, 0.0483, {}, None, scan, own=[], co_alias={"GOOD": "twin co"})
    assert rows[0]["co"] == "twin co" and rows[0]["own"] is False  # 대상 밖 상장은 우리 대상 쪽 회사 키로


def test_refresh_keeps_old_info_when_new_info_empty():
    old_info = {"marketCap": 1.0, "currency": "USD"}
    cache = {"A": {"fetched": "2026-09-01", "info": old_info}}
    vm.refresh(["A", "B"], cache, date(2026, 11, 12), budget_sec=60, workers=1, log=lambda _m: None,
               fetch=lambda t: {"inc": {"x": {}}, "bal": {}, "cf": {}, "info": {}})
    assert cache["A"]["info"] == old_info and cache["A"]["info_at"] == "2026-09-01"
    assert cache["A"]["fetched"] == "2026-11-12"  # 재무는 새로 받은 것
    assert cache["B"]["info"] == {} and "info_at" not in cache["B"]
    fresh = {"A": {"fetched": "2026-09-01", "info": old_info}}
    vm.refresh(["A"], fresh, date(2026, 11, 12), budget_sec=60, workers=1, log=lambda _m: None,
               fetch=lambda t: {"inc": {}, "bal": {}, "cf": {}, "info": {"marketCap": 2.0}})
    assert fresh["A"]["info"] == {"marketCap": 2.0} and fresh["A"]["info_at"] == "2026-11-12"


def test_load_scan_info_reads_ranked_parquet_and_lists(tmp_path):
    import gzip as _gz

    snap = tmp_path / "snapshot"
    snap.mkdir()
    pd.DataFrame({"티커": ["BRK.B", "ABC"], "fund_market_cap": [1e12, np.nan], "fund_sector": ["Financial Services", "Unknown"],
                  "fund_industry": ["Insurance - Diversified", None]}).to_parquet(snap / "sp500_ranked.parquet")
    (tmp_path / "sp500.json.gz").write_bytes(_gz.compress(json.dumps(
        {"items": {"ABC": {"company": "ABC Corp", "sector": "Technology", "industry": "Software"}}}).encode()))
    got = vm.load_scan_info([tmp_path, None])
    assert got[vm._scan_key("BRK-B")]["marketCap"] == 1e12
    assert got["ABC"] == {"ticker": "ABC", "shortName": "ABC Corp", "sector": "Technology",
                          "industry": "Software"}  # 'Unknown' · NaN 은 건너뜀
    assert got[vm._scan_key("BRK-B")]["ticker"] == "BRK.B"  # 원래 표기 그대로


# ── 대상 넓히기 · 저자 모르게 (2026-11-12): 미국 상장 + TSX, 시가총액 US$10억 이상, 같은 회사는 하나

def _scan(**items: dict) -> dict:
    return {vm._scan_key(t): {"ticker": t, **v} for t, v in items.items()}


def test_company_key_and_page_coverage():
    assert vm.company_key("CGI Inc.") == vm.company_key("CGI Inc") == "cgi"
    assert vm.company_key("Brown-Forman Corporation") == "brown forman"
    assert vm.page_coverage(None) is None and vm.page_coverage({"coverage": []}) is None
    cov = vm.page_coverage({"coverage": [{"ticker": "RHI", "target": 57.9, "rating": "OVERWEIGHT"},
                                         {"ticker": "GIB-A.TO", "target": 306.0}, {"ticker": "RHI"}]})
    assert [c["ticker"] for c in cov] == ["RHI", "GIB-A.TO"]  # 중복은 하나
    assert (cov[0]["ccy"], cov[1]["ccy"], cov[0]["rating"]) == ("USD", "CAD", "OVERWEIGHT")
    assert vm.author_tickers(cov)[:2] == ["RHI", "GIB-A.TO"] and "FISV" in vm.author_tickers(cov)  # 리서치 글은 그대로


def test_own_universe_rules_and_dedup():
    tech = {"sector": "Technology", "industry": "Software"}
    scan = _scan(**{
        "BIG": {"market": "nasdaq", "marketCap": 2e9, "shortName": "Big Co", **tech},
        "SMALL": {"market": "nasdaq", "marketCap": 5e8, "shortName": "Small Co", **tech},
        "BANK": {"market": "nasdaq", "marketCap": 9e9, "shortName": "A Bank", "sector": "Financial Services",
                 "industry": "Banks - Regional"},
        "FDS": {"market": "sp500", "marketCap": 1.2e10, "shortName": "FactSet", "sector": "Financial Services",
                "industry": "Financial Data & Stock Exchanges"},
        "TSXOK.TO": {"market": "tsx", "marketCap": 1.5e9, "shortName": "Maple One", **tech},   # C$15억 × 0.72 = US$10.8억
        "TSXLOW.TO": {"market": "tsx", "marketCap": 1.2e9, "shortName": "Maple Two", **tech},  # US$8.6억 → 밖
        "VENT.V": {"market": "tsxv", "marketCap": 5e9, "shortName": "Venture", **tech},
        "BF-A": {"market": "nasdaq", "marketCap": 1.5e10, "shortName": "Brown-Forman Corporation", **tech},
        "BF-B": {"market": "sp500", "marketCap": 1.5e10, "shortName": "Brown-Forman Corporation", **tech},
        "GIB": {"market": "nasdaq", "marketCap": 2.3e10, "shortName": "CGI Inc.", **tech},
        "GIB-A.TO": {"market": "tsx", "marketCap": 3.2e10, "shortName": "CGI Inc", **tech},
        "NONAME": {"market": "nasdaq", "marketCap": 3e9, **tech},
        # 이름 표기가 달라도 티커 뿌리 + 비슷한 시가총액이면 같은 회사 (Rogers: 'Communication' · 'Communications')
        "RCI": {"market": "nasdaq", "marketCap": 1.77e10, "shortName": "Rogers Communication, Inc.", **tech},
        "RCI-B.TO": {"market": "tsx", "marketCap": 2.5e10, "shortName": "Rogers Communications Inc.", **tech},
        "FM.TO": {"market": "tsx", "marketCap": 2.7e10, "shortName": "First Quantum Minerals", **tech},
        "FM": {"market": "nasdaq", "marketCap": 4e9, "shortName": "Some Other FM Co", **tech},  # 뿌리만 같고 시총이 다름
        # 같은 회사 주식 종류 둘 — 시가총액은 A 가 조금 크지만 거래대금이 큰 B
        "TECK-A.TO": {"market": "tsx", "marketCap": 4.29e10, "shortName": "Teck Resources Limited", "adv": 2e5, **tech},
        "TECK-B.TO": {"market": "tsx", "marketCap": 4.28e10, "shortName": "Teck Resources Limited", "adv": 6e7, **tech},
    })
    own = vm.own_universe(scan, 0.72, sp500=["BF-B", "FDS"])
    assert set(own) == {"BIG", "FDS", "TSXOK.TO", "BF-B", "GIB-A.TO", "NONAME", "RCI-B.TO", "FM.TO", "FM", "TECK-B.TO"}
    assert own["TECK-B.TO"]["aliases"] == ["TECK-A.TO"]
    assert own["TSXOK.TO"]["mcap_usd"] == pytest.approx(1.08e9) and own["GIB-A.TO"]["market"] == "tsx"
    assert own["GIB-A.TO"]["aliases"] == ["GIB"] and own["RCI-B.TO"]["aliases"] == ["RCI"] and own["BF-B"]["aliases"] == ["BF-A"]


def test_update_log_blind_groups_and_author_close():
    px = _px()
    px["GONE"] = px["AAA"]
    day = str(px.index[-1].date())
    mine = _row("AAA", **{"pass": True, "pass_v2": True})
    theirs = _row("GIB-A.TO", **{"pass": True, "pass_v2": True}) | {"own": False}  # 저자 종목이라 비교용으로만 계산
    log = vm.update_log(vm.load_log(None), [mine, theirs], px, day, authors=["GIB-A.TO", "GONE"])
    groups = {(e["t"], e["g"]) for e in log["entries"]}
    assert ("AAA", "ours") in groups and ("GIB-A.TO", "ours") not in groups and ("GIB-A.TO", "v2") not in groups
    assert {("GIB-A.TO", "author"), ("GONE", "author")} <= groups
    log = vm.update_log(log, [mine, theirs], px, day, authors=["GIB-A.TO"])  # 저자 목록에서 GONE 이 빠짐
    gone = next(e for e in log["entries"] if e["t"] == "GONE")
    assert gone["xr"] == "저자 목록에서 빠짐"
    assert vm.top_v1([mine, theirs]) == ["AAA"]


def test_compute_set_adds_authors_and_open_log_names():
    log = {"entries": [{"t": "HELD", "g": "ours", "d": "2026-11-13"}, {"t": "DONE", "g": "v2", "d": "2026-11-13", "x": "2026-11-14"}]}
    got = vm.compute_set(["AAA", "RHI"], [{"ticker": "RHI"}, {"ticker": "NEWCOV.TO"}], log)
    assert got[:2] == ["AAA", "RHI"] and "NEWCOV.TO" in got and "FISV" in got and "MRK" in got  # 리서치 · A-Series
    assert "HELD" in got and "DONE" not in got and len(got) == len(set(got))


def test_retry_info_stops_when_blocked():
    cache = {t: {"inc": {}, "info": {}} for t in ("A", "B", "C", "D")} | {"E": {"inc": {}, "info": {"currency": "USD"}}}
    calls: list[str] = []

    def fake(t: str) -> dict:
        calls.append(t)
        return {"currency": "USD"} if t == "A" else {}

    stats = vm.retry_info(cache, list(cache), date(2026, 11, 13), fetch=fake, log=lambda _m: None)
    assert stats["want"] == 4 and stats["ok"] == 1 and cache["A"]["info_at"] == "2026-11-13"
    assert "E" not in calls  # 정보가 있는 종목은 다시 안 받음


def test_main_resets_log_for_new_universe_and_waits_for_warmup(tmp_path, monkeypatch):
    import gzip as _gz

    out, prev = tmp_path / "out", tmp_path / "prev"
    (out / "snapshot").mkdir(parents=True)
    prev.mkdir()
    n = 600  # 대상 600종목 중 재무가 있는 건 GOOD 하나 → 기록 준비 중
    tickers = ["GOOD"] + [f"T{i:03d}" for i in range(n - 1)]
    pd.DataFrame({"티커": tickers, "fund_market_cap": [1e10] * n, "fund_sector": ["Technology"] * n,
                  "fund_industry": ["Software"] * n}).to_parquet(out / "snapshot" / "nasdaq_ranked.parquet")
    (out / "nasdaq.json.gz").write_bytes(_gz.compress(json.dumps(
        {"items": {t: {"company": f"{t} Corp", "market": "nasdaq"} for t in tickers}}).encode()))
    (prev / vm.LOG_NAME).write_text(json.dumps({"schema": 1, "start": "2026-11-11", "entries": [
        {"t": "RHI", "g": "author", "d": "2026-11-11", "p": 42.0, "bench": "SPY", "bp": 700.0}]}))
    (prev / vm.CACHE_NAME).write_bytes(_gz.compress(json.dumps(
        {"schema": 1, "items": {"GOOD": quality_company(fcf=900.0)}}).encode()))
    idx = pd.bdate_range("2021-11-01", "2026-11-13")
    monkeypatch.setattr(vm, "download_prices", lambda ts: pd.DataFrame(
        {"GOOD": np.linspace(60, 100, len(idx)), "SPY": np.linspace(300, 500, len(idx))}, index=idx))
    monkeypatch.setattr(vm, "risk_free", lambda: (0.0483, "test"))
    monkeypatch.setattr(vm, "fx_rates", lambda pairs: {})
    assert vm.main(["--out-dir", str(out), "--prev-dir", str(prev), "--no-fetch"]) == 0
    model = json.loads((out / vm.OUT_NAME).read_text())
    log = json.loads((out / "snapshot" / vm.LOG_NAME).read_text())
    assert log["universe"] == vm.UNIVERSE_ID and log["entries"] == []  # 예전 대상(S&P 500) 기록은 새로 시작, 준비 중이라 비어 있음
    assert model["tracking"]["warmup"] == {"have": 1, "want": n}
    assert model["params"]["universe"]["size"] == n and model["funnel"]["universe"] == n
    good = next(r for r in model["rows"] if r["ticker"] == "GOOD")
    assert good["own"] is True and good["name"] == "Good Co"  # 캐시에 정보가 있으면 그대로


def test_refresh_pauses_when_blocked_then_resumes():
    """막히면(연속 실패) 쉬었다가 실패한 종목부터 이어 받는다 — 처음 30번은 막힘, 그 뒤 풀림."""
    calls = {"n": 0}

    def fake(t: str) -> dict:
        calls["n"] += 1
        if calls["n"] <= 30:
            raise RuntimeError("429")
        return {"inc": {"x": {}}, "bal": {}, "cf": {}, "info": {"currency": "USD"}}

    naps: list[float] = []
    cache: dict = {}
    names = [f"T{i:02d}" for i in range(40)]
    stats = vm.refresh(names, cache, date(2026, 11, 13), budget_sec=3600, fetch=fake, workers=1,
                       log=lambda _m: None, pause_sec=5, sleep=naps.append)
    assert stats["ok"] == 40 and stats["fail"] == 0 and stats["left"] == 0  # 풀린 직후 실패한 5개도 맨 뒤에서 다시
    assert stats["pauses"] == 1 and naps == [5]
    blocked: dict = {}
    stats = vm.refresh(names, blocked, date(2026, 11, 13), budget_sec=3600, workers=1, log=lambda _m: None,
                       fetch=lambda t: (_ for _ in ()).throw(RuntimeError("429")), pause_sec=5, pause_max=1,
                       sleep=lambda _s: None)
    assert stats["pauses"] == 1 and stats["ok"] == 0 and stats["fail"] == 25 and stats["left"] == 15
