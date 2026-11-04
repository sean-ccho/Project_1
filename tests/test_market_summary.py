"""웹 '투자자 한눈에' (2026-11-03) — market_summary · stock_detail 순수 함수 + 전체분석 연결 (네트워크 없음)."""
import gzip
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_universe_scan as rus  # noqa: E402
from web_export import market_summary as ms  # noqa: E402
from web_export import stock_detail as sd  # noqa: E402
from web_export import universe as wu  # noqa: E402


def _ohlcv(close: list[float]) -> pd.DataFrame:
    idx = pd.bdate_range("2025-01-01", periods=len(close))
    c = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1_000.0})


# ── price_stats

def test_price_stats_rising_series() -> None:
    df = _ohlcv(list(np.linspace(50, 100, 260)))
    st = ms.price_stats(df)
    assert st["ret_1d"] == pytest.approx(100 / df["Close"].iloc[-2] - 1)
    assert st["above50"] is True and st["above200"] is True
    assert st["new_high"] is True and st["new_low"] is False
    last252 = df.iloc[-252:]
    assert st["range_52w"] == {"low": round(last252["Low"].min(), 2), "high": round(last252["High"].max(), 2)}


def test_price_stats_uses_chart_prices_for_range_and_handles_short_history() -> None:
    adj = _ohlcv([10.0] * 30)
    chart = _ohlcv([12.0] * 30)
    st = ms.price_stats(adj, chart)
    assert st["range_52w"] == {"low": round(12 * 0.99, 2), "high": round(12 * 1.01, 2)}
    assert st["above50"] is None and st["above200"] is None  # 일봉이 모자람
    assert ms.price_stats(pd.DataFrame())["ret_1d"] is None


# ── 업종 중앙값 · 카드

def test_industry_medians_filters_small_caps_losses_and_outliers() -> None:
    rows = [{"industry": "Semis", "cap": 5e9, "pe": v, "fpe": None} for v in (10, 20, 30, -5, 900)]
    rows.append({"industry": "Semis", "cap": 5e8, "pe": 1.0})  # 시총 $1B 미만 → 빠짐
    rows.append({"industry": None, "cap": 9e9, "pe": 99.0})
    med = ms.industry_medians(rows)
    assert med["Semis"]["n"] == 5
    assert med["Semis"]["pe"] == 20  # 10 · 20 · 30 (적자 · 500배 넘는 값 제외)
    assert med["Semis"]["fpe"] is None  # 값 3개 미만
    assert None not in med


def test_valuation_and_analyst_blocks() -> None:
    row = {"fund_pe": 146.13, "fund_forward_pe": 16.03, "fund_peg": np.nan, "fund_ps": 5.3, "fund_pb": None,
           "fund_ev_ebitda": 25.86, "fund_target_mean": 160.294, "fund_target_high": 186, "fund_target_low": 105,
           "fund_target_median": 165, "fund_analyst_n": 26.0, "fund_rec_mean": 1.82143, "fund_rec_key": "buy"}
    med = {"Drug": {"n": 15, "pe": 27.33, "fpe": 15.31, "peg": 3.04, "ps": 3.57, "pb": 5.37, "ev_ebitda": 13.88}}
    v = ms.valuation_block(row, "Drug", med)
    assert v["pe"] == 146.13 and v["peg"] is None and v["ind"]["fpe"] == 15.31 and v["n_peers"] == 15
    a = ms.analyst_block(row, {"strong_buy": 5, "buy": 14, "hold": 9, "sell": 0, "strong_sell": 0})
    assert a["mean"] == 160.29 and a["n"] == 26 and a["rec_key"] == "buy" and a["rec"]["buy"] == 14
    assert ms.analyst_block({"fund_target_mean": None}) is None
    assert ms.valuation_block({}, "Drug", med) is None
    assert ms.analyst_block({**row, "fund_rec_key": "nan"})["rec_key"] is None


# ── market.json

def _item(chg: float | None, cap: float, sector: str = "Technology", ed: str | None = None,
          vr: float | None = 1.0, timing: str | None = None) -> dict:
    return {"company": "Co", "price": 10.0, "ret_1d": chg, "vol_ratio": vr, "sector": sector, "earnings_date": ed,
            "fundamentals": {"market_cap": cap},
            "earnings_move": {"timing": timing, "avg": 0.05} if timing else None}


def test_trading_days_after_skips_weekend() -> None:
    assert ms.trading_days_after("2026-11-06", 3) == ["2026-11-09", "2026-11-10", "2026-11-11"]


def test_market_summary_breadth_movers_sectors_calendar() -> None:
    items = {
        "AAA": _item(0.05, 50e9, ed="2026-11-04", timing="amc"),
        "BBB": _item(-0.02, 3e9, sector="Energy", ed="2026-11-04", timing="bmo"),
        "CCC": _item(0.01, 1e9, ed="2026-11-05"),   # S&P 밖 · $2B 미만 → 목록 풀에서 빠짐
        "DDD": _item(-0.10, 20e9, vr=5.0),           # S&P 밖 · $2B 이상
        "EEE": _item(None, 9e9),
    }
    stats = {"AAA": {"above50": True, "above200": True, "new_high": True},
             "BBB": {"above50": False, "above200": True, "new_low": True}, "SPY": {"ret_1d": 0.003}}
    m = ms.market_summary(items, ["AAA", "BBB", "SPY"], stats, "2026-11-03", 0.003, "t")
    b = m["breadth"]
    assert (b["n"], b["adv"], b["dec"]) == (2, 1, 1)
    assert b["above50"] == 0.5 and b["above200"] == 1.0 and b["new_high"] == 1 and b["new_low"] == 1
    assert b["spy"] == 0.003 and b["equal"] == pytest.approx(0.015)
    assert [g["t"] for g in m["gainers"]] == ["AAA", "BBB", "DDD"]
    assert [g["t"] for g in m["losers"]][0] == "DDD"
    assert m["volume"][0]["t"] == "DDD" and m["pool"] == 3
    assert [s["sector"] for s in m["sectors"]] == ["Technology", "Energy"]
    assert m["calendar"][0]["date"] == "2026-11-04"
    assert [i["t"] for i in m["calendar"][0]["items"]] == ["AAA", "BBB"]
    assert m["calendar"][0]["items"][0]["timing"] == "amc" and m["calendar"][0]["n_big"] == 1
    assert m["calendar"][1]["n"] == 1
    assert ms.market_summary(items, [], {}, None, None, "t") is None
    json.dumps(m, allow_nan=False)


# ── stock_detail

def test_parse_yfinance_frames() -> None:
    rec = pd.DataFrame([{"period": "0m", "strongBuy": 5, "buy": 14, "hold": 9, "sell": 0, "strongSell": 0},
                        {"period": "-1m", "strongBuy": 1, "buy": 1, "hold": 1, "sell": 1, "strongSell": 1}])
    assert sd.parse_rec(rec) == {"strong_buy": 5, "buy": 14, "hold": 9, "sell": 0, "strong_sell": 0}
    assert sd.parse_rec(pd.DataFrame()) is None

    idx = pd.to_datetime(["2026-10-29 06:45", "2026-07-29 06:45", "2027-02-01 06:45"]).tz_localize("America/New_York")
    ed = pd.DataFrame({"EPS Estimate": [2.08, 2.03, 1.77], "Reported EPS": [2.49, 2.13, np.nan],
                       "Surprise(%)": [19.59, 5.04, np.nan]}, index=idx)
    eps = sd.parse_eps(ed)
    assert [e["date"] for e in eps] == ["2026-07-29", "2026-10-29"]
    assert eps[-1] == {"date": "2026-10-29", "est": 2.08, "act": 2.49, "surprise": 19.59}

    q = pd.DataFrame({pd.Timestamp("2026-06-30"): [16.6e9, -0.54], pd.Timestamp("2026-03-31"): [16.3e9, -1.72],
                      pd.Timestamp("2025-12-31"): [np.nan, 1.19]}, index=["Total Revenue", "Diluted EPS"])
    rev = sd.parse_revenue(q)
    assert [r["end"] for r in rev] == ["2026-03-31", "2026-06-30"] and rev[-1]["eps_gaap"] == -0.54

    ee = pd.DataFrame({"avg": [1.767], "yearAgoEps": [2.04], "numberOfAnalysts": [18]}, index=["0q"])
    re_ = pd.DataFrame({"avg": [16.3e9], "growth": [-0.005]}, index=["0q"])
    assert sd.parse_next(ee, re_) == {"eps_est": 1.767, "eps_year_ago": 2.04, "n": 18, "rev_est": 16.3e9,
                                      "rev_growth": -0.005}


def test_refresh_order_tiers() -> None:
    today = date(2026, 11, 3)
    cache = {
        "OLD": {"fetched": "2026-10-10"},          # 14일 넘음 → ③
        "FRESH": {"fetched": "2026-11-01"},        # 새것 → 빠짐
        "REP": {"fetched": "2026-10-30"},          # 10-29 발표 → 11-02 전에 받음 → ②
        "REP_OK": {"fetched": "2026-11-03"},       # 10-29 발표 · 11-03 받음 (≥ 11-02) → 빠짐
    }
    rep = {"REP": "2026-10-29", "REP_OK": "2026-10-29", "FRESH": "2026-12-01"}
    order = sd.refresh_order(["OLD", "FRESH", "REP", "REP_OK", "NEW"], cache, today, rep)
    assert order == ["NEW", "REP", "OLD"]


def test_refresh_budget_failures_and_cache_roundtrip(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake(t: str) -> dict:
        calls.append(t)
        if t == "BAD":
            raise RuntimeError("blocked")
        return {"rec": None, "eps": [{"date": "2026-10-29", "est": 1.0, "act": 1.1, "surprise": 10.0}],
                "revenue": [], "next": {"eps_est": 1.2}}

    cache: dict = {"KEEP": {"fetched": "2026-11-02", "eps": [1]}}
    stats = sd.refresh(["A", "BAD", "KEEP"], cache, date(2026, 11, 3), {}, budget_sec=60, fetch=fake, workers=2,
                       log=lambda _m: None)
    assert stats == {"want": 2, "ok": 1, "fail": 1, "left": 0}
    assert cache["A"]["fetched"] == "2026-11-03" and "BAD" not in cache and set(calls) == {"A", "BAD"}
    assert sd.quarters_block(cache["A"], "2027-02-01")["next"] == {"date": "2027-02-01", "eps_est": 1.2}
    assert sd.quarters_block({"eps": [], "revenue": []}, None) is None

    path = tmp_path / "snapshot" / sd.CACHE_NAME
    sd.save_cache(path, cache, keep={"A"})
    back = sd.load_cache(path)
    assert set(back) == {"A"}
    assert sd.load_cache(tmp_path / "none.json.gz") == {}
    (tmp_path / "bad.json.gz").write_bytes(gzip.compress(b"not json"))
    assert sd.load_cache(tmp_path / "bad.json.gz") == {}

    # 시간 한도 0 → 아무것도 안 받고 남은 수만
    stats0 = sd.refresh(["Z"], {}, date(2026, 11, 3), {}, budget_sec=0, fetch=fake, log=lambda _m: None)
    assert stats0["ok"] == 0 and stats0["left"] == 1


def test_partial_fetch_keeps_old_values_and_retries_next_night() -> None:
    """4개 요청 중 일부만 실패 → 못 받은 칸은 전 값 유지, 다음 밤 다시 (연속 3번까지), 다 받으면 표시가 지워진다."""
    old = {"fetched": "2026-10-01", "rec": {"buy": 5}, "eps": [1], "revenue": [{"q": "2026-06"}], "next": {"eps_est": 1.0}}
    cache: dict = {"MRK": dict(old)}

    def partial(_t: str) -> dict:
        return {"rec": {"buy": 7}, "eps": [2], "next": {"eps_est": 1.3}}  # 매출만 실패

    day1 = date(2026, 11, 3)
    sd.refresh(["MRK"], cache, day1, {}, budget_sec=60, fetch=partial, workers=1, log=lambda _m: None)
    ent = cache["MRK"]
    assert ent["revenue"] == [{"q": "2026-06"}] and ent["rec"] == {"buy": 7} and ent["eps"] == [2]
    assert ent["missing"] == ["revenue"] and ent["tries"] == 1 and ent["fetched"] == "2026-11-03"
    # 같은 날 다시 돌려도 안 받고, 다음 날은 다시 받는다 (14일을 기다리지 않음)
    assert sd.refresh_order(["MRK"], cache, day1, {}) == []
    assert sd.refresh_order(["MRK"], cache, date(2026, 11, 4), {}) == ["MRK"]

    # 연속 3번 실패하면 그 뒤는 14일 주기로
    for d in (4, 5):
        sd.refresh(["MRK"], cache, date(2026, 11, d), {}, budget_sec=60, fetch=partial, workers=1, log=lambda _m: None)
    assert cache["MRK"]["tries"] == 3
    assert sd.refresh_order(["MRK"], cache, date(2026, 11, 6), {}) == []
    assert sd.refresh_order(["MRK"], cache, date(2026, 11, 19), {}) == ["MRK"]

    # 다 받으면 missing · tries 가 지워진다
    def full(_t: str) -> dict:
        return {"rec": None, "eps": [3], "revenue": [{"q": "2026-09"}], "next": {}}

    sd.refresh(["MRK"], cache, date(2026, 11, 19), {}, budget_sec=60, fetch=full, workers=1, log=lambda _m: None)
    assert cache["MRK"] == {"fetched": "2026-11-19", "rec": None, "eps": [3], "revenue": [{"q": "2026-09"}], "next": {}}

    # 처음 받는 종목이 일부만 받혀도 받은 칸은 남고 다음 밤에 다시
    new: dict = {}
    sd.refresh(["NEW"], new, day1, {}, budget_sec=60, fetch=partial, workers=1, log=lambda _m: None)
    assert "revenue" not in new["NEW"] and new["NEW"]["missing"] == ["revenue"]


# ── 전체분석 연결

def test_item_info_adds_ret_1d_and_range() -> None:
    row = pd.Series({"티커": "AAA", "현재가격": 10.0, "1일수익률": 0.0123456, "52주최저": 8.0, "52주최고": 12.5})
    info = wu.item_info(row, tv=None, insider=None, isb=None, sqz=None)
    assert info["ret_1d"] == 0.0123 and info["range_52w"] == {"low": 8.0, "high": 12.5}
    info2 = wu.item_info(pd.Series({"티커": "B", "52주최저": np.nan}), tv=None, insider=None, isb=None, sqz=None)
    assert info2["range_52w"] is None and info2["ret_1d"] is None


def test_web_extras_and_write_payloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ranked = pd.DataFrame([
        {"티커": "AAA", "fund_industry": "Semiconductors", "fund_market_cap": 50e9, "fund_pe": 20.0,
         "fund_forward_pe": 15.0, "fund_target_mean": 120.0, "fund_analyst_n": 10, "fund_rec_key": "buy"},
        {"티커": "BBB", "fund_industry": "Semiconductors", "fund_market_cap": 3e9, "fund_pe": 30.0},
        {"티커": "CCC", "fund_industry": "Semiconductors", "fund_market_cap": 2e9, "fund_pe": 40.0},
        {"티커": "SMALL", "fund_industry": "Banks", "fund_market_cap": 5e8},
    ])
    items = {"AAA": _item(0.01, 50e9, ed="2026-11-04"), "BBB": _item(-0.01, 3e9), "CCC": _item(0.0, 2e9),
             "SMALL": _item(0.2, 5e8)}
    fetched: list[str] = []
    from web_export import stock_detail

    def fake(t: str) -> dict:
        fetched.append(t)
        return {"rec": {"strong_buy": 1, "buy": 2, "hold": 0, "sell": 0, "strong_sell": 0},
                "eps": [{"date": "2026-10-29", "est": 1.0, "act": 1.2, "surprise": 20.0}], "revenue": [], "next": {}}

    orig = stock_detail.refresh

    def refresh(*a, **kw):
        kw["fetch"] = fake
        kw["log"] = lambda _m: None
        return orig(*a, **kw)

    monkeypatch.setattr(stock_detail, "refresh", refresh)
    stats = {"AAA": {"ret_1d": 0.01, "above50": True}, "SPY": {"ret_1d": 0.002}}
    details, market = rus.web_extras([ranked], items, stats, ["AAA"], "2026-11-03", None, tmp_path)
    assert sorted(fetched) == ["AAA", "BBB", "CCC"]  # S&P 500 + 시총 $2B 이상만
    assert details["AAA"]["valuation"]["ind"]["pe"] == 30.0 and details["AAA"]["valuation"]["n_peers"] == 3
    assert details["AAA"]["analyst"]["rec"]["buy"] == 2
    assert details["AAA"]["quarters"]["next"]["date"] == "2026-11-04"
    assert "SMALL" not in details
    assert market["breadth"]["spy"] == 0.002 and market["bar_date"] == "2026-11-03"
    assert (tmp_path / "snapshot" / stock_detail.CACHE_NAME).exists()

    payload = wu.market_payload("sp500", "S&P 500", items, bar_date="2026-11-03", counts={})
    sizes = wu.write_payloads(tmp_path / "out", [payload], {}, details, market)
    one = json.loads((tmp_path / "out" / "stocks" / "AAA.json").read_text())
    assert one["detail"]["valuation"]["pe"] == 20.0
    assert "detail" not in json.loads((tmp_path / "out" / "stocks" / "SMALL.json").read_text())
    assert json.loads((tmp_path / "out" / "market.json").read_text())["breadth"]["n"] == 1
    assert sizes["market.json"] > 0
