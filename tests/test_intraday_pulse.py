"""장중 펄스 (web_export.intraday_pulse) — 네트워크 없이. 핵심은 '밤과 같은 계산에 오늘 값 하나를 넣은 것'이 맞는지."""
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from web_export.intraday_pulse import (ET, VOLUME_CURVE, Bar, base_row, heat_flags, live_quote,  # noqa: E402
                                       market_live, market_session, pt1_live, quote_list, run, vol_ratio,
                                       volume_fraction)
from web_export.market_summary import price_stats  # noqa: E402


def _ohlcv(n: int = 320, seed: int = 3) -> pd.DataFrame:
    """배당 없는 일봉 (Adj Close = Close) — 분할만 반영 가격과 배당 조정 가격이 같다."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    days = pd.bdate_range("2025-06-02", periods=n)
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Adj Close": close,
                         "Volume": rng.integers(1_000_000, 3_000_000, n).astype(float)}, index=days)


def test_volume_fraction_follows_curve_and_skips_first_minutes():
    assert volume_fraction(10) is None                      # 개장 15분 안
    assert volume_fraction(60) == pytest.approx(0.147)
    assert volume_fraction(390) == pytest.approx(0.811)
    assert volume_fraction(500) == pytest.approx(0.811)     # 마감 뒤도 곡선 끝
    assert volume_fraction(67.5) == pytest.approx((0.147 + 0.177) / 2)
    # 조기 폐장(13:00, 210분) — 105분째 = 정규장 195분째
    assert volume_fraction(105, 210) == pytest.approx(dict(VOLUME_CURVE)[195])
    xs = [volume_fraction(m) for m in range(15, 391, 5)]
    assert all(b >= a for a, b in zip(xs, xs[1:]))


def test_vol_ratio_equals_nightly_formula_when_day_is_done():
    vol = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 40.0]) * 1e6
    nightly = vol[-1] / vol[-20:].mean()                    # features: 오늘 ÷ 오늘 포함 20일 평균
    assert vol_ratio(vol[-1], 1.0, vol[:-1].sum()) == pytest.approx(nightly)
    # 장 중간 — 지금까지 거래량을 곡선으로 늘린 하루치로 같은 식
    assert vol_ratio(vol[-1] * 0.5, 0.5, vol[:-1].sum()) == pytest.approx(nightly)
    assert vol_ratio(0, 0.5, 1e6) is None
    assert vol_ratio(1e6, None, 1e6) is None


def test_base_plus_today_matches_price_stats_on_full_history():
    df = _ohlcv()
    for cut in (260, 290, 319):                             # 여러 날을 '오늘'로
        hist = df.iloc[:cut + 1]
        row = base_row(hist.iloc[:-1])
        t = hist.iloc[-1]
        q = live_quote(row, Bar(price=t["Close"], high=t["High"], low=t["Low"], volume=t["Volume"],
                                prev_close=hist["Close"].iloc[-2]), 1.0)
        night = price_stats(hist.drop(columns=["Adj Close"]))
        assert q["c"] == pytest.approx(night["ret_1d"])
        assert q["a50"] == night["above50"]
        assert q["a200"] == night["above200"]
        assert q["nh"] == night["new_high"]
        assert q["nl"] == night["new_low"]
        assert (q["lo"], q["hi"]) == pytest.approx((night["range_52w"]["low"], night["range_52w"]["high"]), rel=1e-4)


def test_base_row_short_history_leaves_moving_averages_empty():
    row = base_row(_ohlcv(60))
    assert row[1] is not None and row[2] is None            # 49일 합은 있고 199일 합은 없다
    assert row[6] == "2025-08-22"


def test_live_quote_without_base_has_price_and_change_only():
    q = live_quote(None, Bar(price=11.0, high=11.5, low=10.5, volume=1e6, prev_close=10.0), 0.5)
    assert q["p"] == 11.0 and q["c"] == pytest.approx(0.1)
    assert q["v"] is None and q["nh"] is None and q["a50"] is None


def test_quote_list_packs_flags():
    out = quote_list({"A": {"p": 10.123, "c": 0.01234, "v": 1.234, "lo": 5, "hi": 11, "nh": True, "nl": False},
                      "B": {"p": 0.12345, "c": None, "v": None, "lo": None, "hi": None, "nh": None, "nl": True}})
    assert out["A"] == [10.12, 0.0123, 1.23, 5, 11, 1]
    assert out["B"] == [0.1235, None, None, None, None, 2]


@pytest.mark.parametrize("when,market,state,active", [
    (datetime(2026, 11, 9, 9, 0), "us", "pre", False),
    (datetime(2026, 11, 9, 11, 0), "us", "open", True),
    (datetime(2026, 11, 9, 16, 20), "us", "closed", True),     # 마감 펄스
    (datetime(2026, 11, 9, 17, 0), "us", "closed", False),
    (datetime(2026, 11, 14, 11, 0), "us", "holiday", False),   # 토요일
    (datetime(2026, 11, 26, 11, 0), "us", "holiday", False),   # 미국 추수감사절
    (datetime(2026, 11, 26, 11, 0), "ca", "open", True),       # TSX 는 연다
    (datetime(2026, 10, 12, 11, 0), "ca", "holiday", False),   # 캐나다 추수감사절
    (datetime(2026, 11, 27, 13, 30), "us", "closed", True),    # 조기 폐장 13:00
])
def test_market_session(when, market, state, active):
    s = market_session(market, when.replace(tzinfo=ET))
    assert s.state == state
    assert s.active(when.replace(tzinfo=ET)) is active


def test_early_close_session_length():
    s = market_session("us", datetime(2026, 11, 27, 11, 0, tzinfo=ET))
    assert s.minutes == 210


def test_heat_flags_use_pt1_hard_filter_limits():
    def keys(*a):
        return [h["k"] for h in heat_flags(*a)]

    row = {"RSI": 76.0, "bollinger_pband": 0.5, "5일수익률": 0.02, "52주포지션": 0.5}
    assert keys(row, "모멘텀", "bull") == ["rsi"]
    assert keys({**row, "RSI": 71.0}, "모멘텀", "neutral") == []
    assert keys({**row, "RSI": 71.0}, "모멘텀", "bear") == ["rsi"]                # 약세장 70
    assert keys({**row, "RSI": 50.0, "bollinger_pband": 0.96, "5일수익률": 0.16}, "모멘텀", None) == ["bband", "ret5"]
    assert keys({**row, "RSI": 50.0, "52주포지션": 0.7}, "바닥반등", None) == ["bottom_pos"]
    assert keys({**row, "RSI": 32.0}, "바닥반등", None) == ["bottom_rsi"]
    assert keys({**row, "RSI": 32.0}, "모멘텀", None) == []                        # 데드존은 바닥반등만
    assert heat_flags(row, "모멘텀", None) == [{"k": "rsi", "v": 76.0, "lim": 75.0}]
    assert heat_flags({**row, "RSI": 71.0}, "모멘텀", "bear")[0]["lim"] == 70.0


def test_market_live_counts_breadth_and_drops_calendar():
    meta = {"A": ["A Inc", "Technology", "sp500", 3e9, 10.0, None],
            "B": ["B Inc", "Energy", "sp500", 3e9, 20.0, None],
            "C": ["C Inc", "Energy", "nasdaq", 5e9, 30.0, "abc"]}
    quotes = {"A": {"p": 11.0, "c": 0.1, "v": 2.0, "a50": True, "a200": True, "nh": True, "nl": False},
              "B": {"p": 19.0, "c": -0.05, "v": 0.5, "a50": False, "a200": True, "nh": False, "nl": False},
              "C": {"p": 33.0, "c": 0.1, "v": 3.0, "a50": True, "a200": False, "nh": False, "nl": False}}
    m = market_live(quotes, meta, ["A", "B"], ["A", "B", "C"], "2026-11-09", 0.01, "2026-11-09T11:00:00-05:00")
    b = m["breadth"]
    assert (b["n"], b["adv"], b["dec"], b["new_high"], b["above50"]) == (2, 1, 1, 1, 0.5)
    assert "calendar" not in m and "mega" not in m
    assert [g["t"] for g in m["volume"]][:1] == ["C"]
    assert m["gainers"][0]["cap"] == pytest.approx(5e9 * 33 / 30)    # 시총도 지금 가격으로


def test_pt1_live_reprices_account_and_uses_engine_lines():
    dash = {
        "price_date": "2026-11-06",
        "benchmark": {"closes": [["2026-11-05", 700.0], ["2026-11-06", 710.0]]},
        "accounts": [{"key": "pt1", "positions": [
            {"ticker": "XOM", "entry_date": "2026-11-02", "entry_price": 100.0, "highest_price": 104.0, "price": 103.0,
             "shares": 10.0, "spy_return": 0.01, "strategy": "모멘텀"}],
            "dollar": {"initial": 5000.0, "cash": 4000.0, "equity": 5030.0, "spy_equity": 5100.0}}],
    }
    positions = [{"ticker": "XOM", "entry_date": "2026-11-02", "entry_price": 100.0, "highest_price": 104.0,
                  "strategy": "📈 모멘텀", "defer_count": 0}]
    quotes = {"XOM": {"p": 110.0, "c": 0.0679}, "SPY": {"p": 717.1}}
    out = pt1_live(dash, positions, quotes, date(2026, 11, 9))
    p = out["positions"][0]
    assert p["price"] == 110.0 and p["return"] == pytest.approx(0.10)
    assert p["highest_price"] == 110.0                                   # 고점 = max(기록, 지금 가격) — 잠정
    assert positions[0]["highest_price"] == 104.0                        # 상태는 안 바꾼다
    assert out["dollar"]["equity"] == pytest.approx(4000 + 10 * 110)
    assert out["dollar"]["spy_equity"] == pytest.approx(5100 * 717.1 / 710.0)
    assert p["spy_return"] == pytest.approx(1.01 * 717.1 / 710.0 - 1, abs=1e-4)
    from paper_trading.engine import exit_levels
    lv = exit_levels({**positions[0], "highest_price": 110.0}, as_of=date(2026, 11, 9))
    assert p["exit"]["down"] == pytest.approx(lv["down"]) and p["exit"]["down_tag"] == lv["down_tag"]


def test_load_base_only_reuses_same_day_and_scope(tmp_path):
    import gzip
    import json

    from web_export.intraday_pulse import SCHEMA_VERSION, load_base

    p = tmp_path / "base.json.gz"
    p.write_bytes(gzip.compress(json.dumps({"schema": SCHEMA_VERSION, "for_day": "2026-11-09", "scope": "all",
                                            "t": {}}).encode()))
    assert load_base(p, date(2026, 11, 9), "all") is not None
    assert load_base(p, date(2026, 11, 9), "cover") is None      # 범위가 바뀌면 다시 만든다
    assert load_base(p, date(2026, 11, 10), "all") is None       # 다음 날은 다시 만든다
    assert load_base(tmp_path / "none.json.gz", date(2026, 11, 9)) is None


def test_run_does_nothing_when_markets_are_closed(tmp_path):
    saturday = datetime(2026, 11, 14, 11, 0, tzinfo=ET)
    assert run(tmp_path, now=saturday) == 3
    assert not any(tmp_path.iterdir())
