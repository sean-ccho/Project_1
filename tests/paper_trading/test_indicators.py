"""순수 파이썬 지표·골든크로스 추출 테스트."""

import math

from paper_trading.golden_cross import extract_golden_cross
from paper_trading.indicators import (
    PercentileRanker,
    adx_series,
    atr_series,
    bollinger_pband,
    compute_indicators,
    ema_series,
    num,
    rsi_series,
)


def test_ema_constant_series():
    out = ema_series([5.0] * 30, 10)
    assert all(math.isnan(v) for v in out[:9])
    assert all(abs(v - 5.0) < 1e-12 for v in out[9:])


def test_rsi_extremes():
    assert rsi_series([float(i) for i in range(1, 31)])[-1] == 100.0
    assert rsi_series([float(i) for i in range(30, 0, -1)])[-1] == 0.0
    assert rsi_series([10.0] * 30)[-1] == 50.0


def test_atr_constant_range():
    lows = [float(i) for i in range(40)]
    highs = [lo + 2 for lo in lows]
    closes = [lo + 1 for lo in lows]
    assert abs(atr_series(highs, lows, closes, 14)[-1] - 2.0) < 1e-9


def test_adx_strong_uptrend():
    lows = [float(i) for i in range(60)]
    highs = [lo + 2 for lo in lows]
    closes = [lo + 1.5 for lo in lows]
    adx = adx_series(highs, lows, closes, 14)
    assert math.isnan(adx[26]) and not math.isnan(adx[27])
    assert 25 < adx[-1] <= 100


def test_bollinger_pband():
    assert bollinger_pband([10.0] * 20) == 0.5
    assert bollinger_pband([10.0] * 19 + [12.0]) > 0.8
    assert math.isnan(bollinger_pband([10.0] * 5))


def test_compute_indicators_below_ema50_streak():
    closes = [100.0 + i for i in range(60)] + [100.0, 99.0, 98.0]
    bars = [{"date": str(i), "open": c, "high": c + 1, "low": c - 1, "close": c} for i, c in enumerate(closes)]
    ind = compute_indicators(bars)
    assert ind["below_ema50_streak"] == 3.0
    assert ind["close"] == 98.0 and ind["ema20"] > ind["close"]


def test_num_and_percentile_ranker():
    rows = [{"x": v} for v in (1, 2, 3, 4)] + [{"x": "nan"}, {"x": None}]
    assert num({"x": "1.5"}, "x") == 1.5
    assert math.isnan(num({"x": "nan"}, "x"))
    r = PercentileRanker(rows, ["x"])
    assert r.rank("x", {"x": 4}) == 0.875
    assert r.rank("x", {"x": 1}) == 0.125
    assert r.rank("x", {"x": None}) == 0.5


def _gc_row(ticker: str, **kw) -> dict:
    base = {"티커": ticker, "섹터": "Tech", "현재가격": 100.0, "일봉패턴": "", "주봉패턴": "", "월봉패턴": "",
            "ema_gap_20_50": 0.0, "주봉_MA갭": 0.0, "월봉_MA갭": 0.0}
    base.update(kw)
    return base


def test_extract_golden_cross_rules_and_order():
    rows = [
        _gc_row("ONE", 일봉패턴="골든크로스, 상승삼각형", ema_gap_20_50=0.01),
        _gc_row("FAR", 일봉패턴="골든크로스", ema_gap_20_50=0.2),
        _gc_row("IMM", 주봉패턴="골든크로스임박", 주봉_MA갭=-0.02),
        _gc_row("TWO", 일봉패턴="골든크로스", ema_gap_20_50=0.03, 주봉패턴="골든크로스임박", 주봉_MA갭=-0.04),
    ]
    out = extract_golden_cross(rows)
    assert [g["ticker"] for g in out] == ["TWO", "ONE", "IMM"]
    two = out[0]
    assert two["tf_count"] == 2 and two["has_daily"] is True
    assert two["tf_gaps"] == [("일봉", 0.03), ("주봉", -0.04)]


def test_extract_golden_cross_empty_inputs():
    assert extract_golden_cross(None) == []
    assert extract_golden_cross([]) == []
    assert extract_golden_cross([{"x": 1}]) == []
