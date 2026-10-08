"""scripts/pattern_confirm_research.py 테스트 — patterns.py 와 같은 판정인지 비교."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from pattern_confirm_research import candle_flags, cooldown_mask, trend_context
from screener.patterns import detect_candlestick_patterns, get_trend_context


def _random_ohlc(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = c * np.exp(rng.normal(0, 0.01, n))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.01, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.01, n))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1e6}, index=idx)


def test_matches_screener_patterns():
    df = _random_ohlc(400, 0)
    one = lambda col: df[[col]].rename(columns={col: "X"})
    flags = candle_flags(one("Open"), one("High"), one("Low"), one("Close"))
    trend = trend_context(one("Close"))["X"]
    names = {"1": "bullish_engulfing", "2": "morning_star"}
    for i in range(30, len(df)):
        win = df.iloc[: i + 1]
        tc = {"uptrend": "up", "downtrend": "down", "sideways": "side"}[get_trend_context(win)]
        assert trend.iloc[i] == tc
        res = detect_candlestick_patterns(win)
        found = set(res.pattern_type.split(",")) if res.detected else set()
        for k, name in names.items():
            assert bool(flags[k]["X"].iloc[i]) == (name in found), (i, name)
        assert bool(flags["3"]["X"].iloc[i]) == ("doji" in found and tc == "down"), i


def test_cooldown():
    f = pd.DataFrame({"A": [True, True, False, True] + [False] * 8 + [True]})
    assert cooldown_mask(f, 10)["A"].tolist() == [True] + [False] * 11 + [True]
