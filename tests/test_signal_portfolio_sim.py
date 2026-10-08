"""scripts/signal_portfolio_sim.py 테스트 — 겹치는 보유 비중·비용·진입 시점."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_portfolio_sim import open_to_open, simulate


def _ohlcv(n: int = 10) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=n)
    opens = pd.DataFrame({"AAA": 100 * 1.01 ** np.arange(n), "BBB": np.full(n, 50.0)}, index=idx)
    opens.columns = pd.MultiIndex.from_product([["Open"], opens.columns], names=["Price", "Ticker"])
    return opens


def test_open_to_open():
    r = open_to_open(_ohlcv())
    assert math.isclose(r["AAA"].iloc[0], 0.01)
    assert math.isnan(r["AAA"].iloc[-1])


def test_simulate_single_signal_weights_and_cost():
    r = open_to_open(_ohlcv())
    d0 = r.index[0]
    sig = pd.DataFrame({"date": [d0, d0], "티커": ["AAA", "BBB"], "score": [2.0, 1.0]})
    out = simulate(sig, r, k=1, hold=3, cost=0.001)
    # d0 신호 → d1 시가 진입, d1~d3 보유(비중 1/3), d4 시가 청산. 점수 높은 AAA만
    assert out.iloc[0] == 0.0
    assert math.isclose(out.iloc[1], 0.01 / 3 - 0.001 / 3)
    assert math.isclose(out.iloc[2], 0.01 / 3)
    assert math.isclose(out.iloc[4], -0.001 / 3)
    assert math.isclose(out.sum(), 0.01 - 2 * 0.001 / 3)


def test_simulate_skips_signal_without_room():
    r = open_to_open(_ohlcv(5))
    sig = pd.DataFrame({"date": [r.index[3]], "티커": ["AAA"], "score": [1.0]})
    assert simulate(sig, r, k=1, hold=3).abs().sum() == 0.0
