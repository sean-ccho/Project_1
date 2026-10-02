"""쉬운 방법 비교 기준(benchmarks) 테스트."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from paper_trading.benchmarks import benchmark_summary, momentum_topn_returns


def _trend_closes(n: int = 320) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({
        "SPY": [100 * 1.001 ** i for i in range(n)],
        "UP": [100 * 1.002 ** i for i in range(n)],
        "DOWN": [100 * 0.999 ** i for i in range(n)],
        "FLAT": [100.0] * n,
    }, index=idx)


def test_momentum_picks_strongest_with_one_day_lag():
    closes = _trend_closes()
    dates = list(closes.index[260:])
    r = momentum_topn_returns(closes, dates, top_n=1, cost_per_side=0.0)
    assert r.iloc[0] == 0.0 and r.iloc[1] == 0.0  # 첫날 신호 → 다음날 종가 체결
    assert r.iloc[5] == pytest.approx(0.002)  # 그 뒤로 UP 수익률


def test_momentum_cost_reduces_return():
    closes = _trend_closes()
    dates = list(closes.index[260:])
    free = momentum_topn_returns(closes, dates, top_n=2, cost_per_side=0.0)
    paid = momentum_topn_returns(closes, dates, top_n=2, cost_per_side=0.001)
    assert paid.sum() < free.sum()
    assert paid.iloc[1] == pytest.approx(-0.001)  # 첫 매수 회전율 1.0 × 10bp


def test_momentum_respects_membership():
    closes = _trend_closes()
    dates = list(closes.index[260:])
    r = momentum_topn_returns(closes, dates, top_n=1, cost_per_side=0.0,
                              members_on=lambda d: frozenset({"FLAT"}))
    assert r.abs().max() == 0.0


def test_benchmark_summary_spy_clone_has_no_alpha():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2023-01-02", periods=420)
    closes = pd.DataFrame(
        {t: 100 * np.cumprod(1 + rng.normal(0.0003 * (i + 1), 0.01, len(idx)))
         for i, t in enumerate(["SPY", "A", "B", "C", "D"])},
        index=idx,
    )
    equity = 5000 * closes["SPY"].iloc[260:] / closes["SPY"].iloc[260]
    out = benchmark_summary(equity, closes, top_n=2, cost_per_side=0.0)
    assert out["베타_시장"] == pytest.approx(1.0, abs=0.01)
    assert abs(out["알파_연"]) < 1e-3
    assert out["기준_모멘텀_N"] == 2
    assert "모멘텀대비_판정" in out


def test_benchmark_summary_needs_enough_data():
    closes = _trend_closes(100)
    assert benchmark_summary(closes["SPY"].iloc[:30], closes) == {}
