"""scripts/value_v3pm_backtest.py — v3 PM 10칸 · 비중 수익률 (2026-11-16)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import value_v3pm_backtest as pm  # noqa: E402
import value_v3_backtest as vb  # noqa: E402


def test_slots_fill_spy():
    assert pm.slots(["A", "B"]) == {"A": 0.1, "B": 0.1, "SPY": pytest.approx(0.8)}
    assert pm.slots([]) == {"SPY": 1.0}
    assert pm.slots([f"T{i}" for i in range(12)]) == {f"T{i}": 0.1 for i in range(10)}
    assert pm.equal([]) == {"SPY": 1.0}


def test_run_weighted_matches_run_series_for_equal_weights():
    p1 = {"A": np.array([1.1, 1.21]), "B": np.array([1.0, 1.0]), "SPY": np.array([1.01, 1.02])}
    p2 = {"A": np.array([1.0, 1.0]), "C": np.array([0.9, 0.9]), "SPY": np.array([1.0, 1.01])}
    m1, per1 = pm.run_weighted([pm.equal(["A", "B"]), pm.equal(["A", "C"])], [p1, p2], cost=0.01)
    m2, per2 = vb.run_series([["A", "B"], ["A", "C"]], [p1, p2], cost=0.01)
    assert m1 == pytest.approx(m2) and per1 == pytest.approx(per2)
    m, _ = pm.run_weighted([pm.slots(["A"])], [p1], cost=0.0)
    assert m[0] == pytest.approx(0.1 * 0.1 + 0.9 * 0.01)


def test_add_pm_rows_puts_sec_shares_on_same_years():
    rows = vb.index_rows([("2018-12-31", "2019-02-15", 110.0, 0), ("2019-12-31", "2020-02-15", 100.0, 0)])
    ent = {"inc": {"Total Revenue": {"2018-12-31": 1.0, "2019-12-31": 1.0}}, "bal": {}, "cf": {}}
    out = pm.add_pm_rows(ent, {"rows": {"shares": rows}}, "2020-05-31")
    assert out["inc"]["Diluted Average Shares"] == {"2018-12-31": 110.0, "2019-12-31": 100.0}
    assert out["rows_v"] == pm.vm.ROWS_VERSION
