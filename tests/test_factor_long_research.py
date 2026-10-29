"""factor_long_research: French CSV 파싱 · 부트스트랩 · 해석적 z · CAPM 알파 점검."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import factor_long_research as fl  # noqa: E402

SAMPLE = """This file was created by CMPT_ME_PRIOR_RETS using the 202509 CRSP database.
It contains value- and equal-weighted returns for 2x3 portfolios.

  Average Value Weighted Returns -- Monthly
,SMALL LoPRIOR,BIG HiPRIOR
192701,   1.50,  -0.25
192702,  -99.99,   2.00

  Average Equal Weighted Returns -- Monthly
,SMALL LoPRIOR,BIG HiPRIOR
192701,   9.00,   9.00
192702,   9.00,   9.00

  Average Value Weighted Returns -- Annual
,SMALL LoPRIOR,BIG HiPRIOR
1927,  10.00,  20.00
"""

FACTORS = """This file was created using the 202509 CRSP database.
,Mkt-RF,SMB,HML,RF
192607,    2.89,   -2.55,   -2.39,    0.22
192608,    2.64,   -1.14,    3.81,    0.25

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
1927,   29.47,   -2.46,   -3.75,    3.12
"""


def test_parse_picks_value_weighted_monthly_and_scales() -> None:
    tables = fl.parse_tables(SAMPLE)
    assert len(tables) == 2                       # 연간 표는 월간 행이 없어 빠진다
    vw = fl.pick_table(tables, factor_file=False)
    assert list(vw.columns) == ["SMALL LoPRIOR", "BIG HiPRIOR"]
    assert vw.loc[pd.Period("1927-01", "M"), "BIG HiPRIOR"] == -0.0025
    assert np.isnan(vw.loc[pd.Period("1927-02", "M"), "SMALL LoPRIOR"])   # -99.99 → NaN
    assert vw.loc[pd.Period("1927-02", "M"), "BIG HiPRIOR"] == 0.02


def test_parse_factor_file_first_monthly_table() -> None:
    ff = fl.pick_table(fl.parse_tables(FACTORS), factor_file=True)
    assert list(ff.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(ff) == 2
    assert abs(ff.loc[pd.Period("1926-07", "M"), "RF"] - 0.0022) < 1e-12


def test_boot_identical_series_has_zero_difference() -> None:
    x = np.random.default_rng(0).normal(0.005, 0.04, 240)
    b = fl.boot_dsharpe(x, x, n_boot=500)
    assert b["dS"] == 0.0 and b["lo"] == 0.0 and b["hi"] == 0.0 and b["p"] == 1.0


def test_boot_detects_large_edge() -> None:
    rng = np.random.default_rng(1)
    m = rng.normal(0.005, 0.045, 720)
    a = m + 0.006 + rng.normal(0.0, 0.01, 720)    # 월 +0.6% 알파, 추적오차 작음
    b = fl.boot_dsharpe(a, m, n_boot=2000)
    assert b["dS"] > 0.3 and b["lo"] > 0 and b["p"] < 0.01


def test_memmel_z_is_antisymmetric() -> None:
    rng = np.random.default_rng(2)
    a, b = rng.normal(0.01, 0.05, 120), rng.normal(0.005, 0.04, 120)
    assert abs(fl.memmel_z(a, b) + fl.memmel_z(b, a)) < 1e-12


def test_capm_recovers_alpha_and_beta() -> None:
    rng = np.random.default_rng(3)
    m = rng.normal(0.006, 0.045, 1200)
    a = 0.002 + 0.8 * m + rng.normal(0.0, 0.005, 1200)
    alpha, t, beta = fl.capm(a, m)
    assert abs(alpha - 0.024) < 0.006 and abs(beta - 0.8) < 0.02 and t > 5


def test_breakeven_cost_zeroes_sharpe_gap() -> None:
    rng = np.random.default_rng(4)
    idx = pd.period_range("1990-01", periods=360, freq="M")
    mkt = rng.normal(0.008, 0.045, 360)
    gross = mkt + 0.003 + rng.normal(0.0, 0.02, 360)
    rf = np.full(360, 0.002)
    df = pd.DataFrame({"a": gross, "mkt": mkt, "rf": rf}, index=idx)
    c = fl.breakeven_cost(df, cost=0.0)
    a_net = gross - c / 12 - rf
    assert abs(fl.sharpe_m(a_net) - fl.sharpe_m(mkt - rf)) < 1e-9
