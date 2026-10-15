"""연구소 실행 테스트 — 사전 등록 문구, 고원 선택·느린 주기, 이웃 비율, 통과 조건, 결합 비중."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lab_families as lf  # noqa: E402
import lab_search as ls  # noqa: E402


def test_spec_fixes_rules_before_results():
    for c in "ABCDEF":
        s = ls.spec(c)
        assert f"lab {c}" in s
    for c in "ABCDE":
        s = ls.spec(c)
        assert "PBO" in s and "1.5배" in s and "0.05 이내" in s and "하루 지연" in s and "2008-01~2015-12" in s
    assert "해외 6개" in ls.spec("A") and "해외 6개" in ls.spec("E") and "해외" not in ls.spec("B")
    assert ls.FAMILY_IDS["A"] == "lab_A_v1" and ls.PBO_MAX == 0.3 and ls.LEV_CAP == 1.5 and ls.INTL_MIN == 4


def test_select_takes_plateau_and_slower_schedule():
    fam = lf.FAMILIES["D"]
    grid = fam.grid()
    keys = [fam.key(c) for c in grid]
    scores = np.zeros(len(grid))
    sharpe = np.zeros(len(grid))
    spike = keys.index(fam.key({"menu": "G4", "wk": "IV", "wl": 21, "sched": "D2"}))
    scores[spike] = 2.0                                                     # 혼자 튀는 값 (이웃은 0)
    for k, s in enumerate(lf.SCHEDULE_ORDER):                               # 넓은 고원: AW·IV 126/252 의 모든 주기
        scores[keys.index(fam.key({"menu": "AW", "wk": "IV", "wl": 126, "sched": s}))] = 1.0
        scores[keys.index(fam.key({"menu": "AW", "wk": "IV", "wl": 252, "sched": s}))] = 1.0 + 0.01 * k
    for s, v in zip(lf.SCHEDULE_ORDER, (0.60, 0.70, 0.73, 0.75, 0.76)):
        sharpe[keys.index(fam.key({"menu": "AW", "wk": "IV", "wl": 252, "sched": s}))] = v
    smooth = ls.smoothed_scores(fam, grid, scores)
    chosen, best = ls.select(fam, grid, sharpe, smooth)
    assert grid[best] == {"menu": "AW", "wk": "IV", "wl": 252, "sched": "D2"}    # 튀는 값이 아니라 고원
    assert grid[chosen] == {"menu": "AW", "wk": "IV", "wl": 252, "sched": "D10"}  # Sharpe 0.05 이내 중 가장 느린 주기


def test_neighbor_share():
    fam = lf.FAMILIES["E"]
    grid = fam.grid()
    sharpe = np.array([0.6 if c["k"] <= 2 else 0.4 for c in grid])
    i = [fam.key(c) for c in grid].index(fam.key({"k": 2, "j": 3}))
    share, n = ls.neighbor_share(fam, grid, sharpe, i, 0.5)
    assert n == 4 and share == pytest.approx(0.75)                          # k=1·j=2·j=4 는 높고 k=3 은 낮다


def test_gate_requires_sharpe_cagr_and_mdd():
    pb = {"Sharpe": 0.5, "CAGR": 0.10, "MDD": -0.30}
    assert ls.gate_ok({"Sharpe": 0.6}, {"CAGR": 0.11, "MDD": -0.25}, pb)
    assert not ls.gate_ok({"Sharpe": 0.6}, {"CAGR": 0.09, "MDD": -0.25}, pb)
    assert not ls.gate_ok({"Sharpe": 0.6}, {"CAGR": 0.11, "MDD": -0.35}, pb)
    assert not ls.gate_ok({"Sharpe": 0.5}, {"CAGR": 0.11, "MDD": -0.25}, pb)


def test_combo_target_mixes_sleeves():
    idx = pd.bdate_range("2000-01-03", periods=600)
    rng = np.random.default_rng(3)
    cols = ["US", "DEV", "EM", "LTT", "ITT", "REIT", "GOLD", "CMDTY"]
    u = lf.Universe(pd.DataFrame(rng.normal(0.0003, 0.01, (600, 8)), index=idx, columns=cols),
                    pd.Series(0.0, index=idx))
    sleeves = {"D": {"menu": "SB2", "wk": "EW", "wl": 0, "sched": "M"},
               "A": {"trend": "none", "n": 0, "sigma": 0.0, "vl": 0, "sched": "W"}}
    tgt, sched = ls.combo_target(None, sleeves, u, "EW")
    assert sched == "W"                                                     # 군 중 가장 빠른 주기
    row = tgt.dropna(how="all").iloc[-1]
    assert row["US"] == pytest.approx(0.5 * 0.5 + 0.5 * 1.0) and row["ITT"] == pytest.approx(0.25)
    assert row.sum() == pytest.approx(1.0)
