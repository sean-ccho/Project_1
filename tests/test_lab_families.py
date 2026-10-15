"""연구소 전략군 테스트 — 격자·이웃, 미래 정보 없음, 쓸 수 있는 자산, 대피 자산, 달력 창."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lab_families as lf  # noqa: E402


def _multi(n: int = 900, seed: int = 0) -> lf.Universe:
    idx = pd.bdate_range("2000-01-03", periods=n)
    rng = np.random.default_rng(seed)
    cols = ["US", "DEV", "EM", "LTT", "ITT", "REIT", "GOLD", "CMDTY"]
    ret = pd.DataFrame(rng.normal(0.0003, 0.01, (n, len(cols))), index=idx, columns=cols)
    ret.loc[: idx[300], "GOLD"] = np.nan                                      # 금은 나중에 생긴다
    return lf.Universe(ret, pd.Series(0.0001, index=idx))


def _index(n: int = 900, seed: int = 1) -> lf.Universe:
    idx = pd.bdate_range("2000-01-03", periods=n)
    rng = np.random.default_rng(seed)
    return lf.Universe(pd.DataFrame({"EQ": rng.normal(0.0004, 0.01, n)}, index=idx), pd.Series(0.0001, index=idx))


def test_grid_sizes():
    assert lf.grid_sizes() == {"A": 960, "B": 320, "C": 360, "D": 100, "E": 25}


def test_neighbors_move_one_knob_one_step():
    fam = lf.FAMILIES["A"]
    keys = {fam.key(c) for c in fam.grid()}
    cfg = {"trend": "sma", "n": 200, "sigma": 0.15, "vl": 63, "sched": "W"}
    nb = fam.neighbors(cfg, keys)
    assert len(nb) == 8 and fam.key({**cfg, "n": 250}) in nb and fam.key({**cfg, "sched": "D10"}) in nb
    none = {"trend": "none", "n": 0, "sigma": 0.0, "vl": 0, "sched": "M"}
    assert fam.neighbors(none, keys) == [fam.key({**none, "sched": "W"})]   # 사다리 끝 · 범주형은 이웃 아님
    e = lf.FAMILIES["E"]
    assert len(e.neighbors({"k": 3, "j": 3}, {e.key(c) for c in e.grid()})) == 4


@pytest.mark.parametrize("code", ["A", "B", "C", "D", "E"])
def test_no_lookahead_in_targets(code):
    fam = lf.FAMILIES[code]
    u = _index() if fam.kind == "index" else _multi()
    cut = u.ret.index[650]
    if code == "E":                                                          # 달력 창은 거래 달력을 미리 안다 → 월말에서 자른다
        cut = u.ret.index[u.ret.index.month != u.ret.index[651].month][-1]
    g = fam.grid()
    for cfg in g[:: max(1, len(g) // 12)]:
        full = fam.target(u, cfg).loc[:cut]
        part = fam.target(u.truncate(cut), cfg)
        pd.testing.assert_frame_equal(full.iloc[:-2] if code == "E" else full,
                                      part.iloc[:-2] if code == "E" else part, check_freq=False)


def test_eligibility_and_base_weights():
    u = _multi()
    assert not u.eligible["GOLD"].iloc[400] and u.eligible["GOLD"].iloc[600]
    b = lf.base_weights(u, lf.MENUS["G8"], "EW", 0)
    assert b.iloc[400]["GOLD"] == 0 and b.iloc[400].sum() == pytest.approx(1.0)
    assert b.iloc[700]["GOLD"] == pytest.approx(1 / 8)
    assert b.iloc[100].isna().all()                                           # 아무 자산도 기록 253일이 안 됨
    iv = lf.base_weights(u, lf.MENUS["G4"], "IV", 63)
    v = u.vol(63)[lf.MENUS["G4"]].iloc[700]
    assert iv.iloc[700].to_numpy() == pytest.approx((1 / v / (1 / v).sum()).to_numpy())


def test_dual_momentum_sends_negative_slots_to_safe_asset():
    idx = pd.bdate_range("2000-01-03", periods=300)
    ret = pd.DataFrame({"US": 0.001, "DEV": -0.001, "EM": 0.0005, "ITT": 0.0002, "LTT": 0.0, "REIT": 0.0,
                        "GOLD": 0.0, "CMDTY": 0.0}, index=idx)
    u = lf.Universe(ret, pd.Series(0.0, index=idx))
    t = lf.FAMILIES["C"].target(u, {"menu": "R3", "safe": "ITT", "score": "mom", "n": 63, "k": 2, "sched": "M"})
    last = t.iloc[-1]
    assert last["US"] == 0.5 and last["EM"] == 0.5 and last["ITT"] == 0
    t3 = lf.FAMILIES["C"].target(u, {"menu": "R3", "safe": "ITT", "score": "mom", "n": 63, "k": 3, "sched": "M"})
    assert t3.iloc[-1]["DEV"] == 0 and t3.iloc[-1]["ITT"] == pytest.approx(1 / 3)
    tc = lf.FAMILIES["C"].target(u, {"menu": "R3", "safe": "CASH", "score": "mom", "n": 63, "k": 3, "sched": "M"})
    assert tc.iloc[-1].sum() == pytest.approx(2 / 3)                           # 나머지는 현금


def test_month_window_and_e_target_shift():
    idx = pd.bdate_range("2024-01-01", "2024-02-29")
    w = lf.month_window(idx, 2, 3)
    assert list(idx[w].strftime("%m-%d")) == ["01-01", "01-02", "01-03", "01-30", "01-31", "02-01", "02-02", "02-05",
                                              "02-28", "02-29"]
    u = lf.Universe(pd.DataFrame({"EQ": 0.0}, index=pd.bdate_range("2023-01-02", "2024-02-09")),
                    pd.Series(0.0, index=pd.bdate_range("2023-01-02", "2024-02-09")))
    t = lf.FAMILIES["E"].target(u, {"k": 2, "j": 3})["EQ"]
    assert t.loc["2024-01-26"] == 1.0 and t.loc["2024-01-25"] == 0.0         # 1/26 결정 → 1/29 종가 체결 → 1/30 수익부터
    assert np.isnan(t.iloc[-1])


def test_trend_signals():
    u = _index()
    s = lf.trend_signal(u, "EQ", "mom", 126)
    m = u.mom(126)["EQ"]
    assert ((s == 1.0) == (m > 0)).all()
    x = lf.trend_signal(u, "EQ", "xma", "20/100")
    assert ((x == 1.0) == (u.sma(20)["EQ"] > u.sma(100)["EQ"])).all()
    t = lf.FAMILIES["A"].target(u, {"trend": "none", "n": 0, "sigma": 0.10, "vl": 21, "sched": "M"})["EQ"]
    v = u.vol(21)["EQ"]
    i = 500
    assert t.iloc[i] == pytest.approx(min(1.0, 0.10 / v.iloc[i]))
