"""shadow_track — 다음날 시가 진입 수익률, TOP5 − SPY, 월말 상위 20% − 평균."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import shadow_track as st  # noqa: E402


def _opens(n: int = 120) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-01", periods=n)
    g = np.arange(n, dtype=float)
    return pd.DataFrame({"SPY": 100 * 1.001 ** g, "IWM": 100 * 1.001 ** g,
                         "AAA": 100 * 1.003 ** g, "BBB": 100 * 1.001 ** g, "CCC": 100 * 0.999 ** g}, index=idx)


def test_fwd_enters_next_session_and_waits_until_done():
    o = _opens(30)
    r = st.fwd(o, o.index[0], 21)
    assert np.isclose(r["AAA"], 1.003 ** 21 - 1)        # 기록일 다음 거래일 시가 → 21거래일 뒤
    assert st.fwd(o, o.index[-5], 21) is None              # 아직 안 끝난 기록은 빠진다


def test_evaluate_top_and_cross_section():
    o = _opens()
    days = []
    for d in o.index[:40]:
        days += [{"date": d, "signal": "PT1S_TOP5", "ticker": "AAA", "rank": 1, "value": 0.5, "flag": True, "price": 1.0},
                 {"date": d, "signal": "POOL_S", "ticker": "AAA", "rank": None, "value": np.nan, "flag": False, "price": 1.0},
                 {"date": d, "signal": "POOL_S", "ticker": "CCC", "rank": None, "value": np.nan, "flag": False, "price": 1.0}]
    df = pd.DataFrame(days)
    res, _ = st.evaluate(df, o)
    top = next(r for r in res if r["test_id"] == "PT1S_TOP5|h21")
    assert top["n_obs"] == 40
    assert np.isclose(top["mean_ex"], 1.003 ** 21 - 1.001 ** 21)   # AAA − SPY
    assert top["vs_pool"] > 0                                       # AAA − (AAA·CCC 평균)
    empty = next(r for r in res if r["test_id"] == "PT1N_TOP5|h21")
    assert empty["판정(참고)"] == "표본 없음"


def test_big_moves_and_squeeze_evaluation():
    idx = pd.bdate_range("2026-01-01", periods=60)
    n = len(idx)
    opens = pd.DataFrame({"SPY": np.full(n, 100.0), "IWM": np.full(n, 100.0), "SQZ": np.full(n, 10.0),
                          "P1": np.full(n, 10.0), "P2": np.full(n, 10.0)}, index=idx)
    high = opens.copy()
    low = opens.copy()
    high.loc[idx[10], "SQZ"] = 14.0      # +40% 고가 → 기록일 idx[5]·idx[8] 의 21일 안
    low.loc[idx[12], "P2"] = 6.0         # 표본 하나는 −40%
    px = {"Open": opens, "High": high, "Low": low}
    m = st.big_moves(px, idx[5], ["SQZ", "P1", "P2"])
    assert bool(m.loc["SQZ", "up"]) and not bool(m.loc["P1", "up"]) and bool(m.loc["P2", "down"])
    assert st.big_moves(px, idx[-3], ["SQZ"]) is None        # 아직 21일이 안 지남
    rows = []
    for d in idx[:20]:
        rows += [{"date": d, "signal": "SQ_FIRE", "ticker": "SQZ", "rank": None, "value": 0.5, "flag": True, "price": 10.0},
                 {"date": d, "signal": "POOL_SQ", "ticker": "P1", "rank": None, "value": 0.01, "flag": False, "price": 10.0},
                 {"date": d, "signal": "POOL_SQ", "ticker": "P2", "rank": None, "value": 0.01, "flag": False, "price": 10.0}]
    out = st.evaluate_v2(pd.DataFrame(rows), px)
    fire = next(r for r in out if r["test_id"] == "SQ_FIRE|up30")
    assert fire["n_obs"] == 20 and fire["up"] > fire["pool_up"]
    assert next(r for r in out if r["test_id"] == "SQ_FUEL|up30")["판정(참고)"] == "표본 없음"
