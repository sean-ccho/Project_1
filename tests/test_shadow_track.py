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
