"""web_export.sqz_signals: 끝난 봉 판단 · 배지 값 (십자 연속 수 · 풀림 · 연두 전환 · 진행 중 봉)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from paper_trading.market_date import MARKET_TZ
from screener.squeeze_momentum import squeeze_frame
from web_export.sqz_signals import FIRE_BARS, clean_daily, period_complete, ticker_sqz, tf_state


def _et(*args: int) -> datetime:
    return datetime(*args, tzinfo=MARKET_TZ)


def _ohlc(n: int, end: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = c * (1 + rng.normal(0, 0.005, n))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.02, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.02, n))
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=idx)


def test_week_complete_after_friday_close() -> None:
    d = pd.DatetimeIndex(["2026-10-14"])  # 수요일, 그 주 금요일 = 10-16
    assert not period_complete(d, "W", _et(2026, 10, 16, 16, 0))[0]
    assert period_complete(d, "W", _et(2026, 10, 16, 16, 31))[0]
    assert period_complete(d, "W", _et(2026, 10, 17, 2, 30))[0]


def test_month_complete_on_last_weekday() -> None:
    d = pd.DatetimeIndex(["2026-10-30"])  # 10-31 은 토요일 → 마지막 평일 10-30
    assert not period_complete(d, "M", _et(2026, 10, 30, 12, 0))[0]
    assert period_complete(d, "M", _et(2026, 10, 30, 16, 31))[0]
    assert not period_complete(pd.DatetimeIndex(["2026-09-15"]), "M", _et(2026, 9, 29, 17, 0))[0]
    assert period_complete(pd.DatetimeIndex(["2026-09-15"]), "M", _et(2026, 9, 30, 17, 0))[0]


def test_unfinished_day_dropped() -> None:
    df = _ohlc(30, "2026-10-19")
    assert clean_daily(df, _et(2026, 10, 19, 11, 0)).index[-1] == pd.Timestamp("2026-10-16")
    assert clean_daily(df, _et(2026, 10, 19, 17, 0)).index[-1] == pd.Timestamp("2026-10-19")


def test_state_matches_indicator_and_badge_rules() -> None:
    now = _et(2026, 10, 21, 18, 0)  # 수요일 장 마감 뒤 → 이번 주봉은 진행 중
    df = _ohlc(1500, "2026-10-21", seed=7)
    out = ticker_sqz(df, now)
    assert set(out) == {"D", "W", "M"}
    for tf in ("D", "W", "M"):
        st = out[tf]
        f = squeeze_frame(clean_daily(df, now), tf)
        done = f.iloc[:-1] if tf != "D" else f
        if tf == "W":
            assert st.get("live") is not None and st["live"]["date"] == "2026-10-21"
        if tf == "D":
            assert "live" not in st
        last = done.iloc[-1]
        assert st["date"] == done.index[-1].strftime("%Y-%m-%d")
        assert st["dot"] == last["dot"] and st["color"] == last["color"]
        assert abs(st["val"] - last["val"]) < 1e-3
        on = done["sqz_on"].to_numpy()
        run = 0
        while run < len(on) and on[len(on) - 1 - run]:
            run += 1
        assert st["run"] == run
        rel = [a for a in range(FIRE_BARS) if on[-2 - a] and not on[-1 - a]]
        assert (st.get("fire") is None) == (not rel)
        if rel:
            assert st["fire"]["ago"] == rel[0]
        lime = [a for a in range(FIRE_BARS) if done["lime_sqz"].iloc[-1 - a]]
        assert (st.get("lime") is None) == (not lime)


def test_fire_found_somewhere_in_history() -> None:
    """배지 규칙이 실제로 풀림을 잡는지: 기록을 잘라 가며 마지막 끝난 봉이 풀림인 날을 찾는다."""
    df = _ohlc(800, "2026-10-16", seed=3)
    f = squeeze_frame(df, "D")
    rel = np.flatnonzero(f["valid"].to_numpy() & f["sqz_on"].shift(1, fill_value=False).to_numpy()
                         & ~f["sqz_on"].to_numpy())
    i = int(rel[-1])
    cut = df.loc[:f.index[i]]
    st = tf_state(cut, "D", _et(2026, 10, 17, 2, 30))
    assert st is not None and st.get("fire") == {"ago": 0, "run": st["fire"]["run"], "color": f["color"].iloc[i]}
    assert st["fire"]["run"] >= 1 and st["dot"] != "black"


def test_sqz_block_never_breaks_export(monkeypatch) -> None:
    import pandas as pd

    import web_export.dashboard as wd
    import web_export.sqz_signals as Q

    def boom(prices: dict) -> dict:
        raise RuntimeError("bad prices")

    df = _ohlc(1200, "2026-10-16", seed=1)
    out = wd.sqz_block(["BRK.B", "MISSING"], {"BRK-B": df})
    assert set(out) == {"BRK.B"} and set(out["BRK.B"]) == {"D", "W", "M"}
    monkeypatch.setattr(Q, "sqz_states", boom)
    assert wd.sqz_block(["AAPL"], {"AAPL": df}) == {}
    assert wd.ticker_info(pd.Series({"티커": "AAPL"}))["sqzmom"] is None  # build_dashboard 가 채우기 전 기본값
