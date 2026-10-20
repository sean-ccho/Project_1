"""squeeze_momentum: Pine SQZMOM_LB 를 봉마다 그대로 따라 계산한 느린 참조 구현과 같은지."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from screener.squeeze_momentum import pine_linreg, squeeze_frame, system_position, to_ohlc_bars


def _ohlc(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = c * (1 + rng.normal(0, 0.005, n))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.02, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.02, n))
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=idx)


def _reference(df: pd.DataFrame, n: int = 20, kc: float = 1.5) -> pd.DataFrame:
    """Pine 원본을 봉마다 한 줄씩 (반복문) 계산."""
    h, l, c = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
    m = len(c)
    nan = float("nan")
    tr = [nan] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, m)]
    rows = []
    src = [nan] * m
    for i in range(m):
        if i >= n - 1:
            w = c[i - n + 1:i + 1]
            basis = sum(w) / n
            sd = math.sqrt(sum((x - basis) ** 2 for x in w) / n)
            trw = tr[i - n + 1:i + 1]
            rangema = sum(trw) / n if all(x == x for x in trw) else nan
            hh, ll = max(h[i - n + 1:i + 1]), min(l[i - n + 1:i + 1])
            src[i] = c[i] - ((hh + ll) / 2 + basis) / 2
            lbb, ubb = basis - kc * sd, basis + kc * sd
            lkc, ukc = basis - kc * rangema, basis + kc * rangema
            on = lbb > lkc and ubb < ukc
            off = lbb < lkc and ubb > ukc
        else:
            on = off = False
        val = nan
        if i >= 2 * n - 2:
            ys = src[i - n + 1:i + 1]
            xs = list(range(n))
            xm, ym = sum(xs) / n, sum(ys) / n
            slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)
            val = ym + slope * (n - 1 - xm)
        rows.append({"val": val, "sqz_on": on, "sqz_off": off})
    out = pd.DataFrame(rows, index=df.index)
    prev = out["val"].shift(1).fillna(0.0)
    out["color"] = np.where(out["val"].isna(), "",
                            np.where(out["val"] > 0, np.where(out["val"] > prev, "lime", "green"),
                                     np.where(out["val"] < prev, "red", "maroon")))
    return out


def test_linreg_matches_polyfit_and_lsma_identity() -> None:
    x = np.random.default_rng(1).normal(size=80).cumsum()
    got = pine_linreg(x, 20)
    assert np.isnan(got[:19]).all()
    for i in range(19, 80):
        b, a = np.polyfit(np.arange(20), x[i - 19:i + 1], 1)
        assert got[i] == pytest.approx(a + b * 19, abs=1e-10)
    s = pd.Series(x)
    wma = s.rolling(20).apply(lambda w: np.dot(w, np.arange(1, 21)) / 210, raw=True)
    lsma = 3 * wma - 2 * s.rolling(20).mean()
    np.testing.assert_allclose(got[19:], lsma.to_numpy()[19:], atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_daily_matches_reference(seed: int) -> None:
    df = _ohlc(300, seed)
    f = squeeze_frame(df, "D")
    ref = _reference(df)
    np.testing.assert_allclose(f["val"].to_numpy(), ref["val"].to_numpy(), rtol=1e-10, atol=1e-10, equal_nan=True)
    assert (f["sqz_on"] == ref["sqz_on"]).all()
    assert (f["sqz_off"] == ref["sqz_off"]).all()
    assert (f["color"] == ref["color"]).all()
    assert f["sqz_on"].any() and f["sqz_off"].any()


def test_events_follow_definitions() -> None:
    f = squeeze_frame(_ohlc(1500, 3), "D")
    v = f[f["valid"]]
    prev = f["val"].shift(1).loc[v.index]
    prev_on = f["sqz_on"].shift(1).loc[v.index].astype(bool)
    prev_color = f["color"].shift(1).loc[v.index]
    assert (v["fire"] == (prev_on & ~v["sqz_on"] & (v["val"] > 0) & (v["val"] > prev))).all()
    assert (v["mom0"] == ((v["val"] > 0) & (prev <= 0))).all()
    assert (v["turn"] == ((v["color"] == "maroon") & (prev_color == "red"))).all()
    assert (v["lime_sqz"] == (prev_on & (v["color"] == "lime") & (prev_color != "lime"))).all()
    assert v["fire"].sum() > 5 and v["mom0"].sum() > 5 and v["turn"].sum() > 5 and v["lime_sqz"].sum() > 5
    assert not f.loc[~f["valid"], ["fire", "mom0", "turn", "lime_sqz", "rising", "falling"]].any().any()


def test_weekly_monthly_bars() -> None:
    df = _ohlc(400, 4)
    w = to_ohlc_bars(df, "W")
    wk = df.index.to_period("W-FRI")
    first = df.groupby(wk).head(1)
    assert w.index[0] == df.index[wk == wk[0]].max()
    assert w["Open"].iloc[0] == first["Open"].iloc[0]
    assert w["High"].iloc[1] == df["High"][wk == wk[wk != wk[0]][0]].max()
    m = to_ohlc_bars(df, "M")
    assert len(m) == df.index.to_period("M").nunique()
    fw = squeeze_frame(df, "W", as_of=df.index[-1])
    assert fw.index.equals(w.index)
    np.testing.assert_allclose(fw["val"], _reference(w)["val"], rtol=1e-10, atol=1e-10, equal_nan=True)


def test_system_position_enters_on_event_and_exits_on_first_fall() -> None:
    f = pd.DataFrame({"fire": [False, True, False, False, True, False, True, False],
                      "falling": [False, False, False, True, False, True, False, True]})
    pos = system_position(f, "fire")
    assert pos.tolist() == [False, True, True, False, True, False, True, False]


def test_engine_books_map_weekly_bars_to_days() -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from sqz_vs_pt1 import ticker_books

    df = _ohlc(700, 5)
    books = ticker_books("X", df)
    w = books["SQZ-W"].set_index("date")
    f = squeeze_frame(df, "W", df.index[-1])
    f = f[f["complete"]]
    lime = (f["color"] == "lime") & f["valid"]
    for day in df.index[300:340]:
        last_bar = f.index[f.index <= day].max()
        assert w.at[day, "hold_next"] == bool(lime.loc[last_bar])
        is_bar_day = day in f.index
        turn_down = is_bar_day and bool(f.at[day, "falling"]) and bool(f["rising"].shift(1, fill_value=False).at[day])
        assert w.at[day, "exit_full"] == turn_down
    s = books["SQZ-W-FIRE"].set_index("date")
    assert not s.loc[~s.index.isin(f.index), "buy_ev"].any()
    pos = system_position(f, "fire")
    assert s.loc[f.index, "hold_next"].tolist() == pos.tolist()
