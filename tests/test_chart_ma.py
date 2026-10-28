"""chart_ma: 차트(TradingView SeanEMA)와 같은 선으로 교차 라벨 — 배당 되돌리기 · 교차/임박/데드 · 끝난 봉 · 예고."""
import numpy as np
import pandas as pd
import pytest

from screener.chart_ma import cross_label, split_only_close
from screener.features import compute_features_for_ticker, feature_row_from_set
from screener.ma_cross import pine_ema, to_bars


def _yahoo_adjust(close, div):
    """Yahoo 방식 배당 조정: A_t = C_t × Π(배당락일 j > t) (1 − D_j / C_(j−1))."""
    c, d = np.asarray(close, float), np.asarray(div, float)
    f, cur = np.ones(len(c)), 1.0
    for t in range(len(c) - 1, 0, -1):
        f[t] = cur
        if d[t] > 0:
            cur *= 1 - d[t] / c[t - 1]
    f[0] = cur
    return c * f


def _daily(c, start="2023-01-02"):
    return pd.Series(np.asarray(c, float), index=pd.bdate_range(start, periods=len(c)))


def _first_cross(s, up=True):
    e10, e20 = pine_ema(s, 10), pine_ema(s, 20)
    hit = (e10 > e20) & (e10.shift(1) <= e20.shift(1)) if up else (e10 < e20) & (e10.shift(1) >= e20.shift(1))
    return int(np.flatnonzero(hit.to_numpy())[0]), e10, e20


def test_split_only_close_undoes_dividend_adjustment():
    rng = np.random.default_rng(1)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    div = np.zeros(300)
    div[[60, 120, 180, 240]] = [0.5, 0.5, 0.55, 0.55]
    df = pd.DataFrame({"Close": _yahoo_adjust(c, div), "Dividends": div}, index=pd.bdate_range("2022-01-03", periods=300))
    np.testing.assert_allclose(split_only_close(df).to_numpy(), c, rtol=1e-12)
    # 백테스트: 나중에 받은 가격을 잘라 쓰면 뒤 배당이 섞여 어긋난다 → 전체 기간으로 미리 되돌린 CloseSplit 칸이 이긴다
    sliced = df.iloc[:200].assign(Close=df["Close"].iloc[:200] * 0.97)
    assert np.abs(split_only_close(sliced).to_numpy() / c[:200] / 0.97 - 1).max() > 1e-4
    with_col = sliced.assign(CloseSplit=split_only_close(df).iloc[:200])
    np.testing.assert_allclose(split_only_close(with_col).to_numpy(), c[:200], rtol=1e-12)


def test_backtest_price_map_carries_full_history_split_close():
    from screener.backtest import _prepare_price_map
    rng = np.random.default_rng(2)
    c = 30 * np.exp(np.cumsum(rng.normal(0, 0.01, 250)))
    div = np.zeros(250)
    div[[50, 150]] = 0.3
    idx = pd.bdate_range("2022-01-03", periods=250)
    frame = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": _yahoo_adjust(c, div), "Volume": 1.0, "Dividends": div},
                         index=idx)
    raw = pd.concat({"AAA": frame}, axis=1)
    np.testing.assert_allclose(_prepare_price_map(raw)["AAA"]["CloseSplit"].to_numpy(), c, rtol=1e-12)


def test_split_only_close_without_dividends_is_close():
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=pd.bdate_range("2022-01-03", periods=3))
    assert split_only_close(df).tolist() == [1.0, 2.0, 3.0]
    assert split_only_close(df.assign(Dividends=0.0)).tolist() == [1.0, 2.0, 3.0]


def test_daily_imminent_then_cross_then_none():
    s = _daily(np.concatenate([np.linspace(120, 100, 80), np.linspace(100, 130, 40)]))
    k, e10, e20 = _first_cross(s)
    assert cross_label(s.iloc[:k], "D").label == "golden_cross_imminent"         # 교차 전날
    two = cross_label(s.iloc[:k + 3], "D")                                         # 교차 2봉 뒤
    assert two.label == "golden_cross"
    assert two.gap == pytest.approx(float(e10.iloc[k + 2] / e20.iloc[k + 2] - 1))
    assert cross_label(s.iloc[:k + 9], "D").label == "none"                       # 5봉이 지나면 라벨 없음


def test_daily_death_cross_mirror():
    s = _daily(np.concatenate([np.linspace(100, 120, 80), np.linspace(120, 90, 40)]))
    k, _, _ = _first_cross(s, up=False)
    assert cross_label(s.iloc[:k], "D").label == "death_cross_imminent"
    assert cross_label(s.iloc[:k + 2], "D").label == "death_cross"


def test_weekly_uses_finished_bars_and_previews_the_running_week():
    # 하락 → 바닥 20주 횡보 → 이번 주(월~수, 진행 중) 30% 급등: 끝난 주봉은 교차 전, 진행 중인 주봉에서만 교차
    c = np.concatenate([np.linspace(200, 100, 300), np.full(100, 100.0), np.full(3, 130.0)])
    s = pd.Series(c, index=pd.bdate_range("2019-01-07", periods=len(c)))
    assert s.index[-1].day_name() == "Wednesday"
    w = to_bars(s, "W")
    e10, e20 = pine_ema(w, 10), pine_ema(w, 20)
    assert e10.iloc[-2] < e20.iloc[-2] and e10.iloc[-1] > e20.iloc[-1]
    cl = cross_label(s, "W")
    assert cl.label == "none" and cl.preview == "golden_cross"
    assert cl.gap == pytest.approx(float(e10.iloc[-2] / e20.iloc[-2] - 1))          # 갭도 끝난 봉 값
    # 금요일까지 가서 그 주가 끝나면 예고가 아니라 라벨
    s2 = pd.concat([s, pd.Series([130.0, 130.0], index=pd.bdate_range(s.index[-1] + pd.Timedelta(days=1), periods=2))])
    done = cross_label(s2, "W")
    assert done.label == "golden_cross" and done.preview == ""


def test_not_enough_bars_is_none():
    s = _daily([10.0] * 15)
    assert cross_label(s, "D").label == "none"
    assert cross_label(s, "M").label == "none"


def test_feature_row_has_chart_line_gaps_and_previews():
    c = np.concatenate([np.linspace(200, 100, 300), np.full(100, 100.0), np.full(3, 130.0)])
    idx = pd.bdate_range("2019-01-07", periods=len(c))
    open_ = np.concatenate([[c[0]], c[:-1]])
    df = pd.DataFrame({"Open": open_, "High": np.maximum(open_, c) * 1.01, "Low": np.minimum(open_, c) * 0.99,
                       "Close": c, "Volume": np.full(len(c), 1_000_000.0)}, index=idx)
    row = feature_row_from_set("TEST", compute_features_for_ticker(df))
    w = cross_label(pd.Series(c, index=idx), "W")
    assert row["주봉_MA갭"] == pytest.approx(w.gap)
    assert row["주봉_예고"] == "골든크로스" and "골든크로스" not in str(row["주봉패턴"])
    assert row["일봉_MA갭"] == pytest.approx(cross_label(pd.Series(c, index=idx), "D").gap)
    assert row["월봉_예고"] in ("", "골든크로스", "골든크로스임박", "데드크로스", "데드크로스임박")


# ---------------------------------------------------------------------------
# 2026-10-27: 모든 패턴을 같은 바탕으로 — 차트 가격 묶음 · 끝난 봉 · 4선 정배열 · Pine 계산
# ---------------------------------------------------------------------------

def _ohlcv(c, start="2010-01-04"):
    c = np.asarray(c, float)
    idx = pd.bdate_range(start, periods=len(c))
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.01, "Low": np.minimum(o, c) * 0.99, "Close": c,
                         "Volume": np.full(len(c), 1e6)}, index=idx)


def test_split_only_ohlcv_divides_open_high_low_by_the_same_factor():
    from screener.chart_ma import split_only_ohlcv
    raw = _ohlcv(50 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.01, 300))))
    div = np.zeros(300)
    div[[50, 150, 250]] = [0.4, 0.45, 0.5]
    adj_close = _yahoo_adjust(raw["Close"], div)
    f = adj_close / raw["Close"].to_numpy()
    adj = raw.copy()
    for col in ("Open", "High", "Low", "Close"):
        adj[col] = raw[col].to_numpy() * f
    adj["Dividends"] = div
    out = split_only_ohlcv(adj)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        np.testing.assert_allclose(out[col].to_numpy(), raw[col].to_numpy(), rtol=1e-12)


def test_chart_view_never_sees_after_as_of():
    # 미리 만든 전체 봉(백테스트 얼린 파일)에서 본 as_of 시점 = as_of 까지 잘라 만든 봉 (진행 중인 봉 값 · complete 까지)
    from screener.chart_ma import chart_prices, chart_view
    df = _ohlcv(60 * np.exp(np.cumsum(np.random.default_rng(5).normal(0.0003, 0.02, 3000))))
    full = chart_prices(df)
    for as_of in ("2015-06-10", "2017-03-31", "2018-12-28", "2020-02-28", "2021-07-14"):
        a = pd.Timestamp(as_of)
        cut = chart_prices(df.loc[:a])
        for x, y in zip(chart_view(full, a), chart_view(cut, a)):
            pd.testing.assert_frame_equal(x, y, check_freq=False)


def test_weekend_month_end_is_finished_after_the_last_friday():
    from screener.ma_cross import bar_complete
    nov = pd.DatetimeIndex([pd.Timestamp("2024-11-29")])            # 11월 30일 = 토요일
    assert bar_complete(nov, "M", pd.Timestamp("2024-11-29")).all()
    assert not bar_complete(nov, "M", pd.Timestamp("2024-11-28")).any()
    oct_ = pd.DatetimeIndex([pd.Timestamp("2024-10-30")])           # 10월 31일 = 목요일 → 31일 장이 끝나야
    assert not bar_complete(oct_, "M", pd.Timestamp("2024-10-30")).any()
    assert bar_complete(oct_, "M", pd.Timestamp("2024-10-31")).all()


def test_uptrend_stage_uses_only_lines_that_exist():
    from screener.chart_ma import uptrend_stage
    rise = lambda n: pd.Series(np.linspace(50, 150, n), index=pd.bdate_range("2000-01-03", periods=n))  # noqa: E731
    assert uptrend_stage(rise(19)) == ""             # EMA20 도 없음
    assert uptrend_stage(rise(60)) == "forming"      # EMA100 · 200 없음 → 형성중까지만
    assert uptrend_stage(rise(150)) == "full"        # EMA10 > 20 > 100 (200 없음)
    assert uptrend_stage(rise(400)) == "full"        # 네 선 다
    down = pd.Series(np.linspace(150, 50, 400), index=pd.bdate_range("2000-01-03", periods=400))
    assert uptrend_stage(down) == ""
    # 네 선 정렬인데 종가가 EMA10 아래 · EMA20 위 → 눌림목
    s = pd.concat([rise(400), pd.Series([148.6], index=[pd.Timestamp("2001-07-30")])])
    from screener.ma_cross import pine_ema
    e10, e20 = pine_ema(s, 10).iloc[-1], pine_ema(s, 20).iloc[-1]
    assert e20 < s.iloc[-1] <= e10
    assert uptrend_stage(s) == "pullback"


def test_pine_rsi_matches_the_pine_definition():
    from screener.chart_ma import pine_rsi
    c = pd.Series(50 * np.exp(np.cumsum(np.random.default_rng(9).normal(0, 0.02, 120))))
    ch = np.diff(c.to_numpy())
    up, dn = np.clip(ch, 0, None), np.clip(-ch, 0, None)
    au, ad = up[:14].mean(), dn[:14].mean()
    ref = [np.nan] * 14 + [100 - 100 / (1 + au / ad)]
    for u, d in zip(up[14:], dn[14:]):
        au, ad = (au * 13 + u) / 14, (ad * 13 + d) / 14
        ref.append(100 - 100 / (1 + au / ad))
    np.testing.assert_allclose(pine_rsi(c, 14).to_numpy(), np.array(ref), rtol=1e-10)


def test_features_use_long_chart_history_when_given():
    # 받은 일봉은 5년뿐이어도 차트 가격(상장 이후 전체)을 주면 월봉 EMA100 이 생겨 정배열(완전)이 된다
    from screener.chart_ma import chart_prices
    df = _ohlcv(np.linspace(20, 150, 4000), start="2010-01-04")
    recent = df.loc[df.index[-1] - pd.DateOffset(years=5):]
    short = compute_features_for_ticker(recent)
    long_ = compute_features_for_ticker(recent, chart_prices(df))
    assert "정배열(완전)" not in short.monthly_patterns          # 월봉 60개 — EMA100 없음 → 형성중까지만
    assert "정배열(완전)" in long_.monthly_patterns


def test_extend_chart_fills_days_the_long_history_is_missing():
    # 전체 기록(period="max")이 하루 늦게 들어와도 받은 일봉으로 채우면 전체를 다 받은 것과 같다 — 분할 기준이 달라도
    from screener.chart_ma import chart_prices, chart_view, extend_chart
    df = _ohlcv(60 * np.exp(np.cumsum(np.random.default_rng(11).normal(0.0003, 0.02, 2600))))
    as_of = df.index[-1]
    want = chart_view(chart_prices(df), as_of)
    for scale in (1.0, 2.0):                                   # 2.0 = 나중에 2:1 분할이 있어 차트 파일 값이 두 배
        lagging = df.loc[: df.index[-3]].copy()
        lagging[["Open", "High", "Low", "Close"]] *= scale
        lagging["Volume"] /= scale
        got = chart_view(extend_chart(chart_prices(lagging), df), as_of)
        for x, y in zip(got, want):
            scaled = y.copy()
            scaled[["Open", "High", "Low", "Close"]] *= scale
            scaled["Volume"] /= scale
            pd.testing.assert_frame_equal(x, scaled, check_freq=False, rtol=1e-9)


def test_cross_label_tail_matches_full_history(monkeypatch):
    # 교차 판정을 마지막 CROSS_TAIL 봉에서만 해도 (선은 전체 기록) 전체로 계산한 라벨 · 간격 · 예고와 같다 (2026-10-27)
    import screener.chart_ma as cm
    from screener.chart_ma import chart_prices, chart_view
    for seed in range(12):
        df = _ohlcv(50 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0.0002, 0.022, 3200))), start="2008-01-02")
        cp = chart_prices(df)
        for as_of in df.index[[-1, -7, -25, -140, -600]]:
            d, w, m = chart_view(cp, as_of)
            for close, tf, bars in ((d["Close"], "D", False), (w["Close"], "W", True), (m["Close"], "M", True)):
                monkeypatch.setattr(cm, "CROSS_TAIL", None)
                full = cm.cross_label(close, tf, as_of, bars=bars)
                monkeypatch.setattr(cm, "CROSS_TAIL", 60)
                tail = cm.cross_label(close, tf, as_of, bars=bars)
                assert (tail.label, tail.preview) == (full.label, full.preview)
                assert tail.gap == full.gap or (np.isnan(tail.gap) and np.isnan(full.gap))
