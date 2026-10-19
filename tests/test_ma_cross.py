"""ma_cross: TradingView 범례 값 재현 · 봉 경계 · 미래 미사용 · 앱 임박 규칙과 같은 답.

고정 데이터 tests/fixtures/tv_legend_closes.parquet = HL·T·UNH·ABNB·APP·ABBV 비조정 일봉 종가 (상장 ~ 2026-10-15,
yfinance auto_adjust=False). 범례 값은 2026-10-16 새벽 캡처된 TradingView 차트(SeanEMA)에서 읽은 숫자다.
ABNB·APP 는 상장 몇 년 안 된 종목이라 EMA 시작값(Pine = 처음 n봉 평균)이 맞아야 1센트까지 같다.
ABBV 월봉 EMA200 은 봉이 200개가 안 돼 TV 가 '∅'(값 없음)로 보여 준다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from screener.ma_cross import bar_complete, cross_frame, moving_average, pine_ema, to_bars
from screener.patterns import detect_golden_cross, get_monthly_data, get_weekly_data

FIXTURE = Path(__file__).parent / "fixtures" / "tv_legend_closes.parquet"

# (종목, 봉): (EMA10, EMA20, EMA100, EMA200) — 2026-10-15 봉 기준 TradingView 범례
LEGEND = {
    ("T", "D"): (23.58, 24.10, 24.52, 24.87),
    ("UNH", "D"): (384.54, 381.91, 384.80, 375.00),
    ("HL", "W"): (17.61, 17.69, 14.09, 10.87),
    ("T", "W"): (24.21, 24.38, 24.50, 23.29),
    ("HL", "M"): (16.57, 14.56, 8.02, 6.64),
    ("T", "M"): (24.74, 24.76, 23.03, 23.01),
    ("ABNB", "W"): (None, None, 141.46, 139.34),
    ("APP", "W"): (None, None, 385.41, 289.21),
    ("ABBV", "M"): (246.48, 228.72, 157.98, float("nan")),
}


@pytest.fixture(scope="module")
def closes() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE)


def _close_ok(ours: float, tv: float, rel: float) -> bool:
    return abs(ours - tv) <= 0.011 or abs(ours - tv) / tv <= rel


@pytest.mark.parametrize("key", list(LEGEND))
def test_ema_matches_tradingview_legend(closes: pd.DataFrame, key: tuple[str, str]) -> None:
    ticker, tf = key
    bars = to_bars(closes[ticker], tf)
    for n, tv in zip((10, 20, 100, 200), LEGEND[key]):
        if tv is None:
            continue
        ours = float(moving_average(bars, n, "ema").iloc[-1])
        if np.isnan(tv):
            assert np.isnan(ours), f"{ticker} {tf} EMA{n}: TV 는 ∅ 인데 ours {ours}"
            continue
        # 월봉 EMA200 은 200개월 전 옛 기록 차이가 남아 0.3~0.6% 어긋난다 (알려진 한계)
        rel = 0.01 if (tf == "M" and n == 200) else 0.0005
        assert _close_ok(ours, tv, rel), f"{ticker} {tf} EMA{n}: ours {ours:.4f} vs TV {tv}"


def test_pine_ema_seed() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    e = pine_ema(s, 3)
    assert e.iloc[:2].isna().all()
    assert e.iloc[2] == pytest.approx(2.0)  # 처음 3봉 평균
    a = 2 / 4
    assert e.iloc[3] == pytest.approx(a * 4 + (1 - a) * 2.0)
    assert e.iloc[5] == pytest.approx(a * 6 + (1 - a) * (a * 5 + (1 - a) * (a * 4 + (1 - a) * 2.0)))
    assert pine_ema(s.iloc[:2], 3).isna().all()


def test_bars_match_patterns_resample(closes: pd.DataFrame) -> None:
    s = closes["UNH"].dropna()
    df = pd.DataFrame({"Close": s})
    for tf, ref in (("W", get_weekly_data(df)), ("M", get_monthly_data(df))):
        bars = to_bars(s, tf)
        np.testing.assert_allclose(bars.to_numpy(), ref["Close"].to_numpy())
        # 날짜 = 그 봉의 마지막 거래일 (원래 일봉 날짜 중 하나)
        assert bars.index.isin(s.index).all()
        freq = "W-FRI" if tf == "W" else "M"
        assert (bars.index.to_period(freq) == ref.index.to_period(freq)).all()


def test_pine_crossover_definition() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    # 단기 2 · 장기 3 SMA: 수동으로 교차 지점을 만든다
    close = pd.Series([10.0, 9.0, 8.0, 8.0, 11.0, 4.0], index=idx)
    f = cross_frame(close, 2, 3, "sma", "D")
    ms, ml = close.rolling(2).mean(), close.rolling(3).mean()
    for i in range(3, len(close)):
        up = ms.iloc[i] > ml.iloc[i] and ms.iloc[i - 1] <= ml.iloc[i - 1]
        down = ms.iloc[i] < ml.iloc[i] and ms.iloc[i - 1] >= ml.iloc[i - 1]
        assert bool(f["gc"].iloc[i]) == up
        assert bool(f["dc"].iloc[i]) == down
    assert f["gc"].iloc[4] and f["dc"].iloc[5]
    assert f["bars_since"].iloc[5] == 0 and f["bars_since"].iloc[4] == 0


@pytest.mark.parametrize("tf", ["D", "W", "M"])
def test_no_lookahead(closes: pd.DataFrame, tf: str) -> None:
    s = closes["T"].dropna().iloc[-3000:]
    full = cross_frame(s, 10, 20, "ema", tf)
    cut_at = s.index[-400]
    part = cross_frame(s.loc[:cut_at], 10, 20, "ema", tf)
    cols = ["close", "ma_s", "ma_l", "gap", "golden", "gc", "dc", "bars_since", "gi", "gi_plus"]
    # 주·월봉은 잘린 마지막 봉이 진행 중이라 값이 다르다 → 그 전 봉까지만 같아야 한다
    n = len(part) if tf == "D" else len(part) - 1
    pd.testing.assert_frame_equal(part[cols].iloc[:n], full[cols].loc[part.index[:n]])


def test_bar_complete_flags() -> None:
    dates = pd.DatetimeIndex(["2026-10-09", "2026-10-15"])
    assert bar_complete(dates, "W", pd.Timestamp("2026-10-15")).tolist() == [True, False]
    assert bar_complete(pd.DatetimeIndex(["2026-09-30", "2026-10-15"]), "M",
                        pd.Timestamp("2026-10-15")).tolist() == [True, False]
    assert bar_complete(dates, "D", pd.Timestamp("2026-10-15")).all()


def _app_label(df: pd.DataFrame, tf: str) -> str:
    """앱이 그 봉 데이터로 내는 크로스 라벨 (patterns.py 호출 인자 그대로, 단 일봉도 전체 기록)."""
    if tf == "D":
        return detect_golden_cross(df, short_period=20, long_period=50, ma_type="ema",
                                   require_short_slope_confirm=False, imminent_close_filter_ema=10).pattern_type
    if tf == "W":
        return detect_golden_cross(get_weekly_data(df), short_period=10, long_period=40,
                                   recent_window=5, slope_lookback=5).pattern_type
    return detect_golden_cross(get_monthly_data(df), short_period=3, long_period=10,
                               recent_window=3, slope_lookback=3).pattern_type


@pytest.mark.parametrize("tf,short,long,kind,step", [("D", 20, 50, "ema", 7), ("W", 10, 40, "sma", 5),
                                                     ("M", 3, 10, "sma", 21)])
@pytest.mark.parametrize("ticker", ["HL", "T", "UNH"])
def test_imminent_matches_app_rule(closes: pd.DataFrame, ticker: str, tf: str, short: int, long: int,
                                   kind: str, step: int) -> None:
    s = closes[ticker].dropna().iloc[-2500:]
    full = pd.DataFrame({"Close": s, "Volume": 1_000_000.0})
    checked = hits = 0
    for cut in s.index[-1500::step]:
        last = cross_frame(s.loc[:cut], short, long, kind, tf).iloc[-1]  # 앱처럼 그날까지 데이터만
        app = _app_label(full.loc[:cut], tf)
        assert bool(last["gi"]) == (app == "golden_cross_imminent"), f"{ticker} {tf} {cut.date()}: app={app}"
        checked += 1
        hits += app == "golden_cross_imminent"
    assert checked > 50
