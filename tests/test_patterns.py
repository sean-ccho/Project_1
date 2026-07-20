"""차트 패턴 감지 회귀 테스트.

합성(이상적) OHLCV로 각 detector가 올바르게 감지/거부하는지 검증한다.
특히 다음 4종은 과거 구조적으로 감지되지 않던 버그를 수정한 회귀 가드다:
  - 상승/하락 삼각형: 수평 추세선 R²≈0 → 게이트 차단 (평탄도 검증으로 수정)
  - 하락쐐기: 수렴 조건 반전 (peak_slope < trough_slope 로 수정)
  - 헤드앤숄더: 진입추세를 좌어깨 직전에서 측정 (트레일링 페널티 제거)
"""
import numpy as np
import pandas as pd

from screener.patterns import (
    detect_triangle, detect_wedge, detect_double_bottom_top,
    detect_head_and_shoulders, detect_cup_with_handle,
    detect_candlestick_patterns, detect_golden_cross,
)


# ---------------------------------------------------------------------------
# 합성 데이터 생성기
# ---------------------------------------------------------------------------

def make_df(close, vol=None):
    close = np.asarray(close, dtype=float)
    n = len(close)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    open_ = np.concatenate([[close[0]], close[:-1]])
    if vol is None:
        vol = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"Open": open_, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Volume": vol},
        index=idx,
    )


def zigzag(points, total):
    return np.interp(np.arange(total), [p[0] for p in points], [p[1] for p in points])


def osc(n, cycles, lo_fn, hi_fn):
    out = np.empty(n)
    for i in range(n):
        t = i / (n - 1)
        phase = np.sin(2 * np.pi * cycles * t - np.pi / 2)
        lo, hi = lo_fn(t), hi_fn(t)
        out[i] = lo + (hi - lo) * (phase + 1) / 2
    return out


def _wedge(peak_vals, trough_vals):
    pts = [(0, (peak_vals[0] + trough_vals[0]) / 2),
           (7, peak_vals[0]), (14, trough_vals[0]), (22, peak_vals[1]), (29, trough_vals[1]),
           (37, peak_vals[2]), (44, trough_vals[2]), (52, peak_vals[3]), (59, trough_vals[2] - 0.3)]
    return zigzag(pts, 60)


DECLINING_VOL = np.linspace(2e6, 0.8e6, 60)


# ---------------------------------------------------------------------------
# 가격 패턴 — 삼각형
# ---------------------------------------------------------------------------

def test_symmetrical_triangle():
    c = osc(60, 4, lambda t: 88 + 8 * t, lambda t: 112 - 8 * t)
    r = detect_triangle(make_df(c, DECLINING_VOL))
    assert r.detected and r.pattern_type == "symmetrical_triangle"


def test_ascending_triangle():
    """고점 수평(R²≈0) + 저점 상승 — 평탄도 검증으로 감지되어야 한다 (회귀 가드)."""
    pts = [(0, 94), (7, 100.0), (15, 88), (22, 99.6), (30, 92),
           (37, 100.0), (44, 96), (52, 99.7), (59, 98)]
    c = zigzag(pts, 60)
    r = detect_triangle(make_df(c, DECLINING_VOL))
    assert r.detected and r.pattern_type == "ascending_triangle"


def test_descending_triangle():
    """고점 하락 + 저점 수평(R²≈0) — 평탄도 검증으로 감지되어야 한다 (회귀 가드)."""
    pts = [(0, 88), (7, 103), (17, 85), (27, 98), (37, 85), (47, 93), (57, 85), (59, 88)]
    c = zigzag(pts, 60)
    r = detect_triangle(make_df(c, DECLINING_VOL))
    assert r.detected and r.pattern_type == "descending_triangle"


# ---------------------------------------------------------------------------
# 가격 패턴 — 쐐기
# ---------------------------------------------------------------------------

def test_rising_wedge():
    # 둘 다 상승, 저점선이 더 가파름(수렴), 기울기차 <20%
    c = _wedge([100, 102, 104, 106], [90, 92.2, 94.4])
    r = detect_wedge(make_df(c, DECLINING_VOL))
    assert r.detected and r.pattern_type == "rising_wedge"


def test_falling_wedge():
    """둘 다 하락, 고점선이 더 가파름(수렴) — 수렴조건 수정 회귀 가드."""
    c = _wedge([110, 108, 106, 104], [100, 98.2, 96.4])
    r = detect_wedge(make_df(c, DECLINING_VOL))
    assert r.detected and r.pattern_type == "falling_wedge"


def test_broadening_is_not_falling_wedge():
    """발산형(broadening)은 falling_wedge로 오감지되면 안 된다 (오탐 가드)."""
    c = _wedge([110, 108.2, 106.4, 104.6], [100, 98, 96])
    r = detect_wedge(make_df(c, DECLINING_VOL))
    assert r.pattern_type != "falling_wedge"


# ---------------------------------------------------------------------------
# 가격 패턴 — 더블 바텀/탑, 컵
# ---------------------------------------------------------------------------

def test_double_bottom():
    pts = [(0, 125), (20, 100), (30, 88), (40, 99), (52, 88.5), (59, 104)]
    r = detect_double_bottom_top(make_df(zigzag(pts, 60)))
    assert r.detected and r.pattern_type == "double_bottom"


def test_double_top():
    pts = [(0, 75), (20, 100), (30, 112), (40, 101), (52, 111.5), (59, 96)]
    r = detect_double_bottom_top(make_df(zigzag(pts, 60)))
    assert r.detected and r.pattern_type == "double_top"


def test_cup_with_handle():
    cup = np.concatenate([
        np.full(5, 100.0),
        100 - 22 * np.sin(np.linspace(0, np.pi, 55)),
        np.linspace(99.5, 100, 5),
        np.linspace(100, 96.5, 12), np.linspace(96.5, 101, 13),
    ])[:90]
    r = detect_cup_with_handle(make_df(cup), lookback=90)
    assert r.detected and r.pattern_type == "cup_with_handle"


# ---------------------------------------------------------------------------
# 반전 패턴 — 헤드앤숄더
# ---------------------------------------------------------------------------

def test_head_and_shoulders():
    """좌어깨 진입이 상승추세 → 천장 패턴 유효 (트레일링 페널티 제거 회귀 가드)."""
    pts = [(0, 70), (15, 95), (22, 106), (29, 97), (37, 120), (45, 97), (52, 106), (59, 90)]
    r = detect_head_and_shoulders(make_df(zigzag(pts, 60)))
    assert r.detected and r.pattern_type == "head_and_shoulders"


def test_inverse_head_and_shoulders():
    pts = [(0, 135), (15, 108), (22, 98), (29, 107), (37, 84), (45, 107), (52, 98), (59, 114)]
    r = detect_head_and_shoulders(make_df(zigzag(pts, 60)))
    assert r.detected and r.pattern_type == "inverse_head_and_shoulders"


def test_hs_after_downtrend_rejected():
    """하락추세 중에 형성된 '천장' 모양은 H&S로 인정되면 안 된다 (오탐 가드)."""
    # 진입 자체가 하락추세(좌어깨 직전이 내림) → 천장 반전으로 보기 어려움
    pts = [(0, 130), (15, 112), (22, 108), (29, 100), (37, 118), (45, 100), (52, 108), (59, 92)]
    r = detect_head_and_shoulders(make_df(zigzag(pts, 60)))
    assert r.pattern_type != "head_and_shoulders"


# ---------------------------------------------------------------------------
# 캔들스틱 패턴
# ---------------------------------------------------------------------------

def _set_bar(df, i, o, h, l, c):
    df.iloc[i, df.columns.get_loc("Open")] = o
    df.iloc[i, df.columns.get_loc("High")] = h
    df.iloc[i, df.columns.get_loc("Low")] = l
    df.iloc[i, df.columns.get_loc("Close")] = c


def test_doji():
    df = make_df(list(np.linspace(100, 100, 6)))
    _set_bar(df, -1, 100.0, 102.0, 98.0, 100.05)
    assert "doji" in detect_candlestick_patterns(df).pattern_type


def test_bullish_engulfing():
    df = make_df(list(np.linspace(120, 100, 25)))  # 하락추세
    _set_bar(df, -2, 102, 102.5, 98.5, 99)
    _set_bar(df, -1, 98, 103.5, 97.5, 103)
    assert "bullish_engulfing" in detect_candlestick_patterns(df).pattern_type


def test_bearish_engulfing():
    df = make_df(list(np.linspace(80, 100, 25)))  # 상승추세
    _set_bar(df, -2, 98, 101.5, 97.5, 101)
    _set_bar(df, -1, 102, 102.5, 96.5, 97)
    assert "bearish_engulfing" in detect_candlestick_patterns(df).pattern_type


def test_morning_star():
    df = make_df(list(np.linspace(120, 100, 25)))  # 하락추세
    _set_bar(df, -3, 105, 105.5, 99.5, 100)
    _set_bar(df, -2, 98.5, 99.2, 98.0, 98.7)
    _set_bar(df, -1, 99.5, 104.5, 99.3, 104)
    assert "morning_star" in detect_candlestick_patterns(df).pattern_type


def test_evening_star():
    df = make_df(list(np.linspace(80, 100, 25)))  # 상승추세
    _set_bar(df, -3, 95, 100.5, 94.5, 100)
    _set_bar(df, -2, 101.5, 102.0, 101.0, 101.3)
    _set_bar(df, -1, 100.5, 100.7, 95.0, 95.5)
    assert "evening_star" in detect_candlestick_patterns(df).pattern_type


# ---------------------------------------------------------------------------
# 이동평균 패턴 — 골든/데드크로스
# ---------------------------------------------------------------------------

def _truncate_after_cross(close, golden=True):
    """50/200 SMA 교차 직후 2봉만 남겨 교차가 최근 5봉 안에 오게 자른다."""
    s = pd.Series(close)
    s50 = s.rolling(50).mean(); s200 = s.rolling(200).mean()
    for i in range(200, len(close)):
        up = s50.iloc[i - 1] < s200.iloc[i - 1] and s50.iloc[i] >= s200.iloc[i]
        dn = s50.iloc[i - 1] > s200.iloc[i - 1] and s50.iloc[i] <= s200.iloc[i]
        if (golden and up) or (not golden and dn):
            return close[:i + 3]
    return close


def test_golden_cross():
    c = np.concatenate([np.linspace(120, 80, 205), np.linspace(80, 210, 150)])
    c = _truncate_after_cross(c, golden=True)
    r = detect_golden_cross(make_df(c))
    assert r.detected and r.pattern_type == "golden_cross"


def test_death_cross():
    c = np.concatenate([np.linspace(80, 200, 205), np.linspace(200, 60, 150)])
    c = _truncate_after_cross(c, golden=False)
    r = detect_golden_cross(make_df(c))
    assert r.detected and r.pattern_type == "death_cross"


def test_golden_cross_imminent():
    # 장기 하락 후 반등하나 아직 교차 전 (단기<장기, 갭 좁혀지는 중)
    c = np.concatenate([np.linspace(120, 85, 180), np.linspace(85, 99, 80)])
    r = detect_golden_cross(make_df(c))
    assert r.detected and r.pattern_type == "golden_cross_imminent"


def _truncate_after_ema_cross(close, short_period, long_period, golden=True):
    s = pd.Series(close)
    es = s.ewm(span=short_period, adjust=False).mean()
    el = s.ewm(span=long_period, adjust=False).mean()
    for i in range(long_period, len(close)):
        up = es.iloc[i - 1] < el.iloc[i - 1] and es.iloc[i] >= el.iloc[i]
        dn = es.iloc[i - 1] > el.iloc[i - 1] and es.iloc[i] <= el.iloc[i]
        if (golden and up) or (not golden and dn):
            return close[:i + 3]
    return close


def test_golden_cross_ema_daily():
    # 일봉 EMA 20/50 골든크로스: 장기 하락 후 강한 반등 → 교차 직후 2봉만 남김
    c = np.concatenate([np.linspace(120, 70, 80), np.linspace(70, 200, 80)])
    c = _truncate_after_ema_cross(c, short_period=20, long_period=50, golden=True)
    r = detect_golden_cross(c if False else make_df(c),
                            short_period=20, long_period=50, ma_type="ema")
    assert r.detected and r.pattern_type == "golden_cross"


def test_golden_cross_imminent_ema_daily():
    # 충분히 긴 하락 후 짧은 약반등 — EMA20 < EMA50 유지하며 갭 좁혀지는 중
    c = np.concatenate([np.linspace(120, 80, 170), np.linspace(80, 82, 30)])
    r = detect_golden_cross(make_df(c),
                            short_period=20, long_period=50, ma_type="ema")
    assert r.detected and r.pattern_type == "golden_cross_imminent"


def test_golden_cross_imminent_lmt_scenario():
    # LMT 류 추세 전환 초입: 강세 후 급락, 약반등 시작.
    # EMA50이 EMA20보다 빨리 떨어져 갭 좁혀지나, EMA20 자체는 10일 평균으로 아직 하락 중.
    # 실제 일봉 호출과 동일한 인자(slope OFF + close>EMA10 필터)로 임박 검출되어야 함.
    c = np.concatenate([
        np.linspace(460, 690, 60),   # 상승
        np.linspace(690, 510, 30),   # 급락
        np.linspace(510, 533, 15),   # 약반등 — close가 EMA10 위로 올라옴
    ])
    df = make_df(c)
    r = detect_golden_cross(
        df, short_period=20, long_period=50, ma_type="ema",
        require_short_slope_confirm=False,
        imminent_close_filter_ema=10,
    )
    assert r.detected and r.pattern_type == "golden_cross_imminent"
    # slope 게이트 켜진 상태(=주봉/월봉 기본)에서는 검출 안 됨을 확인
    r_on = detect_golden_cross(
        df, short_period=20, long_period=50, ma_type="ema",
        require_short_slope_confirm=True,
    )
    assert not r_on.detected


def test_golden_cross_imminent_filtered_by_close_below_short_ema():
    # SCHW/ADBE/AEHL 류: 갭은 좁혀지나 close가 단기 EMA 아래(가격 반등 없음).
    # close > EMA10 필터로 임박 신호 차단되어야 함.
    # 시나리오: 충분히 긴 하락 후 거의 횡보 (close가 EMA10 아래에서 머무름).
    c = np.concatenate([
        np.linspace(120, 80, 170),   # 긴 하락
        np.linspace(80, 79, 30),     # 약간 더 하락 (close가 EMA10 아래)
    ])
    df = make_df(c)
    # 필터 없으면 임박 검출
    r_no_filter = detect_golden_cross(
        df, short_period=20, long_period=50, ma_type="ema",
        require_short_slope_confirm=False,
    )
    # 필터 켜면 차단
    r_with_filter = detect_golden_cross(
        df, short_period=20, long_period=50, ma_type="ema",
        require_short_slope_confirm=False,
        imminent_close_filter_ema=10,
    )
    # close가 EMA10 아래면 필터로 차단
    assert not r_with_filter.detected
