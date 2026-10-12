"""주봉·월봉 패턴과 패턴 라벨 연결 테스트.

test_patterns.py 는 감지 함수 하나하나를 일봉 모양 합성 데이터로 본다. 여기서는 그 위 단계를 본다:
일봉→주봉·월봉 변환, 정배열 3단계, 정배열(완전·눌림목)이면 골든크로스 태그 제거, 최소 데이터 조건,
그리고 점수 계산이 찾는 라벨이 스크리너가 실제로 만드는 라벨과 같은지 (오타 하나면 가중치가 조용히 0 이 된다).
"""
import numpy as np
import pandas as pd

from screener.features import compute_features_for_ticker
from screener.patterns import (
    detect_golden_cross,
    detect_monthly_patterns,
    detect_weekly_patterns,
    get_monthly_data,
    get_weekly_data,
)

# features.py 의 한글 라벨 (일봉 d_pattern_map, 주봉·월봉 pattern_map 의 값)
DAILY_LABELS = {
    "대칭삼각형", "상승삼각형", "하락삼각형", "상승쐐기", "하락쐐기", "이중바닥", "이중천장", "헤드앤숄더",
    "역헤드앤숄더", "컵앤핸들", "골든크로스", "데드크로스", "골든크로스임박", "데드크로스임박",
    "강세잉걸핑", "약세잉걸핑", "모닝스타", "이브닝스타", "도지", "차트반전",
}
HIGHER_TF_LABELS = {
    "골든크로스", "골든크로스임박", "이중바닥", "역헤드앤숄더", "컵앤핸들", "상승삼각형", "하락쐐기",
    "강세잉걸핑", "모닝스타", "정배열(완전)", "정배열(눌림목)", "정배열(형성중)", "차트반전",
}


def make_daily(close, start="2022-01-03"):
    close = np.asarray(close, dtype=float)
    idx = pd.bdate_range(start, periods=len(close))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"Open": open_, "High": np.maximum(open_, close) * 1.01, "Low": np.minimum(open_, close) * 0.99,
         "Close": close, "Volume": np.full(len(close), 1_000_000.0)},
        index=idx,
    )


def decline_then_rally(n1, n2, lo, hi, top=200.0, start="2022-01-03"):
    return make_daily(np.concatenate([np.linspace(top, lo, n1), np.linspace(lo, hi, n2)]), start)


def rise_then_dip(dip_days, dip):
    return make_daily(np.concatenate([np.linspace(50, 150, 400), np.linspace(150, 150 * (1 - dip), dip_days)]))


def names(patterns):
    return [name for name, _ in patterns]


def _numbered(days):
    n = len(days)
    return pd.DataFrame({"Open": np.arange(n) + 100.0, "High": np.arange(n) + 105.0, "Low": np.arange(n) + 95.0,
                         "Close": np.arange(n) + 101.0, "Volume": np.arange(1, n + 1) * 1000.0}, index=days)


# ---------------------------------------------------------------------------
# 일봉 → 주봉·월봉 변환
# ---------------------------------------------------------------------------

def test_weekly_bars_with_holiday_and_unfinished_week():
    # 2026-03-30(월) ~ 04-14(화). 04-03(금) 휴장, 마지막 주는 월·화만 있음
    days = pd.bdate_range("2026-03-30", "2026-04-14").drop(pd.Timestamp("2026-04-03"))
    w = get_weekly_data(_numbered(days))
    assert list(w.index) == [pd.Timestamp("2026-04-03"), pd.Timestamp("2026-04-10"), pd.Timestamp("2026-04-17")]
    first = w.iloc[0]  # 월~목 4일
    assert (first["Open"], first["High"], first["Low"], first["Close"]) == (100.0, 108.0, 95.0, 104.0)
    assert first["Volume"] == (1 + 2 + 3 + 4) * 1000.0
    last = w.iloc[-1]  # 안 끝난 주도 그날까지로 마지막 봉이 된다 (TradingView 차트와 같음)
    assert (last["Open"], last["High"], last["Low"], last["Close"]) == (109.0, 115.0, 104.0, 111.0)
    assert last["Volume"] == (10 + 11) * 1000.0


def test_monthly_bars_label_month_end_and_keep_unfinished_month():
    days = pd.bdate_range("2026-01-02", "2026-03-10")
    df = _numbered(days)
    m = get_monthly_data(df)
    assert list(m.index) == [pd.Timestamp("2026-01-31"), pd.Timestamp("2026-02-28"), pd.Timestamp("2026-03-31")]
    jan = df[df.index.month == 1]
    assert m.iloc[0]["Open"] == jan["Open"].iloc[0] and m.iloc[0]["Close"] == jan["Close"].iloc[-1]
    assert m.iloc[0]["High"] == jan["High"].max() and m.iloc[0]["Low"] == jan["Low"].min()
    assert m.iloc[-1]["Close"] == df["Close"].iloc[-1]  # 3월은 10일까지 — 마지막 봉 종가는 그날 종가
    assert m["Volume"].sum() == df["Volume"].sum()


# ---------------------------------------------------------------------------
# 정배열 3단계와 골든크로스 태그
# ---------------------------------------------------------------------------

def test_steady_rise_is_full_uptrend_on_both_timeframes():
    df = make_daily(np.linspace(50, 150, 400))
    assert names(detect_weekly_patterns(df)) == ["weekly_uptrend_full"]
    assert names(detect_monthly_patterns(df)) == ["monthly_uptrend_full"]


def test_pullback_stage():
    assert names(detect_weekly_patterns(rise_then_dip(10, 0.05))) == ["weekly_uptrend_pullback"]
    assert names(detect_monthly_patterns(rise_then_dip(10, 0.08))) == ["monthly_uptrend_pullback"]


def test_weekly_golden_cross_dropped_when_uptrend_full():
    # 주봉 SMA10/40 이 최근 5주 안에 교차했어도, 이미 정배열(완전)이면 골든크로스 태그는 뺀다
    df = decline_then_rally(250, 60, 100, 220)
    weekly = get_weekly_data(df)
    gc = detect_golden_cross(weekly, short_period=10, long_period=40, recent_window=5, slope_lookback=5)
    assert gc.pattern_type == "golden_cross"
    assert names(detect_weekly_patterns(df)) == ["weekly_uptrend_full"]


def test_weekly_golden_cross_kept_while_uptrend_forming():
    df = decline_then_rally(250, 60, 100, 180)
    assert names(detect_weekly_patterns(df)) == ["golden_cross", "weekly_uptrend_forming"]


def test_monthly_golden_cross_dropped_when_uptrend_full():
    df = decline_then_rally(500, 80, 100, 180)
    monthly = get_monthly_data(df)
    gc = detect_golden_cross(monthly, short_period=3, long_period=10, recent_window=3, slope_lookback=3)
    assert gc.pattern_type == "golden_cross"
    assert names(detect_monthly_patterns(df)) == ["monthly_uptrend_full"]


def test_monthly_golden_cross_kept_while_uptrend_forming():
    df = decline_then_rally(400, 40, 80, 125, start="2020-01-02")
    assert names(detect_monthly_patterns(df)) == ["golden_cross", "monthly_uptrend_forming"]


# ---------------------------------------------------------------------------
# 최소 데이터 조건
# ---------------------------------------------------------------------------

def test_not_enough_history():
    assert detect_weekly_patterns(make_daily(np.linspace(50, 100, 59))) == []    # 일봉 60개 미만
    assert detect_weekly_patterns(make_daily(np.linspace(50, 100, 80))) == []    # 주봉 16개 → 20개 미만
    assert detect_monthly_patterns(make_daily(np.linspace(50, 100, 119))) == []  # 일봉 120개 미만
    assert detect_monthly_patterns(make_daily(np.linspace(50, 100, 200))) == []  # 월봉 10개 → 12개 미만
    # 주봉 정배열은 EMA50 이라 주봉 50개가 있어야 한다 (일봉 200개 ≈ 주봉 40개)
    assert not any(n.startswith("weekly_uptrend") for n in names(detect_weekly_patterns(make_daily(np.linspace(50, 100, 200)))))


# ---------------------------------------------------------------------------
# 라벨 연결: 점수 계산이 찾는 라벨 = 스크리너가 만드는 라벨
# ---------------------------------------------------------------------------

def test_scoring_keys_are_labels_the_screener_makes():
    import screener.config as cfg
    from paper_trading import candidate_selector
    from screener import signals

    assert set(signals._BULLISH_DAILY_REVERSAL) <= DAILY_LABELS
    assert set(signals._BULLISH_DAILY_MOMENTUM) <= DAILY_LABELS
    for weights in (cfg.WEEKLY_PATTERN_WEIGHTS_BOTTOM, cfg.WEEKLY_PATTERN_WEIGHTS_MOMENTUM,
                    cfg.MONTHLY_PATTERN_WEIGHTS_BOTTOM, cfg.MONTHLY_PATTERN_WEIGHTS_MOMENTUM):
        assert set(weights) <= HIGHER_TF_LABELS
    assert set(candidate_selector._BULLISH_PATTERNS) <= DAILY_LABELS | HIGHER_TF_LABELS


def test_feature_pattern_strings_use_known_korean_labels():
    # 새 패턴을 추가하고 한글 이름을 안 달면 영어 이름이 그대로 나가 점수에서 빠진다
    rng = np.random.default_rng(7)
    shapes = [np.linspace(50, 150, 1000), np.concatenate([np.linspace(200, 100, 700), np.linspace(100, 180, 300)])]
    shapes += [60 * np.exp(np.cumsum(rng.normal(drift, 0.02, 1000))) for drift in (-0.001, 0.0, 0.0005, 0.001) * 3]
    daily, higher = set(), set()
    for close in shapes:
        fs = compute_features_for_ticker(make_daily(close, start="2021-01-04"))
        daily |= {x.strip() for x in fs.daily_patterns.split(",") if x.strip()}
        for s in (fs.weekly_patterns, fs.monthly_patterns):
            higher |= {x.strip() for x in s.split(",") if x.strip()}
    assert daily and higher
    assert daily <= DAILY_LABELS, daily - DAILY_LABELS
    assert higher <= HIGHER_TF_LABELS, higher - HIGHER_TF_LABELS
