"""주봉·월봉 패턴과 패턴 라벨 연결 테스트.

test_patterns.py 는 감지 함수 하나하나를 일봉 모양 합성 데이터로 본다. 여기서는 그 위 단계를 본다:
일봉→주봉·월봉 변환, 정배열 3단계 (2026-10-27 차트 4선 EMA10/20/100/200 · 있는 선으로만), 끝난 봉만 보는지,
정배열이어도 골든크로스 · 임박 태그는 남는지, 최소 데이터 조건,
그리고 점수 계산이 찾는 라벨이 스크리너가 실제로 만드는 라벨과 같은지 (오타 하나면 가중치가 조용히 0 이 된다).
"""
import numpy as np
import pandas as pd

from screener.chart_ma import CrossLabel, cross_label, ohlcv_bars, split_only_close
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
    "골든크로스", "골든크로스임박", "데드크로스", "데드크로스임박", "이중바닥", "역헤드앤숄더", "컵앤핸들", "상승삼각형", "하락쐐기",
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


def test_chart_bars_have_the_same_values_as_resampled_bars():
    # chart_ma.ohlcv_bars (날짜 = 봉의 마지막 거래일) 값 = get_weekly_data / get_monthly_data (날짜 = 금요일 · 말일)
    df = make_daily(np.linspace(50, 150, 700), start="2023-01-02")
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for tf, resampled in (("W", get_weekly_data(df)), ("M", get_monthly_data(df))):
        bars = ohlcv_bars(df, tf)
        np.testing.assert_allclose(bars[cols].to_numpy(), resampled[cols].to_numpy())
        assert (bars["end"].to_numpy() == resampled.index.to_numpy()).all()


# ---------------------------------------------------------------------------
# 정배열 3단계 (차트 4선)와 골든크로스 태그
# ---------------------------------------------------------------------------

LONG_RISE = make_daily(np.linspace(50, 150, 3000), start="2014-01-01")   # 약 12년: 주봉 4선 다 있음, 월봉 EMA200 없음


def test_long_steady_rise_is_full_uptrend_on_both_timeframes():
    assert names(detect_weekly_patterns(LONG_RISE)) == ["weekly_uptrend_full"]
    assert names(detect_monthly_patterns(LONG_RISE)) == ["monthly_uptrend_full"]   # 월봉은 EMA10 > 20 > 100 (200 은 아직 없음)


def test_young_stock_is_forming_at_most():
    # 상장 1년 반: 주봉 EMA100 · 200 이 아직 없다 → '있는 선으로만' 이면 형성중까지만. 월봉은 EMA20 도 없어 정배열 없음
    df = make_daily(np.linspace(50, 150, 400))
    assert names(detect_weekly_patterns(df)) == ["weekly_uptrend_forming"]
    assert names(detect_monthly_patterns(df)) == []


def test_pullback_stage():
    def rise_then(days, dip):
        return make_daily(np.concatenate([np.linspace(50, 150, 3000), np.linspace(150, 150 * (1 - dip), days)]),
                          start="2014-01-01")
    assert names(detect_weekly_patterns(rise_then(5, 0.02))) == ["weekly_uptrend_pullback"]
    assert names(detect_monthly_patterns(rise_then(40, 0.08))) == ["monthly_uptrend_pullback"]


def test_forming_when_long_lines_are_still_above():
    # 오래 내린 뒤 반등: EMA10 > EMA20 이지만 EMA100 · 200 이 아직 위 → 형성중
    df = make_daily(np.concatenate([np.linspace(200, 60, 2500), np.linspace(60, 90, 150)]), start="2014-01-01")
    assert names(detect_weekly_patterns(df)) == ["weekly_uptrend_forming"]
    assert names(detect_monthly_patterns(df)) == ["golden_cross", "monthly_uptrend_forming"]


# 2026-10-26: 교차는 차트 선(EMA10/EMA20, 끝난 봉 — screener.chart_ma)으로 본다. 아래 데이터는 그 기준으로 고른 것
# 2026-10-27: 정배열(완전 · 눌림목)이어도 골든크로스 · 임박 태그를 지우지 않는다

def test_weekly_golden_cross_kept_while_uptrend_forming():
    df = decline_then_rally(250, 30, 100, 150)
    assert cross_label(split_only_close(df), "W").label == "golden_cross"
    assert names(detect_weekly_patterns(df)) == ["golden_cross", "weekly_uptrend_forming"]


def test_monthly_golden_cross_kept_while_uptrend_forming():
    df = decline_then_rally(500, 120, 100, 220)
    assert cross_label(split_only_close(df), "M").label == "golden_cross"
    assert names(detect_monthly_patterns(df)) == ["golden_cross", "monthly_uptrend_forming"]


def test_golden_tags_kept_with_uptrend_full(monkeypatch):
    # 교차 결과는 고정하고 규칙만 본다 — 정배열(완전)이어도 골든크로스 · 임박 태그가 남는다 (예전엔 지워졌다)
    import screener.patterns as pt
    for label in ("golden_cross", "golden_cross_imminent"):
        monkeypatch.setattr(pt, "cross_label", lambda close, tf, as_of=None, _l=label, **_: CrossLabel(_l, 0.0, ""))
        assert names(detect_weekly_patterns(LONG_RISE)) == [label, "weekly_uptrend_full"]
        assert names(detect_monthly_patterns(LONG_RISE)) == [label, "monthly_uptrend_full"]


# ---------------------------------------------------------------------------
# 끝난 봉만 본다 (진행 중인 주 · 달은 TradingView 에서도 아직 바뀌는 봉)
# ---------------------------------------------------------------------------

def test_patterns_ignore_the_running_week_and_month():
    rng = np.random.default_rng(3)
    for drift in (-0.0008, -0.0004, 0.0, 0.0004, 0.0008, 0.0012):
        close = 60 * np.exp(np.cumsum(rng.normal(drift, 0.02, 2603)))
        df = make_daily(close, start="2015-01-05")            # 마지막 날 = 수요일
        assert df.index[-1].dayofweek < 4
        last_friday = df.index[df.index.dayofweek == 4][-1]
        last_month_end = df.index[df.index < df.index[-1].to_period("M").start_time][-1]
        assert detect_weekly_patterns(df) == detect_weekly_patterns(df.loc[:last_friday])
        assert detect_monthly_patterns(df) == detect_monthly_patterns(df.loc[:last_month_end])


# ---------------------------------------------------------------------------
# 최소 데이터 조건
# ---------------------------------------------------------------------------

def test_not_enough_history():
    assert detect_weekly_patterns(make_daily(np.linspace(50, 100, 59))) == []    # 일봉 60개 미만
    assert detect_weekly_patterns(make_daily(np.linspace(50, 100, 80))) == []    # 끝난 주봉 16개 → 20개 미만
    assert detect_monthly_patterns(make_daily(np.linspace(50, 100, 119))) == []  # 일봉 120개 미만
    assert detect_monthly_patterns(make_daily(np.linspace(50, 100, 200))) == []  # 끝난 월봉 9개 → 12개 미만
    # 주봉 40개: EMA100 이 아직 없어 정배열은 형성중까지만
    got = names(detect_weekly_patterns(make_daily(np.linspace(50, 100, 200))))
    assert [n for n in got if n.startswith("weekly_uptrend")] == ["weekly_uptrend_forming"]


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
