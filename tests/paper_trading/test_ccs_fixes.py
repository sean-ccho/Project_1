"""2026-10-08 CCS 점검 수정 테스트: MACD 단위, 보유 종목 오늘 점수(R2 스위치), 백테스트 실적일 달력."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from paper_trading.candidate_selector import _apply_hard_filters, _score_entry_timing, held_ccs_v1_today
from paper_trading.config_override import config_overrides
from paper_trading.earnings_calendar import EarningsCalendar
from paper_trading.engine import current_ccs


def _bottom(macd_hist: float, close: float) -> pd.Series:
    # RSI 50·볼린저 0.5·거래량 Z 0 → MACD 말고는 타이밍 점수 0
    return pd.Series({"전략구분": "📉 바닥반등", "RSI": 55.0, "bollinger_pband": 0.5, "거래량Z(20)": -1.0,
                      "macd_hist": macd_hist, "현재가격": close})


def test_bottom_macd_threshold_is_relative_to_price():
    # 330달러 종목 −0.92 (−0.28%) 는 '깊은 마이너스 아님' → 가점. 옛 기준(−0.5달러)은 감점이었다
    assert _score_entry_timing(_bottom(-0.92, 330.0)) == 0.1
    # 5.89달러 종목 −0.13 (−2.3%) 은 깊은 마이너스 → 가점 없음. 옛 기준은 가점이었다
    assert _score_entry_timing(_bottom(-0.13, 5.89)) == 0.0
    # 가격이 없으면 가점 없음
    assert _score_entry_timing(_bottom(-0.1, float("nan"))) == 0.0


def _snapshot() -> pd.DataFrame:
    base = {"RSI": 50.0, "bollinger_pband": 0.5, "거래량Z(20)": 0.0, "macd_hist": 0.0, "현재가격": 100.0,
            "52주포지션": 0.5, "모멘텀_적합도": 7.0, "바닥반등_적합도": 0.0}
    return pd.DataFrame([
        {**base, "티커": "HELD", "전략구분": "📈 모멘텀", "섹터": "Tech"},
        {**base, "티커": "OTHER", "전략구분": "📈 모멘텀", "섹터": "Tech"},
    ])


def test_held_today_score_excludes_own_sector_penalty_and_skips_missing():
    holdings = [{"ticker": "HELD", "sector": "Tech", "ccs_score": 0.9}, {"ticker": "GONE", "sector": "Energy"}]
    today = held_ccs_v1_today(_snapshot(), holdings, "neutral")
    assert set(today) == {"HELD"}  # 스냅샷에 없는 GONE 은 빠진다
    # 다른 보유가 Tech 가 아니므로 섹터 페널티 0 → 같은 행을 보유 없이 매긴 값과 같아야 한다
    alone = held_ccs_v1_today(_snapshot(), [{"ticker": "HELD", "sector": "Tech"}], "neutral")
    assert today["HELD"] == alone["HELD"]


def test_current_ccs_default_is_buy_time_switch_uses_today_with_fallback():
    pos = {"ticker": "HELD", "ccs_score": 0.9}
    debug = {"held_ccs_v1_today": {"HELD": 0.4}}
    assert current_ccs(pos, debug) == 0.9  # 기본 = 매수 당시 점수 (기존 동작)
    with config_overrides({"PT1_REPLACE_TODAY_CCS": True}):
        assert current_ccs(pos, debug) == 0.4
        assert current_ccs({"ticker": "GONE", "ccs_score": 0.7}, debug) == 0.7  # 오늘 스냅샷에 없으면 매수 당시
        # v2 점수가 있으면 v2 우선 (기존 v2 동작 유지)
        assert current_ccs(pos, {**debug, "ccs_v2_by_ticker": {"HELD": 0.6}}) == 0.6


def _passing(days) -> dict:
    return {"티커": "AAA", "전략구분": "📈 모멘텀", "모멘텀_적합도": 7.0, "52주포지션": 0.3, "판단": "1. 매수 후보",
            "RSI": 55.0, "bollinger_pband": 0.5, "5일수익률": 0.02, "최근20일평균거래대금": 5e7,
            "days_to_next_earnings": days, "adx": 30.0, "섹터": "Tech"}


def test_earnings_filter_blocks_within_three_days_only():
    df = pd.DataFrame([_passing(2.0), {**_passing(10.0), "티커": "BBB"}, {**_passing(np.nan), "티커": "CCC"}])
    out, rej = _apply_hard_filters(df, [])
    assert out["티커"].tolist() == ["BBB", "CCC"] and rej["어닝_임박"] == 1


def test_earnings_calendar_days_to_next_and_annotate(tmp_path):
    dates = pd.DataFrame({"티커": ["AAA", "AAA", "BBB"],
                          "accepted_et": pd.to_datetime(["2024-01-25 16:05", "2024-04-25 07:30", "2024-02-01 20:30"])})
    cal = EarningsCalendar(dates)
    assert cal.days_to_next("AAA", pd.Timestamp("2024-01-22")) == 3.0
    assert cal.days_to_next("AAA", pd.Timestamp("2024-01-25")) == 0.0  # 발표 당일
    assert cal.days_to_next("AAA", pd.Timestamp("2024-01-26")) == 90.0  # 다음 분기
    assert np.isnan(cal.days_to_next("AAA", pd.Timestamp("2024-05-01")))  # 이후 일정 모름
    assert np.isnan(cal.days_to_next("ZZZ", pd.Timestamp("2024-01-22")))
    ranked = pd.DataFrame({"티커": ["AAA", "BBB", "SPY"]})
    out = cal.annotate(ranked, pd.Timestamp("2024-01-30"))
    assert out["days_to_next_earnings"].tolist()[:2] == [86.0, 2.0] and np.isnan(out["days_to_next_earnings"].iloc[2])
    # 파일이 없으면 그대로 (기존 동작)
    empty = EarningsCalendar.load(tmp_path / "none.parquet")
    assert empty.annotate(ranked, pd.Timestamp("2024-01-30")) is ranked


def test_scheduled_only_drops_preannouncement_cluster():
    from paper_trading.earnings_calendar import scheduled_only
    dates = pd.DataFrame({"티커": ["AAA"] * 4,
                          "accepted_et": pd.to_datetime(["2024-01-08 08:00",   # 실적 경고 (정규 발표 3주 전)
                                                         "2024-01-30 16:05",   # 정규
                                                         "2024-04-30 16:05",   # 정규
                                                         "2024-07-30 16:05"])})
    kept = scheduled_only(dates)["accepted_et"].dt.strftime("%m-%d").tolist()
    assert kept == ["01-30", "04-30", "07-30"]


def test_calendar_mode_off_and_env(monkeypatch, tmp_path):
    path = tmp_path / "e.parquet"
    pd.DataFrame({"티커": ["AAA"], "accepted_et": pd.to_datetime(["2024-01-30 16:05"])}).to_parquet(path)
    assert len(EarningsCalendar.load(path)) == 1
    assert len(EarningsCalendar.load(path, mode="off")) == 0
    monkeypatch.setenv("BACKTEST_EARNINGS_CAL", "off")
    assert len(EarningsCalendar.load(path)) == 0
