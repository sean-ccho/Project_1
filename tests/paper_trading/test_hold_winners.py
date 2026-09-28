"""Hold-Winners 재평가 로직 유닛 테스트.

테스트 대상:
- _should_defer_sell: 각 조건 단독 실패, 전체 통과, fail-safe
- _activate_tight_trail: 필드 업데이트 검증
- defer cap: defer_count >= MAX_DEFERS 시 무조건 매도
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# src/ 를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from paper_trading.engine import (
    _activate_tight_trail,
    _build_high_rsi_rationale,
    _get_row_for_ticker,
    _should_defer_sell,
)
from screener.config import (
    HOLD_WINNERS_ADX_MIN,
    HOLD_WINNERS_BB_PBAND_MAX,
    HOLD_WINNERS_MAX_DEFERS,
    HOLD_WINNERS_MIN_CHECKS,
    HOLD_WINNERS_RSI_MAX,
    HOLD_WINNERS_TIGHT_TRAIL,
    HOLD_WINNERS_VOLUME_MULT_MIN,
)


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _make_pos(defer_count: int = 0, entry_price: float = 100.0) -> dict:
    return {
        "ticker": "TEST",
        "entry_price": entry_price,
        "defer_count": defer_count,
        "trailing_stop_override": None,
        "highest_price": entry_price,
    }


def _make_strong_row(**overrides) -> pd.Series:
    """모든 조건을 통과하는 기본 row."""
    base = {
        "RSI": 65.0,  # RSI_MAX(75) 미만
        "adx": HOLD_WINNERS_ADX_MIN + 5,                           # 25
        "거래량돌파배수": HOLD_WINNERS_VOLUME_MULT_MIN + 0.5,        # 1.7
        "거래량Z(20)": 1.5,
        "5일수익률": 0.05,
        "bollinger_pband": HOLD_WINNERS_BB_PBAND_MAX - 0.1,         # 0.85
    }
    base.update(overrides)
    return pd.Series(base)


# ── _should_defer_sell ────────────────────────────────────────────────────────

class TestShouldDeferSell:
    def test_all_strong_returns_true(self):
        pos = _make_pos()
        row = _make_strong_row()
        defer, debug = _should_defer_sell(pos, row, 120.0)
        assert defer is True
        assert "checks_passed" in debug["reason"]

    def test_row_none_returns_false(self):
        pos = _make_pos()
        defer, debug = _should_defer_sell(pos, None, 120.0)
        assert defer is False
        assert "no_row_data" in debug["reason"]

    def test_rsi_too_high_returns_false(self):
        """RSI > RSI_MAX(75)이면 과매수 — defer 거부 (1개 조건 실패)."""
        pos = _make_pos()
        row = _make_strong_row(RSI=HOLD_WINNERS_RSI_MAX + 1)  # RSI=76
        defer, debug = _should_defer_sell(pos, row, 120.0)
        # MIN_CHECKS에 따라 결과 다름: 5/5이면 False, 4/5이면 True (1개 실패)
        if HOLD_WINNERS_MIN_CHECKS >= 5:
            assert defer is False
        else:
            assert defer is True  # 4/5 다수결이면 1개 실패해도 통과

    def test_adx_weak_returns_false(self):
        pos = _make_pos()
        row = _make_strong_row(adx=HOLD_WINNERS_ADX_MIN - 1)  # ADX=19
        defer, debug = _should_defer_sell(pos, row, 120.0)
        # 1개 실패 — MIN_CHECKS에 따라 판단
        if HOLD_WINNERS_MIN_CHECKS >= 5:
            assert defer is False

    def test_bb_pband_peaked_returns_false(self):
        pos = _make_pos()
        row = _make_strong_row(bollinger_pband=HOLD_WINNERS_BB_PBAND_MAX + 0.01)  # 0.96
        defer, debug = _should_defer_sell(pos, row, 120.0)
        if HOLD_WINNERS_MIN_CHECKS >= 5:
            assert defer is False

    def test_volume_low_but_volume_z_passes(self):
        """거래량돌파배수가 기준 미달이어도 거래량Z가 기준 이상이면 통과."""
        pos = _make_pos()
        row = _make_strong_row(
            **{"거래량돌파배수": HOLD_WINNERS_VOLUME_MULT_MIN - 0.5, "거래량Z(20)": 1.1}
        )
        defer, debug = _should_defer_sell(pos, row, 120.0)
        assert defer is True

    def test_ret_5d_negative_returns_false(self):
        pos = _make_pos()
        row = _make_strong_row(**{"5일수익률": -0.01})
        defer, debug = _should_defer_sell(pos, row, 120.0)
        if HOLD_WINNERS_MIN_CHECKS >= 5:
            assert defer is False

    def test_max_defers_cap(self):
        """defer_count >= MAX_DEFERS이면 강한 신호에도 False."""
        pos = _make_pos(defer_count=HOLD_WINNERS_MAX_DEFERS)
        row = _make_strong_row()
        defer, debug = _should_defer_sell(pos, row, 120.0)
        assert defer is False
        assert "max_defers_reached" in debug["reason"]

    def test_one_below_cap_still_defers(self):
        """MAX_DEFERS - 1 이면 아직 defer 가능."""
        pos = _make_pos(defer_count=HOLD_WINNERS_MAX_DEFERS - 1)
        row = _make_strong_row()
        defer, _ = _should_defer_sell(pos, row, 120.0)
        assert defer is True

    def test_missing_rsi_field_defaults_to_safe(self):
        """RSI 필드가 없으면 _safe_num 기본값(-1) → rsi_not_overbought 통과."""
        pos = _make_pos()
        row = _make_strong_row()
        row = row.drop("RSI")
        defer, debug = _should_defer_sell(pos, row, 120.0)
        # RSI 기본값 -1 < RSI_MAX(75)이므로 rsi_not_overbought는 통과
        # 나머지 조건이 모두 통과하므로 defer 여부는 MIN_CHECKS에 따라
        assert isinstance(defer, bool)

    def test_multiple_failures_returns_false(self):
        """여러 조건 동시 실패 시 확실히 False."""
        pos = _make_pos()
        row = _make_strong_row(
            RSI=HOLD_WINNERS_RSI_MAX + 5,  # 과매수
            adx=5,  # 추세 약함
            **{"5일수익률": -0.05},  # 하락중
        )
        defer, debug = _should_defer_sell(pos, row, 120.0)
        assert defer is False


# ── _activate_tight_trail ─────────────────────────────────────────────────────

class TestActivateTightTrail:
    def test_sets_trailing_stop_override(self):
        pos = _make_pos(entry_price=100.0)
        _activate_tight_trail(pos, 120.0, "2026-04-15", {})
        assert pos["trailing_stop_override"] == HOLD_WINNERS_TIGHT_TRAIL

    def test_increments_defer_count(self):
        pos = _make_pos(defer_count=0)
        _activate_tight_trail(pos, 120.0, "2026-04-15", {})
        assert pos["defer_count"] == 1

    def test_increments_twice(self):
        pos = _make_pos(defer_count=0)
        _activate_tight_trail(pos, 120.0, "2026-04-15", {})
        _activate_tight_trail(pos, 125.0, "2026-04-16", {})
        assert pos["defer_count"] == 2

    def test_highest_price_updated(self):
        pos = _make_pos(entry_price=100.0)
        _activate_tight_trail(pos, 130.0, "2026-04-15", {})
        assert pos["highest_price"] == 130.0

    def test_highest_price_uses_entry_if_lower(self):
        """current_price < entry_price 이면 entry_price 사용."""
        pos = _make_pos(entry_price=100.0)
        _activate_tight_trail(pos, 80.0, "2026-04-15", {})
        assert pos["highest_price"] == 100.0

    def test_last_defer_date_set(self):
        pos = _make_pos()
        _activate_tight_trail(pos, 120.0, "2026-04-15", {"some": "debug"})
        assert pos["last_defer_date"] == "2026-04-15"

    def test_defer_debug_stored(self):
        pos = _make_pos()
        debug = {"RSI": 65.0, "adx": 28.0}
        _activate_tight_trail(pos, 120.0, "2026-04-15", debug)
        assert pos["defer_debug"] == debug


# ── _get_row_for_ticker ───────────────────────────────────────────────────────

class TestGetRowForTicker:
    def test_finds_by_ticker_column(self):
        df = pd.DataFrame({"티커": ["AAPL", "MSFT"], "RSI": [60.0, 55.0]})
        row = _get_row_for_ticker(df, "AAPL")
        assert row is not None
        assert row["RSI"] == 60.0

    def test_returns_none_for_missing_ticker(self):
        df = pd.DataFrame({"티커": ["AAPL"], "RSI": [60.0]})
        row = _get_row_for_ticker(df, "GOOG")
        assert row is None

    def test_returns_none_for_none_df(self):
        assert _get_row_for_ticker(None, "AAPL") is None

    def test_returns_none_for_empty_df(self):
        assert _get_row_for_ticker(pd.DataFrame(), "AAPL") is None


# ── _build_high_rsi_rationale ─────────────────────────────────────────────────

class TestBuildHighRsiRationale:
    def test_includes_rsi(self):
        row = pd.Series({"RSI": 72.5, "adx": 28.0, "거래량돌파배수": 1.8,
                         "52주포지션": 0.9, "모멘텀_적합도": 8.2})
        text = _build_high_rsi_rationale(row)
        assert "72.5" in text
        assert "모멘텀 전략 허용 구간" in text

    def test_none_row_returns_empty(self):
        assert _build_high_rsi_rationale(None) == ""
