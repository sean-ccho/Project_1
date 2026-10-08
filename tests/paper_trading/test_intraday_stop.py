"""장중 손절(PT1_STOP_INTRADAY) 판정 테스트."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from paper_trading.engine import _resolve_exit_params, intraday_stop_exit

STRAT = "📉 바닥반등"
P = _resolve_exit_params(STRAT)


def _pos(highest=100.0, **kw):
    return {"entry_price": 100.0, "highest_price": highest, "strategy": STRAT, **kw}


def test_no_hit():
    assert intraday_stop_exit(_pos(), 100.0, 100 * (1 - P["stop_loss"]) + 0.01) is None


def test_stop_hit_fills_at_level():
    level = 100 * (1 - P["stop_loss"])
    fill, reason = intraday_stop_exit(_pos(), 99.0, level - 1)
    assert abs(fill - level) < 1e-9 and reason.startswith("손절(장중")


def test_gap_down_fills_at_open():
    fill, _ = intraday_stop_exit(_pos(), 80.0, 79.0)
    assert fill == 80.0


def test_trailing_level_used_when_higher():
    highest = 120.0
    trail = highest * (1 - P["trailing_stop"])
    fill, reason = intraday_stop_exit(_pos(highest=highest), 115.0, trail - 0.5)
    assert abs(fill - trail) < 1e-9 and reason.startswith("트레일링(장중")
    override = intraday_stop_exit(_pos(highest=highest, trailing_stop_override=0.02), 119.0, 117.5)
    assert override is not None and abs(override[0] - highest * 0.98) < 1e-9
