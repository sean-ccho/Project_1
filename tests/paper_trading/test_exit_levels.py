"""exit_levels(메일 '매도 기준' 표시) 가 check_sell_conditions(실제 매도 판단)와 같은 선을 쓰는지 확인."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from paper_trading.engine import check_sell_conditions, exit_levels  # noqa: E402


def _pos(strategy: str, highest: float) -> dict:
    return {"ticker": "AAA", "entry_date": "2026-09-01", "entry_price": 100.0, "highest_price": highest,
            "strategy": strategy}


@pytest.mark.parametrize("strategy, highest, as_of, expect", [
    # 바닥반등 5일째, 고점 105 → 목표가 +18%, 트레일 고점 −6%, 12일째부터 시간익절 +7%
    ("📉 바닥반등", 105.0, "2026-09-06", ("목표가", 118.0, "트레일링", 98.7, ("up", 12, 107.0))),
    # 모멘텀 10일째(시간익절 8일 지남), 한 번도 안 오름 → 시간익절 +5%, 손절 −10%, 19일째부터 +2% 아래면 매도
    ("📈 모멘텀", 100.0, "2026-09-11", ("시간익절", 105.0, "손절", 90.0, ("down", 19, 102.0))),
    # 모멘텀 20일째(장기보유 18일 지남) → 아래 선이 +2% 로 올라감
    ("📈 모멘텀", 100.0, "2026-09-21", ("시간익절", 105.0, "장기보유", 102.0, None)),
])
def test_exit_levels_match_sell_decision(strategy, highest, as_of, expect):
    up_tag, up, down_tag, down, nxt = expect
    lv = exit_levels(_pos(strategy, highest), as_of)
    assert lv["up_tag"] == up_tag and lv["up"] == pytest.approx(up)
    assert lv["down_tag"] == down_tag and lv["down"] == pytest.approx(down)
    if nxt is None:
        assert lv["next"] is None
    else:
        assert (lv["next"]["side"], lv["next"]["day"]) == nxt[:2] and lv["next"]["price"] == pytest.approx(nxt[2])

    pos = _pos(strategy, highest)
    sell_up, reason_up = check_sell_conditions(pos, lv["up"] * 1.001, as_of)
    sell_down, reason_down = check_sell_conditions(pos, lv["down"] * 0.999, as_of)
    assert sell_up and reason_up.startswith(up_tag)
    assert sell_down and reason_down.startswith(down_tag)
    assert check_sell_conditions(pos, (lv["up"] + lv["down"]) / 2, as_of) == (False, "")
