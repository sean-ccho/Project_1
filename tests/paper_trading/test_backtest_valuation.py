"""PT-1 백테스트 평가: 종가가 빈 날(거래정지·상장폐지) 보유 종목은 0원이 아니라 마지막 종가로 평가한다."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from paper_trading.backtest import BtPosition, _close_on_or_before, _record_equity  # noqa: E402

DATES = pd.bdate_range("2026-09-01", periods=4)
CLOSES = pd.DataFrame({"AAA": [10.0, 11.0, np.nan, np.nan], "BBB": [20.0, 21.0, 22.0, 23.0],
                       "CCC": [np.nan] * 4}, index=DATES)


def _pos(ticker: str, shares: float) -> BtPosition:
    return BtPosition(ticker=ticker, entry_price=10.0, entry_date="2026-09-01", strategy="모멘텀",
                      star_rating="", ccs_score=0.5, sector="Technology", shares=shares)


def test_close_on_or_before_uses_last_valid_close():
    assert _close_on_or_before(CLOSES, DATES[1], "AAA") == 11.0
    assert _close_on_or_before(CLOSES, DATES[3], "AAA") == 11.0  # 끊긴 뒤에도 마지막 종가
    assert math.isnan(_close_on_or_before(CLOSES, DATES[3], "CCC"))  # 한 번도 없으면 nan
    assert math.isnan(_close_on_or_before(CLOSES, DATES[3], "ZZZ"))  # 열이 없으면 nan


def test_record_equity_keeps_halted_position_value():
    points: list = []
    positions = [_pos("AAA", 10), _pos("BBB", 5)]
    _record_equity(points, positions, CLOSES, DATES[2], "2026-09-03", cash=100.0, use_capital=True)
    # 예전에는 AAA(종가 없음)를 0원으로 쳐서 210 → 이제 마지막 종가 11 로 320
    assert points == [("2026-09-03", 100.0 + 10 * 11.0 + 5 * 22.0)]
