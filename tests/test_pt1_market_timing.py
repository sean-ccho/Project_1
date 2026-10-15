"""PT-1 + A 시장 타이밍 테스트 — 매도 우선순위·비중 축소 비율, 매수 한도, 매수 막기, 현금 이자, A 비중 날짜."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pt1_market_timing as pm  # noqa: E402


def test_sell_priority_and_trim_fraction():
    keep = (False, "")
    assert pm.timing_sell((True, "손절"), "full", 0.0, True, 0.9) == (True, "손절")       # PT-1 매도가 먼저
    assert pm.timing_sell(keep, "probe", 0.0, True, 0.9) == keep                          # 통과 모드는 원래대로
    assert pm.timing_sell(keep, "trend", 0.0, False, 0.0) == (True, pm.EXIT_REASON)
    ok, reason = pm.timing_sell(keep, "full", 0.6, True, 0.9)
    assert ok and reason.startswith(pm.TRIM_TAG)
    assert float(reason[len(pm.TRIM_TAG):]) == pytest.approx(1 - 0.6 / 0.9)
    assert pm.timing_sell(keep, "full", 0.6, True, 0.64) == keep                          # 5%p 안 → 그대로
    assert pm.timing_sell(keep, "full", 0.6, False, 0.9) == keep                          # 결정일이 아니면 안 줄임
    assert pm.timing_sell(keep, "trend", 0.5, True, 0.9) == keep                          # A추세는 w>0 이면 그대로


def test_buy_cap_and_blocking():
    assert pm.buy_room(2000, 5000, 0.6) == 0.0                                            # 보유 3000 = 한도 3000
    assert pm.capped_allocation(1500, 3000, 5000, 0.8, 2, "full") == pytest.approx(1000)  # 여유 2000 ÷ 빈 자리 2
    assert pm.capped_allocation(1500, 3000, 5000, 1.0, 2, "full") == 1500
    assert pm.capped_allocation(1500, 3000, 5000, 0.5, 2, "trend") == 1500
    assert pm.block_buys("probe", 0.0, 1000, 5000) is False
    assert pm.block_buys("trend", 0.0, 1000, 5000) and pm.block_buys("full", 0.0, 1000, 5000)
    assert pm.block_buys("full", 0.6, 2000, 5000) and not pm.block_buys("trend", 0.6, 2000, 5000)
    assert not pm.block_buys("full", 0.8, 3000, 5000)


def test_cash_interest_uses_previous_day_cash_share():
    idx = pd.bdate_range("2024-01-01", periods=3)
    eq = pd.Series([100.0, 101.0, 102.0], index=idx)
    cash = pd.Series([50.0, 0.0, 0.0], index=idx)
    rf = pd.Series(0.001, index=idx)
    out = pm.with_cash_interest(eq, cash, rf)
    assert out.iloc[0] == pytest.approx(0.01 + 0.5 * 0.001)
    assert out.iloc[1] == pytest.approx(102 / 101 - 1)


@pytest.mark.skipif(not pm.ld.PRICES.exists(), reason="lab 고정 시세 없음")
def test_a_weights_apply_from_month_end_decision():
    w, dec = pm.a_weights()
    assert "2025-04-30" in dec and "2025-05-30" in dec
    assert w["2025-04-30"] == 0.0 and w["2025-05-15"] == 0.0                             # 4월 말 결정(0%) → 5월 내내
    assert w["2025-05-30"] == pytest.approx(0.707, abs=0.002)
    assert w["2025-06-02"] == w["2025-05-30"]


def test_spec_and_config():
    s = pm.spec()
    assert "250일" in s and "12%" in s and "0.05/2" in s and "본 뒤" in s
    assert pm.A_CFG == {"trend": "sma", "n": 250, "sigma": 0.12, "vl": 21, "sched": "M"}
    assert pm.TESTS == ("A추세", "A전체") and pm.P_BONF == pytest.approx(0.025)
