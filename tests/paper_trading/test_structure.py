"""PT-1 구조 실험 스위치 테스트 (기본값 = 기존 동작)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from paper_trading import structure
from paper_trading.config_override import config_overrides
from screener import config as cfg


def _ranked() -> pd.DataFrame:
    return pd.DataFrame({"티커": ["LOW", "MID", "HIGH", "BAD"], "ATR%": [0.01, 0.02, 0.04, 8.6]})


def test_defaults_keep_current_behavior():
    assert cfg.PT1_BEAR_MODE == "none" and cfg.PT1_SIZING == "equal"
    assert not structure.bear_liquidate()
    # equal = 기존 공식 (남은 현금 ÷ 빈자리) — 평가액·변동성과 무관
    assert structure.slot_allocation(3000.0, 9999.0, 3, 2, _ranked(), "HIGH") == 1500.0


def test_overrides_switch_and_restore():
    with config_overrides({"PT1_BEAR_MODE": "liquidate", "PT1_SIZING": "inv_vol"}):
        assert structure.bear_liquidate() and cfg.PT1_SIZING == "inv_vol"
    assert cfg.PT1_BEAR_MODE == "none" and cfg.PT1_SIZING == "equal"


def test_inv_vol_factor_caps_at_one_and_ignores_outliers():
    r = _ranked()  # 정상값 중앙 ATR% = 0.02 (8.6 은 이상치로 제외)
    assert structure.inv_vol_factor(r, "HIGH") == pytest.approx(0.5)
    assert structure.inv_vol_factor(r, "LOW") == 1.0  # 변동성이 낮아도 1 넘게 사지 않는다
    assert structure.inv_vol_factor(r, "ZZZ") == 1.0
    assert structure.inv_vol_factor(None, "HIGH") == 1.0


def test_inv_vol_allocation_uses_equity_slot_and_cash_limit():
    with config_overrides({"PT1_SIZING": "inv_vol"}):
        # 평가액 9000 / 3자리 = 3000, HIGH 계수 0.5 → 1500
        assert structure.slot_allocation(5000.0, 9000.0, 3, 2, _ranked(), "HIGH") == pytest.approx(1500.0)
        # 현금이 모자라면 현금까지만
        assert structure.slot_allocation(800.0, 9000.0, 3, 1, _ranked(), "MID") == pytest.approx(800.0)
