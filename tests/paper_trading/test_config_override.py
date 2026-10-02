"""config_overrides 테스트 (pandas 불필요)."""

import pytest

from paper_trading.config_override import config_overrides
from screener import config as cfg


def test_scalar_and_nested_dict_are_restored():
    orig_version = cfg.CCS_VERSION
    orig_trail = cfg.EXIT_PARAMS["모멘텀"]["trail_activate_pct"]
    orig_stop = cfg.EXIT_PARAMS["모멘텀"]["stop_loss"]
    exit_params_obj = cfg.EXIT_PARAMS
    momentum_obj = cfg.EXIT_PARAMS["모멘텀"]

    with config_overrides({"CCS_VERSION": "v2", "EXIT_PARAMS": {"모멘텀": {"trail_activate_pct": 0.05}}}):
        assert cfg.CCS_VERSION == "v2"
        assert cfg.EXIT_PARAMS["모멘텀"]["trail_activate_pct"] == 0.05
        assert cfg.EXIT_PARAMS["모멘텀"]["stop_loss"] == orig_stop  # 나머지 키는 유지
        assert cfg.EXIT_PARAMS is exit_params_obj                     # 같은 dict 객체

    assert cfg.CCS_VERSION == orig_version
    assert cfg.EXIT_PARAMS["모멘텀"]["trail_activate_pct"] == orig_trail
    assert cfg.EXIT_PARAMS is exit_params_obj
    assert cfg.EXIT_PARAMS["모멘텀"] is momentum_obj


def test_new_nested_key_is_removed_after():
    with config_overrides({"PT2_PARAMS": {"_tmp_key": 1}}):
        assert cfg.PT2_PARAMS["_tmp_key"] == 1
    assert "_tmp_key" not in cfg.PT2_PARAMS


def test_restored_even_on_exception():
    orig = cfg.PT1_REPLACE_ENABLED
    with pytest.raises(RuntimeError):
        with config_overrides({"PT1_REPLACE_ENABLED": not orig}):
            raise RuntimeError("boom")
    assert cfg.PT1_REPLACE_ENABLED == orig


def test_none_to_list_and_back():
    assert cfg.CANDIDATE_ALLOWED_STRATEGIES is None
    with config_overrides({"CANDIDATE_ALLOWED_STRATEGIES": ["모멘텀"]}):
        assert cfg.CANDIDATE_ALLOWED_STRATEGIES == ["모멘텀"]
    assert cfg.CANDIDATE_ALLOWED_STRATEGIES is None


def test_rejects_names_not_read_at_runtime():
    with pytest.raises(ValueError):
        with config_overrides({"ADX_BUY_MIN": 30}):
            pass
