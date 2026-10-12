"""피처 캐시 지문 — 차트 설정 줄은 바꿔도 지문이 그대로, 피처 설정 줄은 바꾸면 달라진다."""
from paper_trading.backtest import _hashed_bytes

CONFIG = b"RSI_PERIOD = 14\nCHARTS_ENABLED = True  # on/off\nCHARTS_TIMEFRAMES = ['1H']\nPATTERN_LOOKBACK_DAYS = 60\n"


def test_chart_lines_do_not_change_config_fingerprint():
    toggled = CONFIG.replace(b"CHARTS_ENABLED = True", b"CHARTS_ENABLED =  False")
    assert _hashed_bytes("screener/config.py", toggled) == _hashed_bytes("screener/config.py", CONFIG)


def test_feature_lines_still_change_fingerprint():
    tuned = CONFIG.replace(b"PATTERN_LOOKBACK_DAYS = 60", b"PATTERN_LOOKBACK_DAYS = 90")
    assert _hashed_bytes("screener/config.py", tuned) != _hashed_bytes("screener/config.py", CONFIG)


def test_other_files_hashed_as_is():
    assert _hashed_bytes("screener/features.py", CONFIG) == CONFIG
