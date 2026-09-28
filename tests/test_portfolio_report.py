"""내계좌 포트폴리오 리포트 기본 테스트."""
import pandas as pd
import numpy as np
import pytest


def _make_ohlcv(n=60) -> pd.DataFrame:
    """간단한 더미 OHLCV 데이터프레임 생성."""
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    close = pd.Series(np.linspace(10, 15, n) + np.random.default_rng(42).normal(0, 0.2, n))
    df = pd.DataFrame({
        "Open": close - 0.1,
        "High": close + 0.3,
        "Low": close - 0.3,
        "Close": close,
        "Volume": np.random.default_rng(42).integers(100_000, 500_000, n).astype(float),
    }, index=idx)
    return df


def test_calc_indicators_returns_dict():
    from screener.portfolio_report import _calc_indicators
    df = _make_ohlcv(60)
    result = _calc_indicators(df)
    assert isinstance(result, dict)
    assert "rsi" in result
    assert "macd_hist" in result
    assert "volume_ratio" in result


def test_calc_indicators_rsi_range():
    from screener.portfolio_report import _calc_indicators
    df = _make_ohlcv(60)
    result = _calc_indicators(df)
    if result["rsi"] is not None:
        assert 0 <= result["rsi"] <= 100, "RSI는 0~100 범위여야 합니다"


def test_calc_indicators_insufficient_data():
    from screener.portfolio_report import _calc_indicators
    df = _make_ohlcv(10)  # 30개 미만
    result = _calc_indicators(df)
    assert result["rsi"] is None


def test_resample_4h():
    from screener.portfolio_report import _resample_4h
    idx = pd.date_range("2025-01-01", periods=200, freq="h")
    df_1h = pd.DataFrame({
        "Open": np.ones(200),
        "High": np.ones(200) + 0.5,
        "Low": np.ones(200) - 0.5,
        "Close": np.ones(200),
        "Volume": np.ones(200) * 1000,
    }, index=idx)
    df_4h = _resample_4h(df_1h)
    assert not df_4h.empty
    assert len(df_4h) < len(df_1h)


def test_build_html_report_contains_ticker():
    from screener.portfolio_report import build_html_report
    dummy_results = {
        "NBM.V": {
            "1H": {"rsi": 55.0, "macd": 0.01, "macd_signal": 0.008, "macd_hist": 0.002, "volume_ratio": 1.3},
            "4H": {"rsi": 60.0, "macd": 0.02, "macd_signal": 0.015, "macd_hist": 0.005, "volume_ratio": 1.8},
            "Daily": {"rsi": 45.0, "macd": -0.01, "macd_signal": -0.005, "macd_hist": -0.005, "volume_ratio": 0.9},
            "Weekly": {"rsi": None, "macd": None, "macd_signal": None, "macd_hist": None, "volume_ratio": None},
            "Monthly": {"rsi": 50.0, "macd": 0.005, "macd_signal": 0.003, "macd_hist": 0.002, "volume_ratio": 1.0},
        }
    }
    html = build_html_report(dummy_results)
    assert "NBM.V" in html
    assert "RSI" in html
    assert "MACD" in html
    assert "TradingView" in html


def test_rsi_cell_colors():
    from screener.portfolio_report import _rsi_cell
    assert "c0392b" in _rsi_cell(75.0)   # 과매수 → 빨강
    assert "1e8449" in _rsi_cell(25.0)   # 과매도 → 초록
    assert "N/A" in _rsi_cell(None)


def test_vol_cell():
    from screener.portfolio_report import _vol_cell
    assert "🔥" in _vol_cell(2.5)     # 급증
    assert "N/A" in _vol_cell(None)
