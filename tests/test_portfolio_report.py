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
    assert "—" in _rsi_cell(None)        # 값 없음 → 회색 대시


def test_vol_cell():
    from screener.portfolio_report import _vol_cell
    assert "🔥" in _vol_cell(2.5)     # 급증
    assert "—" in _vol_cell(None)


class _FakeWs:
    """gspread Worksheet 흉내 — 머리글 붙이기만 기록한다."""

    def __init__(self, cols: int = 30) -> None:
        self.col_count = cols
        self.title = "내계좌"
        self.updates: list[tuple[list, str]] = []

    def add_cols(self, n: int) -> None:
        self.col_count += n

    def update(self, values, rng, value_input_option=None) -> None:
        self.updates.append((values, rng))


def test_ensure_headers_appends_only_missing_at_end():
    """이미 있는 열(웹이 먼저 붙인 손절가·매수이유 포함)은 옮기지 않고, 빠진 것만 끝에 붙인다."""
    from screener.portfolio_report import MY_ACCOUNT_HEADERS, _col_letter, _ensure_headers
    base = [h for h in MY_ACCOUNT_HEADERS if h not in ("ATR%", "베타(1년)", "평균상관", "실적일", "공시")]
    ws = _FakeWs(cols=len(base))
    out = _ensure_headers(ws, base + ["", ""], MY_ACCOUNT_HEADERS)
    assert out[: len(base)] == base
    assert out[len(base):] == ["ATR%", "베타(1년)", "평균상관", "실적일", "공시"]
    assert ws.col_count >= len(out)
    (values, rng), = ws.updates
    assert values == [["ATR%", "베타(1년)", "평균상관", "실적일", "공시"]]
    assert rng == f"{_col_letter(len(base) + 1)}1:{_col_letter(len(base) + 5)}1"  # 기존 열 바로 다음부터


def test_ensure_headers_noop_when_complete():
    from screener.portfolio_report import MY_ACCOUNT_HEADERS, _ensure_headers
    ws = _FakeWs()
    assert _ensure_headers(ws, list(MY_ACCOUNT_HEADERS), MY_ACCOUNT_HEADERS) == list(MY_ACCOUNT_HEADERS)
    assert ws.updates == []


def test_analysis_values_keys_match_headers():
    from screener.portfolio_report import ANALYSIS_HEADERS, RISK_HEADERS, _analysis_values
    vals = _analysis_values({"_price": 12.5, "_atr_pct": 0.031, "_beta": 1.234, "_avg_corr": 0.41,
                             "_earnings": "2026-10-28"}, "now")
    assert set(vals) == set(ANALYSIS_HEADERS + RISK_HEADERS)
    assert vals["ATR%"] == "3.10%"
    assert vals["베타(1년)"] == "1.23"
    assert vals["실적일"] == "'2026-10-28"  # 글자로 쓴다


def test_atr_pct_and_risk_stats():
    from screener.portfolio_report import _atr_pct, add_risk_stats
    idx = pd.bdate_range("2025-01-01", periods=300)
    close = np.linspace(10, 15, 300)
    df = pd.DataFrame({"High": close + 0.3, "Low": close - 0.3, "Close": close}, index=idx)
    atr = _atr_pct(df)
    assert atr is not None and 0 < atr < 0.2
    rng = np.random.default_rng(1)
    spy = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)), index=idx)
    a = spy * (1 + rng.normal(0, 0.001, 300))   # SPY 와 거의 같음 → 베타 ≈ 1
    b = pd.Series(50 * np.cumprod(1 + rng.normal(0, 0.01, 300)), index=idx)
    res = {"A": {"_daily_close": a}, "B": {"_daily_close": b}, "C": {}}
    add_risk_stats(res, spy)
    assert abs(res["A"]["_beta"] - 1.0) < 0.1
    assert res["C"]["_beta"] is None and res["C"]["_avg_corr"] is None
    assert res["A"]["_avg_corr"] is not None and abs(res["A"]["_avg_corr"]) < 0.5


def test_build_alerts_stop_near_earnings_and_missing_stop():
    from datetime import date
    from screener.portfolio_report import build_alerts
    records = [
        {"티커": "AAA", "손절가": "10"},          # 아래
        {"티커": "BBB", "손절가": "$100.00"},     # 2% 위 → 근접
        {"티커": "CCC", "손절가": "50"},          # 멀다 + 실적 3일 후
        {"티커": "DDD", "손절가": ""},            # 손절가 없음
    ]
    results = {"AAA": {"_price": 9.5}, "BBB": {"_price": 102.0},
               "CCC": {"_price": 80.0, "_earnings": "2026-10-19"}, "DDD": {"_price": 5.0, "_earnings": "2026-12-01"}}
    alerts, no_stop = build_alerts(records, results, today=date(2026, 10, 16))
    kinds = {(a["kind"], a["ticker"]) for a in alerts}
    assert kinds == {("stop", "AAA"), ("near", "BBB"), ("earn", "CCC")}
    assert no_stop == ["DDD"]


def test_build_html_report_shows_alerts():
    from screener.portfolio_report import build_html_report
    html = build_html_report({}, [{"kind": "stop", "ticker": "AAA", "text": "AAA 손절가 아래"}], ["DDD"])
    assert "알림 1건" in html and "AAA 손절가 아래" in html and "DDD" in html


def test_build_alerts_new_sec_filing_only():
    """3일 안의 🔴·🟠 공시만 알림 — 오래된 것·노랑은 종목 줄에만."""
    from datetime import date
    from screener.portfolio_report import build_alerts
    from screener.sec_filings import Flag
    new = Flag("offering", "자금 조달", "orange", "2026-10-15", "424B5", 1, "https://sec/x/")
    old = Flag("delisting", "상장폐지 경고", "red", "2026-09-01", "8-K", 1, "https://sec/y/")
    yellow = Flag("officer", "임원 변동", "yellow", "2026-10-16", "8-K", 1, "https://sec/z/")
    records = [{"티커": "AAA", "손절가": "1"}, {"티커": "AAA", "손절가": "1"}]  # 같은 종목 두 줄이어도 한 번
    res = {"AAA": {"_price": 5.0, "_sec": [new, old, yellow]}}
    alerts, _ = build_alerts(records, res, today=date(2026, 10, 16))
    sec = [a for a in alerts if a["kind"] == "sec"]
    assert len(sec) == 1 and "자금 조달" in sec[0]["text"] and "https://sec/x/" in sec[0]["text"]
