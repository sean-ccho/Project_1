"""ETF 탭 데이터 (screener.etf_scan) — 네트워크 없이 만든 가격으로."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.etf_universe import ETF_GROUPS, etf_group_of, etf_tickers  # noqa: E402
from screener.etf_scan import build_rows, payload, sheet_frame  # noqa: E402

DAYS = pd.bdate_range("2024-12-02", periods=300)


def _frame(paths: dict[str, np.ndarray]) -> pd.DataFrame:
    cols = {}
    for t, close in paths.items():
        cols[(t, "Open")], cols[(t, "High")], cols[(t, "Low")], cols[(t, "Close")] = close, close * 1.01, close * 0.99, close
        cols[(t, "Volume")] = np.full(len(close), 1e6)
        cols[(t, "Dividends")] = np.zeros(len(close))
    return pd.DataFrame(cols, index=DAYS)


def test_universe_has_unique_tickers_in_one_group_each():
    tickers = etf_tickers()
    assert len(tickers) == len(set(tickers)) == sum(len(g[3]) for g in ETF_GROUPS)
    assert set(etf_group_of()) == set(tickers)
    assert {"SPY", "BIL", "XIU.TO", "IBIT"} <= set(tickers)


def test_rows_returns_trend_and_risk():
    up = np.linspace(100, 130, len(DAYS))                                   # 꾸준히 오름
    down = np.concatenate([np.linspace(100, 120, 150), np.linspace(120, 90, 150)])  # 올랐다 떨어짐
    bil = np.linspace(100, 104.5, len(DAYS))                                 # 단기 국채 (1년 약 +4%)
    prices = _frame({"SPY": up, "BIL": bil, "XLK": down})
    info = {"SPY": {"name": "SPDR S&P 500", "expense": 0.000945, "aum": 8e11, "yield": 0.01, "currency": "USD"}}
    rows = {r["ticker"]: r for r in build_rows(prices, {}, info, ["SPY", "BIL", "XLK", "QQQ"])}
    assert set(rows) == {"SPY", "BIL", "XLK"}                               # 가격 없는 QQQ 는 빠짐
    spy, xlk = rows["SPY"], rows["XLK"]
    assert spy["group"] == "us_market" and xlk["group"] == "us_sector"
    assert abs(spy["ret"]["y1"] - (up[-1] / up[-253] - 1)) < 1e-12
    assert abs(spy["ret"]["ytd"] - (up[-1] / up[DAYS.year < DAYS[-1].year][-1] - 1)) < 1e-12
    assert spy["above_200"] is True and spy["trend_12m"] is True             # 1년 수익 > BIL
    assert xlk["above_200"] is False and xlk["trend_12m"] is False
    assert abs(xlk["mdd_1y"] - (90 / 120 - 1)) < 1e-3                        # 고점 120 → 90
    assert spy["mdd_1y"] == 0.0 and spy["corr_spy"] == 1.0
    assert spy["name"] == "SPDR S&P 500" and xlk["name"] is None             # 정보 못 받은 칸은 None
    assert set(spy["cross"]) == {"daily", "weekly", "monthly"}
    assert spy["price"] == up[-1]                                            # 배당 0 이라 분할만 반영 = 받은 값


def test_payload_is_strict_json_and_sheet_has_korean_columns():
    prices = _frame({"SPY": np.linspace(100, 130, len(DAYS)), "BIL": np.linspace(100, 104, len(DAYS))})
    rows = build_rows(prices, {}, {}, ["SPY", "BIL"])
    data = payload(rows)
    json.dumps(data, allow_nan=False)                                        # NaN 없음
    assert data["as_of"] == DAYS[-1].date().isoformat()
    assert [g["key"] for g in data["groups"]] == [g[0] for g in ETF_GROUPS]
    assert next(g for g in data["groups"] if g["key"] == "us_market")["tickers"] == ["SPY"]
    sheet = sheet_frame(rows)
    assert list(sheet.columns[:6]) == ["묶음", "티커", "이름", "운용사", "가격", "통화"]
    assert sheet.loc[sheet["티커"] == "SPY", "12개월 추세"].item() == "살아 있음"
