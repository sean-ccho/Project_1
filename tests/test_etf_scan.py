"""ETF 탭 데이터 (screener.etf_scan) — 네트워크 없이 만든 가격으로."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.etf_universe import ETF_GROUPS, ETF_ROLES, ETF_TWINS, etf_group_of, etf_tickers  # noqa: E402
from screener.etf_scan import (build_rows, detail_payload, momentum_list, parse_funds, payload, pick_roles,  # noqa: E402
                               sheet_frame)

DAYS = pd.bdate_range("2024-12-02", periods=300)


def _frame(paths: dict[str, np.ndarray], divs: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    cols = {}
    for t, close in paths.items():
        cols[(t, "Open")], cols[(t, "High")], cols[(t, "Low")], cols[(t, "Close")] = close, close * 1.01, close * 0.99, close
        cols[(t, "Volume")] = np.full(len(close), 1e6)
        cols[(t, "Dividends")] = (divs or {}).get(t, np.zeros(len(close)))
    return pd.DataFrame(cols, index=DAYS)


def _adjusted(raw: np.ndarray, div: np.ndarray) -> np.ndarray:
    """Yahoo 방식 배당 조정 종가: A_t = C_t × Π(배당락일 j > t) (1 − D_j / C_(j−1))."""
    f = np.ones(len(raw))
    for t in range(len(raw) - 2, -1, -1):
        f[t] = f[t + 1] * (1 - div[t + 1] / raw[t]) if div[t + 1] > 0 else f[t + 1]
    return raw * f


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


def test_chart_price_columns_ignore_dividends_and_bil_has_no_trend():
    """배당 큰 ETF: 200일선 · 52주 고점은 차트 가격(분할만 반영), 수익률은 배당 포함. BIL 은 12개월 배지 없음."""
    raw = np.linspace(100, 95, len(DAYS))                                   # 가격은 조금씩 내림
    div = np.zeros(len(DAYS))
    div[20::21] = 0.8                                                        # 매달 0.8 (연 10% 쯤)
    adj = _adjusted(raw, div)
    prices = _frame({"TLT": adj, "BIL": np.linspace(100, 104.5, len(DAYS))}, divs={"TLT": div})
    rows = {r["ticker"]: r for r in build_rows(prices, {}, {}, ["TLT", "BIL"])}
    tlt = rows["TLT"]
    assert abs(tlt["price"] - raw[-1]) < 1e-9
    assert adj[-1] > adj[-200:].mean() and tlt["above_200"] is False        # 배당 포함 가격이면 '위'였다
    assert abs(tlt["high_52w_gap"] - (raw[-1] / raw[-252:].max() - 1)) < 1e-9
    assert tlt["ret"]["y1"] > 0                                              # 가격은 내렸어도 총수익은 +
    assert rows["BIL"]["trend_12m"] is None


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


def test_twins_and_roles_use_known_tickers_once():
    tickers = set(etf_tickers())
    in_twins = [t for _k, _ko, _en, _kind, ts in ETF_TWINS for t in ts]
    assert len(in_twins) == len(set(in_twins)) and set(in_twins) <= tickers   # 한 ETF 는 한 묶음에만
    assert {kind for *_x, kind, _ts in ETF_TWINS} == {"same", "similar"}
    assert all(set(c) <= tickers for _k, _ko, _en, c in ETF_ROLES)


class _FakeFunds:
    """yfinance FundsData 흉내 — 칸 하나는 읽다가 실패."""
    description = "Tracks the S&P 500."
    top_holdings = pd.DataFrame({"Name": ["NVIDIA Corp", "Apple Inc", "Cash"], "Holding Percent": [0.0839, 0.0738, None]},
                                index=pd.Index(["NVDA", "AAPL", "CASH"], name="Symbol"))
    sector_weightings = {"technology": 0.35, "realestate": 0.02, "weird_key": 0.5, "energy": 0.0}
    asset_classes = {"cashPosition": -0.0012, "stockPosition": 1.0004, "bondPosition": 0.0, "otherPosition": 0.0007}

    @property
    def bond_ratings(self):
        raise RuntimeError("Yahoo 실패")


def test_parse_funds_keeps_good_fields_and_skips_bad():
    d = parse_funds(_FakeFunds())
    assert d["desc"] == "Tracks the S&P 500."
    assert d["holdings"] == [{"s": "NVDA", "n": "NVIDIA Corp", "w": 0.0839}, {"s": "AAPL", "n": "Apple Inc", "w": 0.0738}]
    assert d["sectors"] == {"Technology": 0.35, "Real Estate": 0.02}       # 모르는 키 · 0 은 뺌
    assert d["assets"]["cash"] == 0.0 and d["assets"]["stock"] > 1         # 음수 현금은 0
    assert d["ratings"] is None                                            # 주식 ETF 는 등급 안 봄

    class Bond:
        description = ""
        top_holdings = pd.DataFrame(columns=["Name", "Holding Percent"])
        sector_weightings = {}
        asset_classes = {"bondPosition": 0.99, "cashPosition": 0.01}
        bond_ratings = {"aaa": 0.03, "aa": 0.73, "bbb": 0.12, "us_government": 0.52}

    b = parse_funds(Bond())
    assert b["desc"] is None and b["holdings"] == [] and b["sectors"] is None
    assert b["ratings"] == {"aaa": 0.03, "aa": 0.73, "bbb": 0.12, "us_government": 0.52}
    json.dumps(detail_payload({"BND": b}), allow_nan=False)


def _row(t, group="us_market", expense=None, aum=None, m6=None, up=True):
    return {"ticker": t, "group": group, "expense": expense, "aum": aum, "ret": {"m6": m6},
            "above_200": up, "trend_12m": up}


def test_pick_roles_cheapest_big_enough_then_bigger_on_tie():
    rows = [_row("SPY", expense=0.000945, aum=8e11), _row("VOO", expense=0.0003, aum=1.7e12),
            _row("IVV", expense=0.0003, aum=8.8e11),
            _row("BITB", "crypto", 0.0020, 3e9), _row("ARKB", "crypto", 0.0021, 2.9e9),
            _row("IBIT", "crypto", 0.0025, 6.7e10), _row("GBTC", "crypto", 0.015, 1e10),
            _row("FBTC", "crypto", 0.0019, 5e8),                    # 제일 싸지만 운용자산 미달
            _row("GLDM", "commodities", None, 3e10), _row("GLD", "commodities", 0.004, 1.4e11)]
    roles = {p["key"]: p for p in pick_roles(rows)}
    assert roles["sp500"]["pick"] == "VOO" and roles["sp500"]["alts"] == ["IVV", "SPY"]
    assert roles["bitcoin"]["pick"] == "BITB" and roles["bitcoin"]["alts"][0] == "FBTC"
    assert roles["gold"]["pick"] == "GLD" and roles["gold"]["alts"] == ["GLDM"]   # 보수 모르면 못 고름 · 끝
    assert "nasdaq100" not in roles                                               # 후보가 하나도 없으면 빠짐


def test_momentum_list_needs_trend_and_dedupes_twins():
    rows = [_row("SMH", "industry", m6=0.30), _row("SOXX", "industry", m6=0.35),   # 같은 묶음 → SOXX 하나
            _row("XLV", "us_sector", m6=0.27), _row("IGV", "industry", m6=0.50, up=False),
            _row("QQQ", "us_market", m6=0.90), _row("EWT", "international", m6=0.25),
            _row("CPER", "commodities", m6=0.12), _row("VNQ", "real_estate", m6=0.01),
            _row("XLE", "us_sector", m6=0.17), _row("XLB", "us_sector", m6=None)]
    assert momentum_list(rows) == ["SOXX", "XLV", "EWT", "XLE", "CPER"]


def test_payload_has_picks_twins_and_sheet_extra_columns():
    prices = _frame({"SPY": np.linspace(100, 130, len(DAYS)), "VOO": np.linspace(100, 130, len(DAYS)),
                     "BIL": np.linspace(100, 104, len(DAYS))})
    info = {"SPY": {"expense": 0.000945, "aum": 8e11}, "VOO": {"expense": 0.0003, "aum": 1.7e12}}
    holdings = {"VOO": {"holdings": [{"s": "NVDA", "n": "NVIDIA", "w": 0.084}, {"s": "AAPL", "n": "Apple", "w": 0.074}]}}
    rows = build_rows(prices, {}, info, ["SPY", "VOO", "BIL"], holdings)
    by = {r["ticker"]: r for r in rows}
    assert abs(by["VOO"]["top10_weight"] - 0.158) < 1e-12 and by["SPY"]["top10_weight"] is None
    data = payload(rows)
    assert data["picks"]["roles"][0]["pick"] == "VOO"
    assert {"key": "sp500", "ko": "S&P 500", "en": "S&P 500", "kind": "same", "tickers": ["SPY", "VOO"]} in data["twins"]
    sheet = sheet_frame(rows, holdings).set_index("티커")
    assert list(sheet.columns[-3:]) == ["추천", "같은 것", "상위 보유"]
    assert sheet.loc["VOO", "추천"].startswith("⭐") and sheet.loc["SPY", "같은 것"] == "VOO"
    assert sheet.loc["VOO", "상위 보유"] == "NVDA 8.4% · AAPL 7.4%"
