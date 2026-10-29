"""전체 분석(run_universe_scan.py) — 종목 나누기·이름·내부자 시장 집계·신규 상장·웹 JSON·큰 시트 쓰기 (네트워크 없음)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_universe_scan as rus  # noqa: E402
import screener.exporter as ex  # noqa: E402
import screener.insider as ins  # noqa: E402
from data.ticker_fetcher import clean_security_name, listing_symbol  # noqa: E402
from web_export import universe as wu  # noqa: E402


def test_split_universe_removes_sp500_with_dot_or_dash() -> None:
    sp, rest = rus.split_universe(["AAPL", "BRK-B", "GME", "AGM-A", "GME"], ["AAPL", "BRK.B"], ["SPY"])
    assert sp == ["AAPL", "BRK-B", "SPY"]
    assert rest == ["GME", "AGM-A"]


def test_listing_symbol_and_name() -> None:
    assert listing_symbol("AGM.A") == "AGM-A"  # 종류주는 야후 형식으로 살린다
    assert listing_symbol("AAC.W") is None and listing_symbol("AAC.U") is None
    assert listing_symbol("ACHRW") is None  # 나스닥 워런트
    assert listing_symbol("SNOW") == "SNOW"
    assert clean_security_name("Apple Inc. - Common Stock") == "Apple Inc."
    assert clean_security_name("Brown Forman Inc Class B Common Stock") == "Brown Forman Inc"
    assert clean_security_name("Federal Agricultural Mortgage Corporation Common Stock") == \
        "Federal Agricultural Mortgage Corporation"


def _table(header: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="tinytable"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


TICKER_HEADER = ["X", "Filing Date", "Trade Date", "Ticker", "Insider Name", "Title", "Trade Type", "Price", "Qty",
                 "Owned", "ΔOwn", "Value", "1d", "1w", "1m", "6m"]
MARKET_HEADER = TICKER_HEADER[:4] + ["Company Name"] + TICKER_HEADER[4:]


def test_insider_market_matches_per_ticker_summary() -> None:
    """시장 전체 표(회사명 칸이 하나 더 있음)를 티커별로 묶은 결과가 종목별 조회 집계와 같아야 한다."""
    trades = [
        ["", "2026-10-01", "2026-09-30", "ABC", "Kim", "CEO", "P - Purchase", "$10", "100", "1", "+1%", "+$1,000"],
        ["", "2026-10-02", "2026-10-01", "ABC", "Lee", "CFO", "P - Purchase", "$10", "200", "1", "+1%", "+$2,000"],
        ["", "2026-10-03", "2026-10-02", "ABC", "Park", "Dir", "S - Sale", "$11", "-50", "1", "-1%", "-$550"],
        ["", "2026-10-04", "2026-10-03", "BRK.B", "Buffett", "CEO", "S - Sale", "$500", "-1", "1", "-1%", "-$500"],
    ]
    pad = ["", "", "", ""]
    market_rows = [r[:4] + ["Some Co"] + r[4:] + pad for r in trades]
    html = _table(MARKET_HEADER, market_rows)
    df = ins.fetch_insider_market(90, page_size=5000, fetch_html=lambda url, timeout=0: html)
    got = df.set_index("티커")
    expected = ins._aggregate("ABC", [r + pad for r in trades[:3]])
    for col in ("insider_buy_count", "insider_sell_count", "insider_net_value", "insider_last_trade_date", "내부자(90일)"):
        assert got.loc["ABC", col] == expected[col]
    assert "클러스터매수" in got.loc["ABC", "내부자(90일)"]
    assert "BRK-B" in got.index  # 야후 형식


def test_insider_market_returns_empty_on_failure() -> None:
    def boom(url: str, timeout: int = 0) -> str:
        raise OSError("down")

    df = ins.fetch_insider_market(90, fetch_html=boom)
    assert df.empty and list(df.columns) == ins._RESULT_COLUMNS


def test_young_row_simple_indicators() -> None:
    idx = pd.bdate_range("2026-08-03", periods=30)
    close = pd.Series(np.linspace(10, 13, 30), index=idx)
    frame = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                          "Volume": [1000] * 29 + [3000]}, index=idx)
    r = rus.young_row("NEWCO", frame)
    assert r["첫거래일"] == "2026-08-03" and r["상장후거래일"] == 30
    assert abs(r["상장후수익률"] - 0.3) < 1e-9
    assert r["거래량돌파배수"] == 3.0
    assert r["RSI"] == 100.0  # 오르기만 했다
    assert 0 < r["상장후위치"] <= 1


def test_isb_and_sqz_text() -> None:
    isb = {"D": {"rev": {"date": "2026-10-14", "ago": 4}, "exit": {"date": "2026-09-01", "ago": 30}},
           "W": {"bottom": {"date": "2026-09-26", "ago": 3}}}
    assert wu.isb_text(isb) == "일 BUY·REV 10/14 · 주 ◇BOTTOM 9/26"
    assert wu.isb_text({}) == "" and wu.isb_text(None) == ""
    # 웹 배지와 같은 말: 상태(압축 · 발사) + 방향(막대 색), 힘 빠짐은 몇 봉째 + 압축 중이 아닐 때 2봉째부터 ⚠
    sqz = {"D": {"dot": "black", "run": 5, "color": "lime", "color_run": 3},
           "W": {"dot": "gray", "run": 0, "color": "green", "color_run": 2,
                 "fire": {"ago": 1, "run": 35, "color": "lime"}},
           "M": {"dot": "blue", "run": 0, "color": ""}}
    assert wu.sqz_text(sqz) == "일 압축 5봉째 ▲ 상승 중 | 주 ⤴ 위로 발사 1봉 전 ▲ 상승 힘 빠짐 2봉째 ⚠ | 월"
    assert wu.sqz_state_text({"dot": "black", "run": 9, "color": "maroon", "color_run": 4}) == "압축 9봉째 ▼ 하락 힘 빠짐 4봉째"
    assert wu.sqz_state_text({"dot": "gray", "run": 0, "color": "red", "color_run": 2,
                              "fire": {"ago": 0, "run": 2, "color": "red"}}) == "⤵ 아래로 발사 ▼ 하락 중"
    assert wu.sqz_state_text({"dot": "blue", "run": 0, "color": "green", "color_run": 1}) == "▲ 상승 힘 빠짐 1봉째"
    assert wu.sqz_state_text({"dot": "blue", "run": 0, "color": "green"}) == "▲ 상승 힘 빠짐"  # color_run 없는 옛 데이터
    assert wu.sqz_text(None) == ""


def test_payload_drops_index_etfs_and_writes_stock_files(tmp_path: Path) -> None:
    row = pd.Series({"티커": "AAPL", "회사": "Apple Inc.", "현재가격": 250.0, "market": "sp500",
                     "매수적합도_표시": "★★★☆☆ (6.5)", "판단": "1. 매수 후보", "RSI": float("nan")})
    info = wu.item_info(row, tv="NASDAQ:AAPL", insider=None, isb={}, sqz=None)
    assert info["tv"] == "NASDAQ:AAPL" and info["score"] == 6.5 and "first_date" not in info
    young = pd.Series({"티커": "NEWCO", "market": "nasdaq", "첫거래일": "2026-08-03", "상장후거래일": 30,
                       "상장후수익률": 0.3})
    yinfo = wu.item_info(young, tv="NYSE:NEWCO", insider=None, isb=None, sqz=None)
    assert yinfo["first_date"] == "2026-08-03" and yinfo["listed_days"] == 30 and yinfo["score"] is None
    p = wu.market_payload("sp500", "S&P 500", {"AAPL": info, "SPY": info}, bar_date="2026-10-17",
                          counts={"listed": 2})
    assert list(p["items"]) == ["AAPL"]
    sizes = wu.write_payloads(tmp_path, [p, wu.market_payload("new", "신규 상장", {"NEWCO": yinfo}, bar_date=None,
                                                               counts={"listed": 1})])
    assert sizes["stocks/"] == 2
    one = json.loads((tmp_path / "stocks" / "AAPL.json").read_text())
    assert one["market"] == "sp500" and one["info"]["company"] == "Apple Inc."
    json.loads((tmp_path / "new.json").read_text())
    import gzip
    assert json.loads(gzip.decompress((tmp_path / "new.json.gz").read_bytes())) == json.loads((tmp_path / "new.json").read_text())


def test_snapshot_frame_keeps_fallback_columns_without_nan_text() -> None:
    f = pd.DataFrame({"티커": ["A", "B"], "섹터": ["Tech", None], "fund_roe": [0.1, np.nan],
                      "next_earnings_date": ["2026-11-01", None], "RSI": [50, 60]})
    snap = rus.snapshot_frame([f])
    assert list(snap.columns) == ["티커", "섹터", "fund_roe", "next_earnings_date"]
    assert pd.isna(snap.loc[1, "섹터"])  # 'nan' 글자가 섹터로 들어가면 안 된다
    assert snap.loc[0, "섹터"] == "Tech"


class FakeWorksheet:
    def __init__(self, rows: int = 1000, cols: int = 50) -> None:
        self.row_count, self.col_count = rows, cols
        self.calls: list[tuple[str, int]] = []

    def resize(self, rows: int, cols: int) -> None:
        self.row_count, self.col_count = rows, cols
        self.calls.append(("resize", rows))

    def update(self, values=None, range_name=None, value_input_option=None) -> None:
        assert len(values) + (int(range_name[1:]) - 1 if range_name else 0) <= self.row_count, "격자 밖에 쓰기"
        self.calls.append(("update", len(values)))


def test_write_rows_grows_grid_then_chunks() -> None:
    ws = FakeWorksheet()
    rows = [["x"] * 3 for _ in range(4500)]
    ex._write_rows(ws, rows, chunk_rows=2000)
    assert ws.calls[0] == ("resize", 4500)
    assert [c for c in ws.calls if c[0] == "update"] == [("update", 2000), ("update", 2000), ("update", 500)]


def test_write_rows_small_tab_is_one_update() -> None:
    ws = FakeWorksheet()
    ex._write_rows(ws, [["x"] * 3 for _ in range(231)])
    assert ws.calls == [("update", 231)]


def test_sheet_pattern_cells_get_running_bar_preview() -> None:
    """시트 주·월봉 패턴 칸 끝에 예고 — 원본(점수용) 칸은 그대로."""
    ranked = pd.DataFrame([
        {"티커": "AAA", "주봉패턴": "골든크로스임박, 정배열(완전)", "월봉패턴": "", "주봉_예고": "골든크로스", "월봉_예고": "데드크로스임박"},
        {"티커": "BBB", "주봉패턴": "도지", "월봉패턴": np.nan, "주봉_예고": "", "월봉_예고": np.nan},
    ])
    out = ex.prepare_export_dataframe(ranked)
    assert out["주봉패턴"].tolist() == ["골든크로스임박, 정배열(완전) · 예고(이번 주) 골든크로스", "도지"]
    assert out.loc[0, "월봉패턴"] == "예고(이번 달) 데드크로스임박" and pd.isna(out.loc[1, "월봉패턴"])
    assert "주봉_예고" not in out.columns and ranked.loc[0, "주봉패턴"] == "골든크로스임박, 정배열(완전)"


def test_sheet_dividend_cells_go_last() -> None:
    """배당 칸은 맨 끝 (기존 칸 위치 그대로) — 수익률은 %, 금액은 1주당 달러."""
    ranked = pd.DataFrame([
        {"티커": "VZ", "현재가격": 43.71, "RSI": 55.0, "dividend_yield": 0.064379, "annual_dividend": 2.814},
        {"티커": "NVDA", "현재가격": 190.0, "RSI": 61.0, "dividend_yield": 0.0002, "annual_dividend": 0.04},
    ])
    out = ex.prepare_export_dataframe(ranked)
    assert list(out.columns[-2:]) == ["배당수익률(%)", "연 배당금($/주)"]
    assert out["배당수익률(%)"].tolist() == [6.4, 0.0] and out["연 배당금($/주)"].tolist() == [2.814, 0.04]
    assert out.columns.tolist().index("RSI") < out.columns.tolist().index("배당수익률(%)")
