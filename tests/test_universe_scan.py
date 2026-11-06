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


def test_ret_k_and_item_info_long_returns() -> None:
    close = pd.Series(np.linspace(100, 200, 300))
    assert abs(rus._ret_k(close, 252) - (200 / close.iloc[-253] - 1)) < 1e-12
    assert rus._ret_k(close.iloc[:252], 252) is None  # 일봉 252개 = 1년 전 종가 없음
    assert rus._ret_k(pd.Series([0.0, 5.0]), 1) is None  # 0 원은 수익률을 못 낸다
    info = wu.item_info(pd.Series({"티커": "AAPL", "63일수익률": 0.123456, "252일수익률": -0.2}),
                        tv=None, insider=None, isb=None, sqz=None)
    assert info["ret_63d"] == 0.1235 and info["ret_252d"] == -0.2
    assert wu.item_info(pd.Series({"티커": "NEWCO"}), tv=None, insider=None, isb=None, sqz=None)["ret_252d"] is None


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


def test_company_about_goes_to_stock_files_only(tmp_path: Path) -> None:
    fund = pd.DataFrame({"티커": ["BSX", "NEWCO", "OLD"],
                         "fund_summary": ["  Boston Scientific develops medical devices. ", None, "nan"]})
    about = rus.company_about(fund)
    assert about == {"BSX": "Boston Scientific develops medical devices."}
    info = wu.item_info(pd.Series({"티커": "BSX", "market": "sp500", "fund_industry": "Medical Devices"}),
                        tv=None, insider=None, isb=None, sqz=None)
    assert info["industry"] == "Medical Devices"
    p = wu.market_payload("sp500", "S&P 500", {"BSX": info, "OLD": info}, bar_date=None, counts={})
    wu.write_payloads(tmp_path, [p], about)
    assert json.loads((tmp_path / "stocks" / "BSX.json").read_text())["about"] == about["BSX"]
    assert json.loads((tmp_path / "stocks" / "OLD.json").read_text())["about"] is None
    assert "Boston Scientific develops" not in (tmp_path / "sp500.json").read_text()


def test_congress_summary_on_rows_and_trades_in_stock_files(tmp_path: Path) -> None:
    """국회의원 거래 (2026-11-06) — congress_tickers.json 이 있으면 목록에 90일 요약, 종목 파일에 12개월 거래."""
    info = wu.item_info(pd.Series({"티커": "BRK-B", "market": "sp500"}), tv=None, insider=None, isb=None, sqz=None)
    other = wu.item_info(pd.Series({"티커": "AAPL", "market": "sp500"}), tv=None, insider=None, isb=None, sqz=None)
    p = wu.market_payload("sp500", "S&P 500", {"BRK-B": info, "AAPL": other}, bar_date=None, counts={})
    assert rus.load_congress(tmp_path) == ({}, {})                                  # 파일이 없으면 칩 없이
    summ = {"BRK-B": {"buys": 0, "sells": 1, "members": 1, "last": "2026-10-19", "leader": False, "big_buy": False}}
    trades = {"BRK-B": [{"filed": "2026-10-19", "type": "sell", "member": "Pete Sessions"}]}
    (tmp_path / "congress_tickers.json").write_text(json.dumps({"summary": summ, "tickers": trades}))
    s, t = rus.load_congress(tmp_path)
    assert rus.attach_congress([p], s) == 1 and p["items"]["BRK-B"]["congress"]["sells"] == 1
    assert "congress" not in p["items"]["AAPL"]
    wu.write_payloads(tmp_path / "out", [p], congress=t)
    assert json.loads((tmp_path / "out" / "stocks" / "BRK-B.json").read_text())["congress"][0]["member"] == "Pete Sessions"
    assert "congress" not in json.loads((tmp_path / "out" / "stocks" / "AAPL.json").read_text())
    (tmp_path / "congress_tickers.json").write_text("{broken")
    assert rus.load_congress(tmp_path) == ({}, {})


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
    """배당 칸은 맨 끝 (기존 칸 위치 그대로) — 지금 받는 배당: 수익률은 %, 금액은 1주당 1년 달러."""
    ranked = pd.DataFrame([
        {"티커": "VZ", "현재가격": 43.18, "RSI": 55.0, "dividend_rate": 2.832, "fund_dividend_rate": 2.83},
        {"티커": "NVDA", "현재가격": 200.0, "RSI": 61.0, "dividend_rate": 1.0, "fund_dividend_rate": 1.0},
        {"티커": "AIV", "현재가격": 1.61, "RSI": 40.0, "dividend_rate": 2.75, "fund_dividend_rate": 0.0},
    ])
    out = ex.prepare_export_dataframe(ranked)
    assert list(out.columns[-2:]) == ["배당수익률(%)", "연 배당금($/주)"]
    assert out["배당수익률(%)"].tolist() == [6.56, 0.5, 0.0] and out["연 배당금($/주)"].tolist() == [2.832, 1.0, 0.0]
    assert out.columns.tolist().index("RSI") < out.columns.tolist().index("배당수익률(%)")


# ── 번외 (2026-11-05) — 두 시장 밖 종목을 따로 분석해 extra.json 에만 ─────────────


def test_yf_symbol_keeps_canadian_suffix() -> None:
    from web_export.isb_signals import yf_symbol

    assert yf_symbol("abc.v") == "ABC.V" and yf_symbol("XY.TO") == "XY.TO" and yf_symbol("Q.CN") == "Q.CN"
    assert yf_symbol("BRK.B") == "BRK-B" and yf_symbol("BRK-B") == "BRK-B"  # 종류주는 그대로 하이픈


def test_extra_tv_symbol_and_foreign(monkeypatch) -> None:
    monkeypatch.setattr(rus, "TRADINGVIEW_TICKER_MAP", {"ABC.V": "ABCD"})
    assert rus.extra_tv("ABC.V", {}) == "TSXV:ABCD"               # 이름이 다르면 매핑 이름 + 거래소
    assert rus.extra_tv("XYZ.V", {}) == "TSXV:XYZ"
    assert rus.extra_tv("XY-B.TO", {}) == "TSX:XY.B" and rus.extra_tv("Q.CN", {}) == "CSE:Q"
    assert rus.extra_tv("AAPL", {"AAPL": "NASDAQ:AAPL"}) == "NASDAQ:AAPL"   # 미국은 두 시장과 같은 규칙
    assert rus.is_foreign("abc.v") and rus.is_foreign("XY.NE") and not rus.is_foreign("AAPL")
    assert not rus.is_foreign("BRK-B")


def test_penny_prices_keep_four_digits() -> None:
    from web_export.market_summary import price_round

    assert price_round(0.11499) == 0.115 and price_round(0.123456) == 0.1235 and price_round(12.3456) == 12.35
    info = wu.item_info(pd.Series({"티커": "ABC.V", "현재가격": 0.11500000208, "52주최저": 0.1149, "52주최고": 0.82}),
                        tv=None, insider=None, isb=None, sqz=None)
    assert info["price"] == 0.115 and info["range_52w"] == {"low": 0.1149, "high": 0.82}
    big = wu.item_info(pd.Series({"티커": "AAPL", "현재가격": 250.4567, "52주최저": 160.0, "52주최고": 260.123}),
                       tv=None, insider=None, isb=None, sqz=None)
    assert big["price"] == 250.46 and big["range_52w"] == {"low": 160.0, "high": 260.12}   # 1달러 이상은 예전처럼


def _extra_ranked(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"티커": tickers, "회사": [f"{t} Corp" for t in tickers], "현재가격": [0.115] * len(tickers),
                         "섹터": ["Basic Materials"] * len(tickers), "매수적합도_표시": ["★★★☆☆ (3.5)"] * len(tickers),
                         "일봉패턴": [""] * len(tickers), "주봉패턴": ["골든크로스임박"] * len(tickers),
                         "주봉_MA갭": [-0.01] * len(tickers), "fund_short_pct_float": [0.0004] * len(tickers)})


def test_extra_market_payload(monkeypatch, tmp_path: Path) -> None:
    import screener.sec_filings as sec
    import web_export.dashboard as dash

    def fake_analyze(tickers, label, names, strong, prev_dir):
        assert label == "[번외]"
        return {"ranked": _extra_ranked(tickers), "young": {}, "young_fund": pd.DataFrame(), "bar_date": "2026-11-05",
                "no_data": [], "fund": pd.DataFrame({"티커": tickers, "fund_summary": ["Makes batteries."] * len(tickers)})}

    sec_calls: list[list[str]] = []
    monkeypatch.setattr(rus, "analyze_market", fake_analyze)
    monkeypatch.setattr(sec, "fetch_flags", lambda tickers, **kw: sec_calls.append(list(tickers)) or {})
    monkeypatch.setattr(dash, "daily_prices", lambda tickers: {})
    rec = {"buys": 1, "sells": 0, "net_value": 1000.0, "last_date": "2026-10-01", "signal": "buy"}
    none = {"buys": 0, "sells": 0, "net_value": 0.0, "last_date": None, "signal": None}
    class FakeShort:  # FINRA 잔고 — 미국 종목만 있다
        settle = "2026-10-15"

        def get(self, t):
            return {"si": 1e6, "dtc": 3.0, "si_change": 0.1} if t == "XYZ" else None

    payload, about = rus.extra_market(["ABC.V", "XYZ"], {}, {"XYZ": "NYSE:XYZ"}, None, tmp_path, FakeShort(),
                                      {"XYZ": rec}, none)

    assert sec_calls == [["XYZ"]]                                   # 캐나다 상장은 SEC 를 안 찾는다
    assert payload["key"] == "extra" and payload["label"] == "번외" and payload["bar_date"] == "2026-11-05"
    assert payload["counts"] == {"listed": 2, "analyzed": 2, "young": 0, "no_data": 0}
    a, x = payload["items"]["ABC.V"], payload["items"]["XYZ"]
    assert a["market"] == "extra" and a["tv"] == "TSXV:ABC" and a["price"] == 0.115 and a["score"] == 3.5
    assert a["insider"] is None and x["insider"] == rec              # 정보 없음 vs 미국 종목은 시장 전체 조회 결과
    # 캐나다 상장은 Yahoo 잔고 (FINRA 날짜를 붙이지 않는다 — 기준일 없음), 미국은 FINRA
    assert a["short"] == {"pct": 0.0004, "dtc": None, "change": None, "settle": None, "high": False}
    assert x["short"]["dtc"] == 3.0 and x["short"]["settle"] == "2026-10-15"
    assert a["golden"]["timeframes"] == [{"tf": "weekly", "gap": -0.01}] and [g["ticker"] for g in payload["golden"]] == ["ABC.V", "XYZ"]
    assert about == {"ABC.V": "Makes batteries.", "XYZ": "Makes batteries."}
    sizes = wu.write_payloads(tmp_path / "out", [payload], about)
    one = json.loads((tmp_path / "out" / "stocks" / "ABC.V.json").read_text())
    assert one["market"] == "extra" and one["about"] == "Makes batteries." and sizes["stocks/"] == 2
    assert (tmp_path / "out" / "extra.json.gz").exists()


def test_extra_market_without_data_returns_none(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rus, "analyze_market", lambda *a: {"ranked": pd.DataFrame(), "young": {}, "young_fund": pd.DataFrame(),
                                                           "bar_date": "", "no_data": ["ABC.V"], "fund": pd.DataFrame()})
    assert rus.extra_market(["ABC.V"], {}, {}, None, tmp_path, None, {}, None) == (None, {})


def _fake_listings() -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["AAPL"], "name": ["Apple Inc."], "tv": ["NASDAQ:AAPL"]})


def _patch_main_inputs(monkeypatch) -> None:
    import screener.insider as insider_mod
    import screener.short_squeeze as sq

    monkeypatch.setattr(rus, "fetch_listings", _fake_listings)
    monkeypatch.setattr(rus, "strong_sectors", lambda: None)
    monkeypatch.setattr(insider_mod, "fetch_insider_market", lambda: pd.DataFrame())
    monkeypatch.setattr(sq, "short_data", lambda: None)


def test_main_extra_only_writes_extra_file(monkeypatch, tmp_path: Path) -> None:
    from web_export.universe import item_info, market_payload

    _patch_main_inputs(monkeypatch)
    seen: list[list[str]] = []

    def fake_extra(tickers, *a):
        seen.append(list(tickers))
        info = item_info(pd.Series({"티커": tickers[0], "market": "extra", "현재가격": 0.115}), tv="TSXV:ABC",
                         insider=None, isb=None, sqz=None)
        return market_payload("extra", "번외", {tickers[0]: info}, bar_date="2026-11-05", counts={"listed": 1}), {}

    monkeypatch.setattr(rus, "extra_market", fake_extra)
    monkeypatch.setattr(rus, "EXTRA_TICKERS", ("ABC.V", "AAPL"))     # 이미 두 시장에 있는 AAPL 은 건너뛴다
    out = tmp_path / "u"
    assert rus.main(["--out-dir", str(out), "--no-sheet", "--market", "extra"]) == 0
    assert seen == [["ABC.V"]]
    assert json.loads((out / "extra.json").read_text())["items"]["ABC.V"]["price"] == 0.115
    assert not (out / "market.json").exists() and not (out / "sp500.json").exists()   # 시장 숫자 · 두 시장은 안 쓴다


def test_main_survives_extra_failure(monkeypatch, tmp_path: Path) -> None:
    _patch_main_inputs(monkeypatch)

    def boom(*a):
        raise RuntimeError("yahoo down")

    monkeypatch.setattr(rus, "extra_market", boom)
    out = tmp_path / "u"
    assert rus.main(["--out-dir", str(out), "--no-sheet", "--market", "extra"]) == 0   # 번외가 실패해도 계속
    assert not (out / "extra.json").exists()


# ── 캐나다 (2026-11-05) — TSX · TSXV 상장 보통주 전부를 따로 분석해 ca.json 에 ─────────────


def _directory(*companies: tuple[str, list[str]]) -> dict:
    return {"results": [{"symbol": syms[0], "name": name, "instruments": [{"symbol": s, "name": name} for s in syms]}
                        for name, syms in companies]}


def test_ca_listing_filters_and_symbols() -> None:
    from data.ca_listings import fetch_ca_listings, is_fund_name, keep_symbol, yahoo_ticker

    assert is_fund_name("iShares S&P/TSX 60 Index ETF") and is_fund_name("Capital Group Equity Select ETFTM (Canada)")
    assert is_fund_name("Apple (CAD Hedged) CDR") and is_fund_name("Dividend 15 Split Corp.")
    assert not is_fund_name("Getfit Brands Inc.") and not is_fund_name("RioCan Real Estate Investment Trust")
    for fund in ("Sprott Physical Gold Trust", "Royal Canadian Mint - Canadian Gold Reserves", "Manulife Canadian Equity Class",
                 "Canadian Banc Corp.", "Premium Income Corporation"):
        assert is_fund_name(fund), fund
    assert not is_fund_name("Decisive Dividend Corporation") and not is_fund_name("Exchange Income Corporation")
    assert keep_symbol("BBD.B", "tsx") and keep_symbol("REI.UN", "tsx") and keep_symbol("ABC", "tsxv")
    for bad in ("BCE.PR.A", "FFH.U", "XYZ.DB", "ABC.WT", "AEM.CV", "ZST.F"):
        assert not keep_symbol(bad, "tsx")
    assert not keep_symbol("AAAJ.P", "tsxv") and keep_symbol("AAAJ", "tsxv")      # 상장용 껍데기 회사(CPC)
    assert yahoo_ticker("BBD.B", ".TO") == "BBD-B.TO" and yahoo_ticker("rei.un", ".TO") == "REI-UN.TO"

    pages = {
        "tsx": _directory(("Bombardier Inc.", ["BBD.A", "BBD.B"]), ("BMO Equal Weight Banks Index ETF", ["ZEB"]),
                          ("BCE Inc.", ["BCE", "BCE.PR.A"]), ("Fairfax Financial Holdings", ["FFH", "FFH.U"]),
                          ("RioCan Real Estate Investment Trust", ["REI.UN"])),
        "tsxv": _directory(("Abc Minerals Ltd.", ["ABC"]), ("AAJ Capital 3 Corp.", ["AAAJ.P"])),
    }
    out = fetch_ca_listings(lambda url: pages["tsxv" if "/tsxv/" in url else "tsx"])
    assert out["ticker"].tolist() == ["BBD-A.TO", "BBD-B.TO", "BCE.TO", "FFH.TO", "REI-UN.TO", "ABC.V"]
    assert out["tv"].tolist() == ["TSX:BBD.A", "TSX:BBD.B", "TSX:BCE", "TSX:FFH", "TSX:REI.UN", "TSXV:ABC"]
    assert out["exchange"].tolist() == ["TSX"] * 5 + ["TSXV"] and out.loc[5, "name"] == "Abc Minerals Ltd."


def test_yahoo_short_uses_interest_float_and_volume() -> None:
    idx = pd.date_range("2026-10-01", periods=25, freq="B")
    prices = {"ABC.V": pd.DataFrame({"Volume": [1e5] * 5 + [2e5] * 20}, index=idx)}   # 최근 20일 평균 200,000주
    df = pd.DataFrame({"티커": ["ABC.V", "XY-B.TO", "Q.TO"],
                       "fund_short_interest": [1.2e6, np.nan, 5.0], "fund_float_shares": [4e6, 1e7, 0.0],
                       "fund_short_pct_float": [np.nan, 0.03, np.nan]})
    out = rus.add_yahoo_short(df, prices)
    assert out["short_pct"].tolist()[:2] == [0.3, 0.03] and pd.isna(out.loc[2, "short_pct"])  # 잔고 ÷ 유통주식, 없으면 Yahoo 비중
    assert out.loc[0, "short_dtc"] == 6.0 and pd.isna(out.loc[1, "short_dtc"])   # 1.2M ÷ 20만 주 = 6일분
    assert out.loc[0, "[공매도]"] == "공매도 30% · 되사기 6.0일분" and bool(out.loc[0, "short_high"])  # 기준일 없음
    assert out["short_settle"].isna().all() and out.loc[2, "[공매도]"] == ""


def _ca_listings() -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["BBD-B.TO", "ABC.V", "NEW.V"], "name": ["Bombardier Inc.", "Abc Minerals", "Newco"],
                         "tv": ["TSX:BBD.B", "TSXV:ABC", "TSXV:NEW"], "exchange": ["TSX", "TSXV", "TSXV"]})


def _fake_ca_analyze(tickers, label, names, strong, prev_dir):
    assert label == "[캐나다 전체]" and strong is None          # 미국 섹터 ETF 강한 섹터는 안 쓴다
    ranked = _extra_ranked(tickers[:2]).assign(
        fund_industry=["Aerospace & Defense", "Specialty Chemicals"], fund_market_cap=[2e10, 2e7],
        fund_pe=[25.0, None], fund_target_mean=[30.0, None], fund_analyst_n=[12, None], fund_short_interest=[1e6, None],
        fund_float_shares=[1e8, None], **{"1일수익률": [-0.021, 0.15]})
    close = pd.Series([1.0, 1.1, 1.2], index=pd.date_range("2026-11-03", periods=3, freq="B"))
    young = {"NEW.V": pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": [1e4] * 3})}
    stats = {"BBD-B.TO": {"ret_1d": -0.021, "above50": True, "above200": False, "new_high": False, "new_low": True},
             "ABC.V": {"ret_1d": 0.15, "above50": True, "above200": True, "new_high": True, "new_low": False}}
    return {"ranked": ranked, "young": young, "young_fund": pd.DataFrame({"티커": ["NEW.V"], "fund_sector": ["Technology"]}),
            "bar_date": "2026-11-05", "no_data": ["DEAD.V"], "stats": stats,
            "fund": pd.DataFrame({"티커": tickers[:2], "fund_summary": ["Makes jets.", "Mines copper."]})}


def _fake_index(symbols, bar_date):
    assert symbols[0] == "^GSPTSE" and bar_date == "2026-11-05"
    return "^GSPTSE", -0.0123


def test_canada_market_payload(monkeypatch, tmp_path: Path) -> None:
    import web_export.dashboard as dash

    monkeypatch.setattr(rus, "analyze_market", _fake_ca_analyze)
    monkeypatch.setattr(rus, "index_return", _fake_index)
    monkeypatch.setattr(dash, "daily_prices", lambda tickers: {})
    res = rus.canada_market(_ca_listings(), tmp_path)

    # 오늘 탭 캐나다 판 (2026-11-06) — 시장 폭은 TSX C$1B 이상(BBD-B.TO)만, ABC.V(C$2천만)는 목록에서도 빠진다
    mk = res["market"]
    assert mk["market"] == "ca" and mk["index_label"] == "S&P/TSX" and mk["breadth"]["spy"] == -0.0123
    assert (mk["breadth"]["n"], mk["breadth"]["dec"], mk["breadth"]["new_low"]) == (1, 1, 1)
    assert [g["t"] for g in mk["gainers"]] == ["BBD-B.TO"] and mk["min_cap"] == 1e9

    p = res["payload"]
    assert p["key"] == "ca" and p["label"] == "캐나다" and p["bar_date"] == "2026-11-05"
    assert p["counts"] == {"listed": 3, "analyzed": 2, "young": 1, "no_data": 1}
    assert list(p["items"]) == ["BBD-B.TO", "ABC.V", "NEW.V"]                 # 점수 순 + 신규 상장은 뒤에
    b, n, y = (p["items"][t] for t in ("BBD-B.TO", "ABC.V", "NEW.V"))
    assert b["market"] == "tsx" and n["market"] == "tsxv" and y["market"] == "tsxv"   # 웹 시장 버튼 TSX · TSXV
    assert b["tv"] == "TSX:BBD.B" and n["tv"] == "TSXV:ABC" and y["first_date"] == "2026-11-03"
    assert all(v["insider"] is None for v in p["items"].values())              # 내부자 칸은 비운다 (웹은 ceo.ca 링크)
    assert all(not v["sec_flags"] for v in p["items"].values())
    assert b["short"]["pct"] == 0.01 and b["short"]["settle"] is None          # Yahoo 잔고 1M ÷ 유통 100M
    assert [g["ticker"] for g in p["golden"]] == ["BBD-B.TO", "ABC.V"]
    assert res["about"] == {"BBD-B.TO": "Makes jets.", "ABC.V": "Mines copper."}
    assert res["details"]["BBD-B.TO"]["analyst"]["mean"] == 30.0 and res["details"]["BBD-B.TO"]["quarters"] is None
    assert "ABC.V" not in res["details"]                                       # 밸류에이션 · 목표주가 둘 다 없음
    cols = res["sheet"].columns.tolist()
    assert cols[cols.index("티커") + 1] == "거래소" and res["sheet"]["거래소"].tolist() == ["TSX", "TSXV"]
    assert not {"[공시]", "미국 매출 비중", "실적 움직임"} & set(cols) and "[공매도]" in cols
    assert set(res["snapshot"]["티커"]) == {"BBD-B.TO", "ABC.V", "NEW.V"}


def test_main_canada_writes_ca_files(monkeypatch, tmp_path: Path) -> None:
    import data.ca_listings as cal
    import web_export.dashboard as dash

    monkeypatch.setattr(cal, "fetch_ca_listings", _ca_listings)
    monkeypatch.setattr(rus, "fetch_listings", lambda: (_ for _ in ()).throw(AssertionError("미국 목록은 안 받는다")))
    monkeypatch.setattr(rus, "analyze_market", _fake_ca_analyze)
    monkeypatch.setattr(rus, "index_return", _fake_index)
    monkeypatch.setattr(dash, "daily_prices", lambda tickers: {})
    out = tmp_path / "c"
    assert rus.main(["--out-dir", str(out), "--no-sheet", "--market", "ca"]) == 0
    assert json.loads((out / "ca.json").read_text())["key"] == "ca" and (out / "ca.json.gz").exists()
    assert json.loads((out / "stocks" / "BBD-B.TO.json").read_text())["market"] == "ca"
    assert (out / "snapshot" / "ca_ranked.parquet").exists()
    assert json.loads((out / "market.json").read_text())["market"] == "ca"   # canada-data/market.json (오늘 탭 캐나다)
    assert not (out / "sp500.json").exists()


def test_main_canada_without_data_fails(monkeypatch, tmp_path: Path) -> None:
    import data.ca_listings as cal

    monkeypatch.setattr(cal, "fetch_ca_listings", _ca_listings)
    monkeypatch.setattr(rus, "analyze_market", lambda *a: {"ranked": pd.DataFrame(), "young": {}, "young_fund": pd.DataFrame(),
                                                           "bar_date": "", "no_data": ["BBD-B.TO"], "fund": pd.DataFrame()})
    out = tmp_path / "c"
    assert rus.main(["--out-dir", str(out), "--no-sheet", "--market", "ca"]) == 1   # 빈 브랜치를 올리지 않게 실패
    assert not (out / "ca.json").exists()


def test_previous_snapshot_reads_canada_file(tmp_path: Path) -> None:
    from screener.fundamentals_guard import fill_from_previous_snapshot

    pd.DataFrame({"티커": ["BBD-B.TO"], "섹터": ["Industrials"], "fund_pe": [21.0]}).to_parquet(tmp_path / "ca_ranked.parquet")
    assert rus._prev_snapshot_dir(tmp_path) == tmp_path
    fund = pd.DataFrame({"티커": ["BBD-B.TO"], "fund_sector": ["Unknown"], "fund_market_cap": [np.nan], "fund_pe": [np.nan]})
    got = fill_from_previous_snapshot(fund, "[캐나다]", snapshot_dir=tmp_path)
    assert got.loc[0, "fund_pe"] == 21.0 and got.loc[0, "fund_sector"] == "Industrials"
