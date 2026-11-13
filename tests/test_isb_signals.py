"""웹 정렬용 ISB 라벨 (web_export/isb_signals.py) — 봉 완성 판정·최근 라벨 고르기·가격 정리 (네트워크 없음)."""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web_export import isb_signals as S  # noqa: E402

ET = S.market_now().tzinfo


def _now(day: str, hh: int, mm: int = 0) -> datetime:
    y, m, d = map(int, day.split("-"))
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def test_week_complete_waits_for_friday_close() -> None:
    weeks = pd.DatetimeIndex(["2026-10-09", "2026-10-16", "2026-10-21"])  # 금 · 금 · 이번 주 수요일까지
    assert S.week_complete(weeks, _now("2026-10-21", 20)).tolist() == [True, True, False]
    # 금요일 장중엔 그 주가 안 끝났고, 16:30 뒤엔 끝남
    assert S.week_complete(weeks[:2], _now("2026-10-16", 12)).tolist() == [True, False]
    assert S.week_complete(weeks[:2], _now("2026-10-16", 16, 30)).tolist() == [True, True]


def test_drop_unfinished_day_only_during_session() -> None:
    df = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.DatetimeIndex(["2026-10-16", "2026-10-19"]))
    assert len(S.drop_unfinished_day(df, _now("2026-10-19", 11))) == 1
    assert len(S.drop_unfinished_day(df, _now("2026-10-19", 17))) == 2
    assert len(S.drop_unfinished_day(df, _now("2026-10-20", 2, 30))) == 2


def _eng(n: int) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame({"ev_buy": False, "sig_type": "", "btm": False, "ev_topwarn": False, "ev_top": False,
                         "ev_exit": False}, index=idx)


def test_label_flags_match_chart_labels() -> None:
    eng = _eng(4)
    eng.loc[eng.index[0], ["ev_buy", "sig_type"]] = [True, "REV"]
    eng.loc[eng.index[1], ["ev_buy", "sig_type"]] = [True, "BO"]   # BUY·BO 는 REV 가 아니다
    eng.loc[eng.index[2], "ev_top"] = True                          # 보유 중 TRIM·TOP — ◆ TOP 아님
    eng.loc[eng.index[3], ["ev_topwarn", "btm", "ev_exit"]] = True
    f = S.label_flags(eng)
    assert f["rev"].tolist() == [True, False, False, False]
    assert f["top"].tolist() == [False, False, False, True]
    assert f["bottom"].tolist() == [False, False, False, True]
    assert f["exit"].tolist() == [False, False, False, True]


def test_recent_labels_window_and_unfinished_bar() -> None:
    eng = _eng(10)
    eng.loc[eng.index[[1, 6]], "ev_exit"] = True
    eng.loc[eng.index[2], "btm"] = True
    eng.loc[eng.index[9], "ev_topwarn"] = True  # 끝나지 않은 마지막 봉 — 차트에 아직 안 찍힘
    complete = np.array([True] * 9 + [False])
    out = S.recent_labels(eng, complete, window=5)
    # 마지막 끝난 봉 = 8 → 창은 4~8: EXIT 는 가장 최근(6) 하나, BOTTOM(2) 은 창 밖, TOP(9) 은 미완성
    assert out == {"exit": {"date": eng.index[6].strftime("%Y-%m-%d"), "ago": 2}}
    assert S.recent_labels(eng, np.zeros(10, dtype=bool), 5) == {}


def test_frame_rounds_prices_and_drops_empty_rows() -> None:
    idx = pd.DatetimeIndex(["2026-10-15", "2026-10-16", "2026-10-16"], tz="America/New_York")
    cols = pd.MultiIndex.from_product([["BRK-B"], ["Open", "High", "Low", "Close", "Volume"]])
    raw = pd.DataFrame([[25.459999084, 25.5, 25.4, 25.459999084, 100],
                        [np.nan, np.nan, np.nan, np.nan, np.nan],
                        [25.5, 25.6, 25.45, 25.55000019, 200]], index=idx, columns=cols)
    df = S._frame(raw, "BRK-B")
    assert df is not None and df.index.tz is None and len(df) == 2
    assert df["Close"].tolist() == [25.46, 25.55]
    assert S._frame(raw, "AAPL") is None
    assert S.yf_symbol("brk.b") == "BRK-B"


def test_isb_block_never_breaks_export(monkeypatch) -> None:
    import web_export.dashboard as wd

    def boom(tickers: list[str]) -> dict:
        raise RuntimeError("yfinance down")

    monkeypatch.setattr(S, "isb_labels", boom)
    assert wd.isb_block(["AAPL"]) == {}
    assert wd.ticker_info(pd.Series({"티커": "AAPL"}))["isb"] is None  # build_dashboard 가 채우기 전 기본값


def test_bench_symbol_follows_listing_like_pine() -> None:
    # ISB.pine benchSym Auto: TSXV → TSX:JX · TSX → TSX:TSX · 그 밖(미국 · CSE · NEO) → SPY
    assert S.bench_symbol("AAPL") == "SPY"
    assert S.bench_symbol("brk.b") == "SPY"
    assert S.bench_symbol("RY.TO") == "^GSPTSE"
    assert S.bench_symbol("BIP-UN.TO") == "^GSPTSE"
    assert S.bench_symbol("nbm.v") == "^SPCDNX"
    assert S.bench_symbol("ABC.CN") == "SPY" and S.bench_symbol("XYZ.NE") == "SPY"
    assert S.is_junior("NBM.V") and not S.is_junior("RY.TO") and not S.is_junior("AAPL")


def test_junior_params_follow_pine_auto() -> None:
    P = S._engine().IsbParams
    us_d, us_w = P("D"), P("W")
    assert (us_d.use_swing, us_d.use_s2, us_d.use_mkt_filter) == (False, True, True)  # 일봉 Trend Hold + TOP
    assert (us_w.use_swing, us_w.use_s2, us_w.use_mkt_filter) == (True, False, False)
    jr_d, jr_w = P("D", junior=True), P("W", junior=True)
    assert (jr_d.use_swing, jr_d.use_s2, jr_d.use_mkt_filter) == (True, False, False)  # TSXV 는 일봉도 Swing
    assert (jr_w.use_swing, jr_w.use_s2, jr_w.use_mkt_filter) == (True, False, False)
    assert jr_d.reg_len == us_d.reg_len and jr_d.mkt_len == us_d.mkt_len  # 나머지 Auto 값은 주기 그대로


def _ohlcv(close: float, n: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range("2026-09-01", periods=n)
    c = np.full(n, close)  # 고가 = 저가인 봉은 clean_ohlcv 가 가짜 봉으로 지운다
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1000.0}, index=idx)


def test_isb_labels_uses_listing_benchmark_and_skips_missing(monkeypatch) -> None:
    E = S._engine()
    # 벤치마크마다 다른 종가 → build_market 이 어떤 벤치마크로 만들어졌는지 Market 대신 종가로 확인
    monkeypatch.setattr(E, "build_market", lambda bench, macro, p: float(bench["Close"].iloc[-1]))
    seen: dict[str, tuple] = {}

    def fake_labels(daily, markets, now, junior=False, monthly=None):
        seen[float(daily["Close"].iloc[-1])] = (markets["D"], markets["W"], junior)
        return {}

    monkeypatch.setattr(S, "ticker_labels", fake_labels)
    fetched: list[list[str]] = []
    have = {"^VIX": _ohlcv(20), "DX-Y.NYB": _ohlcv(100), "^GSPTSE": _ohlcv(36000)}

    def fake_download(symbols, period=S.HISTORY):
        fetched.append(sorted(symbols))
        return {s: have[s] for s in symbols if s in have}

    monkeypatch.setattr(S, "download_daily", fake_download)
    monkeypatch.setattr(S, "download_monthly", lambda symbols, period=S.MONTHLY_HISTORY: {})  # 월봉 못 받음 → 일·주만
    prices = {"SPY": _ohlcv(680), "AAPL": _ohlcv(1), "RY.TO": _ohlcv(2), "NBM.V": _ohlcv(3)}
    out = S.isb_labels(["AAPL", "RY.TO", "NBM.V"], now=_now("2026-11-12", 20), prices=prices)
    # 미국 = SPY · TSX = ^GSPTSE (넘겨준 prices 에 없어서 그것만 따로 받음) · TSXV 는 ^SPCDNX 를 못 받아 빠짐
    assert seen == {1.0: (680.0, 680.0, False), 2.0: (36000.0, 36000.0, False)}
    assert set(out) == {"AAPL", "RY.TO"}
    assert ["^GSPTSE", "^SPCDNX"] in fetched

    have["^SPCDNX"] = _ohlcv(800)
    seen.clear()
    out = S.isb_labels(["NBM.V"], now=_now("2026-11-12", 20), prices={"NBM.V": _ohlcv(3)})
    assert seen == {3.0: (800.0, 800.0, True)} and set(out) == {"NBM.V"}  # TSXV = 벤처 지수 + 주니어 규칙

    # VIX · DXY 가 없으면 전부 건너뜀 (예전과 같음)
    del have["^VIX"]
    assert S.isb_labels(["AAPL"], now=_now("2026-11-12", 20), prices=prices) == {}


def test_month_complete_and_last_trading_day() -> None:
    # 야후 1mo 날짜 = 그 달 1일. 2026-10-31 은 토요일 → 10월 마지막 평일 = 10/30 (금)
    months = pd.DatetimeIndex(["2026-09-01", "2026-10-01", "2026-11-01"])
    assert S.month_complete(months, _now("2026-10-30", 12)).tolist() == [True, False, False]  # 마지막 평일 장중
    assert S.month_complete(months, _now("2026-10-30", 16, 30)).tolist() == [True, True, False]  # 장 마감 뒤
    assert S.month_complete(months, _now("2026-11-12", 20)).tolist() == [True, True, False]  # 이번 달은 진행 중
    days = pd.DatetimeIndex(["2026-09-29", "2026-09-30", "2026-10-29", "2026-10-30", "2026-11-12"])
    out = S.month_last_days(pd.DatetimeIndex(["2026-08-01", "2026-09-01", "2026-10-01"]), days)
    # 8월은 일봉에 없음 → 그 달 마지막 평일 8/31 (월), 9 · 10월은 일봉의 마지막 거래일
    assert [d.strftime("%Y-%m-%d") for d in out] == ["2026-08-31", "2026-09-30", "2026-10-30"]


def test_monthly_params_follow_pine_auto() -> None:
    P = S._engine().IsbParams
    m, w, d = P("M"), P("W"), P("D")
    # Pine: autoLen/autoK 10/2 · swLen 2 · mktLen 3 · mktUpLen 10 · hlLen 12 · REV 시장 투매 필터 끔(월봉) · Swing
    assert (m.reg_len, m.reg_k, m.swing_len, m.mkt_len, m.mkt_up_len, m.hl_len) == (10, 2, 2, 3, 10, 12)
    assert (m.use_swing, m.use_s2, m.use_mkt_filter, m.need_htf, m.use_rev_mkt) == (True, False, False, False, False)
    assert (w.reg_len, w.reg_k, w.swing_len, w.mkt_len, w.mkt_up_len, w.hl_len, w.use_rev_mkt) == (30, 5, 3, 12, 40, 52, True)
    assert (d.reg_len, d.reg_k, d.swing_len, d.mkt_len, d.mkt_up_len, d.use_rev_mkt) == (150, 20, 3, 60, 200, True)


def _daily_until(end: str, n: int) -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=n)
    c = np.linspace(10, 20, n)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1000.0}, index=idx)


def test_ticker_labels_monthly_only_finished_months_in_window(monkeypatch) -> None:
    """월봉: 끝난 달만 · 창 12달 · 날짜는 그 달 마지막 거래일 (엔진은 가짜 — 라벨 위치와 날짜만 본다)."""
    E = S._engine()

    def fake_run(bars, inp, p, *a, **k):
        eng = pd.DataFrame({"ev_buy": False, "sig_type": "", "btm": False, "ev_topwarn": False, "ev_top": False,
                            "ev_exit": False}, index=bars.index)
        if p.monthly:
            eng.loc[eng.index[-1], "ev_exit"] = True      # 이번 달(진행 중) — 차트에 아직 안 찍힘
            eng.loc[eng.index[-2], "btm"] = True          # 지난달(10월) → 마지막 끝난 봉
            eng.loc[eng.index[-15], "ev_topwarn"] = True  # 13달 전 → 창(12) 밖
        return eng, []

    monkeypatch.setattr(E, "run_engine", fake_run)
    monkeypatch.setattr(E, "prepare_bars", lambda df, mk, p: (E.to_monthly(df) if p.monthly else df, {}))
    daily = _daily_until("2026-11-12", 300)
    midx = pd.date_range(end="2026-11-01", periods=24, freq="MS")
    mc = np.linspace(10, 20, len(midx))
    monthly = pd.DataFrame({"Open": mc, "High": mc * 1.05, "Low": mc * 0.95, "Close": mc, "Volume": 1e5}, index=midx)
    mk = {"D": None, "W": None, "M": None}
    out = S.ticker_labels(daily, mk, _now("2026-11-12", 20), monthly=monthly)
    assert out == {"M": {"bottom": {"date": "2026-10-30", "ago": 0}}}
    # 월봉 시장 자료가 없으면(markets 에 M 없음) 월봉만 건너뜀
    assert S.ticker_labels(daily, {"D": None, "W": None}, _now("2026-11-12", 20), monthly=monthly) == {}
