"""account_engine 체결·청산 메커니즘 테스트 (가짜 전략 모듈 사용, pandas 불필요)."""

from types import SimpleNamespace

from paper_trading.account_engine import (
    AccountState,
    BarSeries,
    catchup_dates,
    load_account,
    process_bar,
    regime_from_rows,
    save_account,
)
from paper_trading.accounts import AccountProfile

D1, D2, D3 = "2026-01-05", "2026-01-06", "2026-01-07"

BASE_PARAMS = {
    "version": "test-v0", "data_dir": "unused", "initial_capital": 10000.0,
    "cost_per_side": 0.0, "max_positions": 2, "max_daily_buys": 2,
    "max_same_sector": 2, "max_gap_up": 0.03, "min_alloc_ratio": 0.25,
}


def _bar(d: str, o: float, h: float, lo: float, c: float) -> dict:
    return {"date": d, "open": o, "high": h, "low": lo, "close": c, "volume": 1e6}


def _cand(ticker: str, close: float = 100.0, atr: float = 2.0, sector: str = "Tech") -> dict:
    return {"ticker": ticker, "score": 0.9, "strategy": "테스트", "sector": sector, "close": close, "atr": atr}


def _strategy(candidates=(), stop=None, target=None, exit_on=None):
    """exit_on: {날짜: 사유} — 그날 장 마감 후 청산 신호."""

    def select_candidates(rows, ctx, params):
        return [c for c in candidates if c["ticker"] not in ctx["exclude"]], {}

    def init_position(pos, order, params):
        if stop is not None:
            pos["stop_price"] = stop
        if target is not None:
            pos["target_price"] = target

    def on_bar_close(pos, ind, bar_date, params):
        return (exit_on or {}).get(bar_date)

    return SimpleNamespace(
        prefilter_tickers=lambda rows, ctx, params: [],
        select_candidates=select_candidates,
        init_position=init_position,
        on_bar_close=on_bar_close,
    )


def _profile(strategy, **overrides) -> AccountProfile:
    return AccountProfile("ptx", "테스트", {**BASE_PARAMS, **overrides}, {}, True, strategy)


def test_buy_fills_at_next_open_with_cost():
    series = {"AAA": BarSeries([_bar(D1, 99, 101, 98, 100), _bar(D2, 101, 103, 100, 102)])}
    prof = _profile(_strategy([_cand("AAA")]), cost_per_side=0.001)
    st = AccountState.new(10000.0)

    r1 = process_bar(st, prof, D1, series, rows=[])
    assert [o["ticker"] for o in r1["orders"]] == ["AAA"]
    assert not st.positions

    process_bar(st, prof, D2, series)
    pos = st.positions[0]
    assert pos["entry_price"] == 101 and pos["entry_date"] == D2
    assert abs(pos["shares"] * 101 * 1.001 - 5000) < 0.01  # 목표 배분 = 평가액 / 최대 포지션
    assert abs(st.cash - 5000) < 0.01
    assert st.equity_history[-1]["equity"] == round(st.cash + pos["shares"] * 102, 2)


def test_gap_up_cancels_buy():
    series = {"AAA": BarSeries([_bar(D1, 99, 101, 98, 100), _bar(D2, 104, 105, 103, 104)])}
    prof = _profile(_strategy([_cand("AAA")]), max_gap_up=0.03)
    st = AccountState.new(10000.0)
    process_bar(st, prof, D1, series, rows=[])
    r = process_bar(st, prof, D2, series)
    assert not st.positions
    assert r["cancelled"][0]["cancel_reason"].startswith("갭상승")


def _enter_then(bar3: dict, stop=95.0, target=105.0, bar2=None):
    bar2 = bar2 or _bar(D2, 100, 101, 99.5, 100.5)
    series = {"AAA": BarSeries([_bar(D1, 99, 101, 98, 100), bar2, bar3])}
    prof = _profile(_strategy([_cand("AAA")], stop=stop, target=target))
    st = AccountState.new(10000.0)
    process_bar(st, prof, D1, series, rows=[])
    process_bar(st, prof, D2, series)
    r = process_bar(st, prof, D3, series)
    return st, r


def test_same_bar_stop_and_target_takes_stop():
    st, r = _enter_then(_bar(D3, 100, 110, 90, 100))
    trade = r["sells"][0]
    assert trade["exit_price"] == 95.0
    assert trade["exit_reason"].startswith("손절")
    assert not st.positions


def test_gap_through_stop_fills_at_open():
    _, r = _enter_then(_bar(D3, 93, 94, 92, 93.5))
    trade = r["sells"][0]
    assert trade["exit_price"] == 93
    assert trade["exit_reason"].startswith("갭손절")


def test_target_hit_intraday():
    _, r = _enter_then(_bar(D3, 101, 106, 100, 104))
    assert r["sells"][0]["exit_price"] == 105.0
    assert r["sells"][0]["exit_reason"].startswith("목표가")


def test_stop_checked_on_entry_day():
    series = {"AAA": BarSeries([_bar(D1, 99, 101, 98, 100), _bar(D2, 100, 101, 94, 96)])}
    prof = _profile(_strategy([_cand("AAA")], stop=95.0))
    st = AccountState.new(10000.0)
    process_bar(st, prof, D1, series, rows=[])
    r = process_bar(st, prof, D2, series)
    assert r["sells"][0]["exit_price"] == 95.0
    assert r["sells"][0]["exit_reason"].startswith("손절")


def test_close_signal_sells_next_open():
    series = {"AAA": BarSeries([_bar(D1, 99, 101, 98, 100), _bar(D2, 100, 101, 99, 100), _bar(D3, 102, 103, 101, 102)])}
    prof = _profile(_strategy([_cand("AAA")], exit_on={D2: "테스트청산"}))
    st = AccountState.new(10000.0)
    process_bar(st, prof, D1, series, rows=[])
    r2 = process_bar(st, prof, D2, series)
    assert r2["orders"][0]["side"] == "SELL"
    r3 = process_bar(st, prof, D3, series)
    trade = r3["sells"][0]
    assert trade["exit_price"] == 102 and trade["exit_reason"].startswith("테스트청산")
    assert trade["holding_bars"] == 1


def test_pending_sell_frees_slot_for_new_buy():
    bars = [_bar(D1, 99, 101, 98, 100), _bar(D2, 100, 101, 99, 100), _bar(D3, 100, 101, 99, 100)]
    series = {"XXX": BarSeries(bars), "YYY": BarSeries(bars)}
    prof = _profile(
        _strategy([_cand("XXX"), _cand("YYY", sector="Health")], exit_on={D2: "청산"}),
        max_positions=1,
    )
    st = AccountState.new(10000.0)
    process_bar(st, prof, D1, series, rows=[])
    r2 = process_bar(st, prof, D2, series, rows=[])
    assert [(o["side"], o["ticker"]) for o in r2["orders"]] == [("SELL", "XXX"), ("BUY", "YYY")]
    r3 = process_bar(st, prof, D3, series)
    assert [t["ticker"] for t in r3["sells"]] == ["XXX"]
    assert [p["ticker"] for p in st.positions] == ["YYY"]


def test_sector_limit_and_daily_buy_limit():
    series = {t: BarSeries([_bar(D1, 99, 101, 98, 100)]) for t in ("A", "B", "C", "D")}
    cands = [_cand("A"), _cand("B"), _cand("C", sector="Health"), _cand("D", sector="Energy")]
    prof = _profile(_strategy(cands), max_positions=5, max_daily_buys=2, max_same_sector=1)
    st = AccountState.new(10000.0)
    r = process_bar(st, prof, D1, series, rows=[])
    assert [o["ticker"] for o in r["orders"]] == ["A", "C"]


def test_catchup_dates():
    cal = BarSeries([_bar(d, 1, 1, 1, 1) for d in (D1, D2, D3, "2026-01-08")])
    assert catchup_dates(cal, D1, "2026-01-08") == [D2, D3]
    assert catchup_dates(cal, None, "2026-01-08") == []
    assert catchup_dates(cal, D3, "2026-01-08") == []


def test_bar_series_never_returns_future_bars():
    s = BarSeries([_bar(d, 1, 1, 1, 1) for d in (D1, D2, D3)])
    assert [b["date"] for b in s.upto(D2)] == [D1, D2]
    assert [b["date"] for b in s.upto(D3, n=2)] == [D2, D3]
    assert s.count_after(D1, D3) == 2


def test_regime_from_rows():
    def spy(close, e50, e200):
        return [{"티커": "SPY", "close": close, "ema50": e50, "ema200": e200}]

    assert regime_from_rows(spy(110, 105, 100)) == "bull"
    assert regime_from_rows(spy(95, 105, 100)) == "bear"
    assert regime_from_rows(spy(102, 104, 100)) == "neutral"
    assert regime_from_rows([]) == "neutral"


def test_save_and_load_roundtrip(tmp_path):
    series = {"AAA": BarSeries([_bar(D1, 99, 101, 98, 100), _bar(D2, 100, 101, 99, 100)])}
    prof = _profile(_strategy([_cand("AAA")]), data_dir=str(tmp_path))
    st = AccountState.new(10000.0)
    process_bar(st, prof, D1, series, rows=[])
    save_account(prof, st)
    loaded = load_account(prof)
    assert loaded.pending == st.pending
    assert loaded.cash == st.cash
    assert loaded.equity_history == st.equity_history
