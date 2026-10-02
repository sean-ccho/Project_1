"""PT-3 일봉 단타 규칙 테스트."""

from datetime import date, timedelta

from paper_trading import pt3_short_term as pt3
from paper_trading.account_engine import BarSeries
from screener.config import PT3_PARAMS


def _params(**overrides) -> dict:
    return {**PT3_PARAMS, **overrides}


def _row(ticker: str, **kw) -> dict:
    base = {
        "티커": ticker, "섹터": "Industrials", "close": 100.0, "현재가격": 100.0,
        "ema50": 95.0, "ema200": 90.0, "atr_value": 2.0, "최근20일평균거래대금": 50e6,
        "RSI": 60.0, "bollinger_pband": 0.6, "5일수익률": 0.02, "변동성압축": 1.2,
        "거래량돌파배수": 1.0, "adx": 15.0, "ema_gap_50_200": 0.05, "20일수익률": 0.03,
        "52주포지션": 0.7, "일봉패턴": "", "저점반전캔들": 0, "매수적합도_표시": "★★★",
    }
    base.update(kw)
    return base


def _series(ticker: str, highs: list[float], closes: list[float]) -> tuple[dict, str]:
    start = date(2026, 1, 1)
    bars = [
        {"date": str(start + timedelta(days=i)), "open": c, "high": h, "low": min(c, h) - 1, "close": c, "volume": 1e6}
        for i, (h, c) in enumerate(zip(highs, closes))
    ]
    return {ticker: BarSeries(bars)}, bars[-1]["date"]


def _ctx(series: dict, bar_date: str, regime: str = "bull") -> dict:
    return {"regime": regime, "exclude": {"SPY", "QQQ", "IWM"}, "bar_date": bar_date, "series": series}


PULLBACK = dict(RSI=45.0, bollinger_pband=0.25, **{"5일수익률": -0.03})
BREAKOUT = dict(RSI=65.0, ema50=90.0, adx=25.0, **{"변동성압축": 0.8, "거래량돌파배수": 2.0, "5일수익률": 0.05})


def test_pullback_selected_when_close_breaks_previous_high():
    series, d = _series("AAA", [99.0] * 21, [98.0] * 20 + [100.0])
    cands, _ = pt3.select_candidates([_row("AAA", **PULLBACK)], _ctx(series, d), _params())
    assert [(c["ticker"], c["setup"], c["strategy"]) for c in cands] == [("AAA", "pullback", "단타-눌림목")]


def test_pullback_without_reversal_rejected():
    series, d = _series("AAA", [99.0] * 21, [98.0] * 20 + [98.5])
    cands, debug = pt3.select_candidates([_row("AAA", **PULLBACK)], _ctx(series, d), _params())
    assert cands == [] and debug["탈락"] == {"반전 미확인": 1}


def test_pullback_confirmed_by_daily_pattern():
    series, d = _series("AAA", [99.0] * 21, [98.0] * 20 + [98.5])
    row = _row("AAA", 일봉패턴="강세잉걸핑", **PULLBACK)
    cands, _ = pt3.select_candidates([row], _ctx(series, d), _params())
    assert cands[0]["setup"] == "pullback"


def test_breakout_records_breakout_level():
    series, d = _series("BBB", [98.0] * 20 + [101.0], [97.0] * 20 + [100.0])
    cands, _ = pt3.select_candidates([_row("BBB", **BREAKOUT)], _ctx(series, d), _params())
    assert cands[0]["setup"] == "breakout"
    assert cands[0]["meta"] == {"breakout_level": 98.0}
    assert cands[0]["close"] == 100.0


def test_breakout_blocked_in_bear_market():
    series, d = _series("BBB", [98.0] * 20 + [101.0], [97.0] * 20 + [100.0])
    cands, debug = pt3.select_candidates([_row("BBB", **BREAKOUT)], _ctx(series, d, "bear"), _params())
    assert cands == [] and debug["돌파셋업"] == "약세장 중단"


def test_prefilter_returns_only_setup_like_rows():
    rows = [_row("AAA", **PULLBACK), _row("BBB", **BREAKOUT), _row("CCC"), _row("SPY", **PULLBACK)]
    ctx = {"exclude": {"SPY", "QQQ", "IWM"}}
    assert pt3.prefilter_tickers(rows, ctx, _params()) == ["AAA", "BBB"]


def _pos(**kw) -> dict:
    pos = {"entry_price": 100.0, "atr_at_signal": 2.0, "stop_price": 97.0, "stop_kind": "손절",
           "highest_price": 101.0, "setup": "pullback", "bars_held": 1}
    pos.update(kw)
    return pos


def test_init_position_sets_stop_and_target():
    pos = {"entry_price": 100.0}
    pt3.init_position(pos, {"atr": 2.0}, _params())
    assert pos["stop_price"] == 97.0 and pos["target_price"] == 104.0


def test_breakeven_after_one_atr():
    pos = _pos(highest_price=102.5)
    assert pt3.on_bar_close(pos, {"rsi": 50.0, "bb_pband": 0.5, "close": 101.0}, "d", _params()) is None
    assert pos["stop_price"] == 100.0 and pos["stop_kind"] == "본절"


def test_pullback_setup_complete_exit():
    reason = pt3.on_bar_close(_pos(), {"rsi": 61.0, "bb_pband": 0.5, "close": 101.0}, "d", _params())
    assert reason == "셋업완료(눌림 회복)"


def test_breakout_failure_exit():
    pos = _pos(setup="breakout", breakout_level=98.0)
    assert pt3.on_bar_close(pos, {"close": 96.9}, "d", _params()) == "돌파실패"
    assert pt3.on_bar_close(_pos(setup="breakout", breakout_level=98.0), {"close": 97.5}, "d", _params()) is None


def test_time_exit_after_max_hold_bars():
    reason = pt3.on_bar_close(_pos(bars_held=9), {"rsi": 50.0, "bb_pband": 0.5, "close": 100.0}, "d", _params())
    assert reason == "시간청산(10거래일)"
