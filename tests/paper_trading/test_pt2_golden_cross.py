"""PT-2 골든크로스 스윙 규칙 테스트."""

from paper_trading import pt2_golden_cross as pt2
from screener.config import PT2_PARAMS


def _params(**overrides) -> dict:
    return {**PT2_PARAMS, **overrides}


def _row(ticker: str, **kw) -> dict:
    base = {
        "티커": ticker, "섹터": "Information Technology", "close": 100.0, "현재가격": 100.0,
        "ema50": 95.0, "ema200": 90.0, "atr_value": 2.0, "최근20일평균거래대금": 50e6,
        "RSI": 55.0, "bollinger_pband": 0.6, "5일수익률": 0.02, "ema_gap_50_200": 0.05,
        "52주포지션": 0.8, "adx": 25.0, "거래량돌파배수": 1.2, "매수적합도_표시": "★★★",
        "일봉패턴": "", "주봉패턴": "", "월봉패턴": "",
        "ema_gap_20_50": 0.0, "주봉_MA갭": 0.0, "월봉_MA갭": 0.0,
    }
    base.update(kw)
    return base


def _ctx(regime: str = "bull") -> dict:
    return {"regime": regime, "exclude": {"SPY", "QQQ", "IWM"}, "bar_date": "2026-01-05"}


def test_daily_cross_selected_with_label_and_meta():
    rows = [_row("AAA", 일봉패턴="골든크로스", ema_gap_20_50=0.01)]
    cands, debug = pt2.select_candidates(rows, _ctx(), _params())
    assert [c["ticker"] for c in cands] == ["AAA"]
    assert cands[0]["strategy"] == "골든크로스(일봉직후)"
    assert cands[0]["atr"] == 2.0 and cands[0]["close"] == 100.0
    assert debug["통과"] == 1


def test_imminent_excluded_by_default_but_allowed_by_param():
    rows = [_row("AAA", 일봉패턴="골든크로스임박", ema_gap_20_50=-0.01)]
    cands, debug = pt2.select_candidates(rows, _ctx(), _params())
    assert cands == [] and debug["탈락"] == {"진입유형": 1}
    cands, _ = pt2.select_candidates(rows, _ctx(), _params(entry_types=["crossed", "imminent"]))
    assert cands[0]["strategy"] == "골든크로스(일봉임박)"


def test_filters_reject_weak_candidates():
    rows = [
        _row("LOW", 일봉패턴="골든크로스", ema_gap_20_50=0.01, close=85.0),             # 200일선 아래
        _row("HOT", 일봉패턴="골든크로스", ema_gap_20_50=0.01, RSI=80.0),               # 과열
        _row("ERN", 일봉패턴="골든크로스", ema_gap_20_50=0.01, days_to_next_earnings=2),  # 어닝 임박
        _row("ILL", 일봉패턴="골든크로스", ema_gap_20_50=0.01, 최근20일평균거래대금=1e6),  # 유동성
    ]
    cands, debug = pt2.select_candidates(rows, _ctx(), _params())
    assert cands == []
    assert debug["탈락"] == {"200일선 아래": 1, "과열": 1, "어닝임박": 1, "유동성": 1}


def test_bear_regime_blocks_new_entries():
    rows = [_row("AAA", 일봉패턴="골든크로스", ema_gap_20_50=0.01)]
    cands, debug = pt2.select_candidates(rows, _ctx("bear"), _params())
    assert cands == [] and "약세장" in debug["skip"]


def test_multi_timeframe_ranks_first():
    rows = [
        _row("ONE", 일봉패턴="골든크로스", ema_gap_20_50=0.01),
        _row("TWO", 일봉패턴="골든크로스", ema_gap_20_50=0.01, 주봉패턴="골든크로스", 주봉_MA갭=0.02),
    ]
    cands, _ = pt2.select_candidates(rows, _ctx(), _params())
    assert [c["ticker"] for c in cands] == ["TWO", "ONE"]


def _pos(**kw) -> dict:
    pos = {"entry_price": 100.0, "entry_date": "2026-01-05", "highest_close": 100.0,
           "stop_price": 96.0, "stop_kind": "손절", "atr_at_signal": 2.0, "ema_aligned_seen": False}
    pos.update(kw)
    return pos


def _ind(**kw) -> dict:
    ind = {"close": 101.0, "ema20": 100.5, "ema50": 100.0, "below_ema50_streak": 0.0,
           "atr": 2.0, "adx": 25.0, "ret_20d": 0.05}
    ind.update(kw)
    return ind


def test_init_position_sets_atr_stop():
    pos = {"entry_price": 100.0}
    pt2.init_position(pos, {"atr": 2.0}, _params())
    assert pos["stop_price"] == 96.0 and pos["stop_kind"] == "손절"


def test_dead_cross_only_after_alignment_seen():
    pos = _pos()
    assert pt2.on_bar_close(pos, _ind(ema20=99.0), "2026-01-06", _params()) is None
    assert pt2.on_bar_close(pos, _ind(ema20=101.0), "2026-01-07", _params()) is None
    assert pos["ema_aligned_seen"] is True
    assert pt2.on_bar_close(pos, _ind(ema20=99.0), "2026-01-08", _params()) == "추세이탈(데드크로스)"


def test_below_ema50_streak_exit():
    reason = pt2.on_bar_close(_pos(), _ind(below_ema50_streak=2.0), "2026-01-08", _params())
    assert reason == "추세이탈(50일선 아래 2일)"


def test_after_30_days_weak_trend_sells_strong_trend_holds():
    weak = pt2.on_bar_close(_pos(), _ind(adx=15.0), "2026-02-04", _params())
    assert weak == "보유 30일 추세약화"

    pos = _pos()
    strong = _ind(close=110.0, ema20=105.0, ema50=100.0, adx=25.0, ret_20d=0.05)
    assert pt2.on_bar_close(pos, strong, "2026-02-04", _params()) is None
    assert pos["trend_checks"] == 1


def test_trailing_starts_only_after_profit_threshold():
    pos = _pos(highest_close=104.0)
    pt2.on_bar_close(pos, _ind(), "2026-01-08", _params())
    assert pos["stop_price"] == 96.0

    pos = _pos(highest_close=110.0)
    pt2.on_bar_close(pos, _ind(close=109.0, ema20=105.0), "2026-01-08", _params())
    assert pos["stop_price"] == 104.0 and pos["stop_kind"] == "트레일링"


def test_trailing_tightens_after_hold_days():
    pos = _pos(highest_close=110.0)
    strong = _ind(close=110.0, ema20=105.0, ema50=100.0)
    pt2.on_bar_close(pos, strong, "2026-02-04", _params())
    assert pos["stop_price"] == 105.0
