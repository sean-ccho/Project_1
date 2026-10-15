"""주봉 BO·BOTTOM 규칙 테스트 — 보유 관리 시뮬레이터, 사용자 규칙 봉별 표, 실행 스크립트 정의."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import isb_bo_bottom as bb  # noqa: E402
import isb_engine as e  # noqa: E402

HAS_DATA = e.PRICE_CACHE.exists() and e.MACRO_PATH.exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="가격 캐시·VIX/DXY 파일 없음 (로컬 전용)")


def _aux(o, h, c, n_btm_stop=None):
    n = len(c)
    dates = pd.date_range("2020-01-03", periods=n, freq="W-FRI")
    return {"o": np.asarray(o, float), "h": np.asarray(h, float), "c": np.asarray(c, float), "atr": np.ones(n),
            "ema20": np.full(n, 5.0), "lg_atr": np.full(n, 0.1), "top_cond": np.zeros(n, bool),
            "stage": np.ones(n, int), "reg_prev": np.ones(n, int), "top_armed_end": np.ones(n, bool),
            "btm_stop": np.asarray(n_btm_stop if n_btm_stop is not None else [np.nan] * n, float),
            "fills": dates, "dates": dates}


def test_simulate_position_t1_tightens_trail():
    aux = _aux(o=[10] * 8, h=[10.2, 12.5, 13.0, 10.5, 10.5, 10.5, 10.5, 10.5], c=[10, 12, 12.5, 9.9, 10, 10, 10, 10])
    sim = e.simulate_position(aux, 0, 9.0, True, e.IsbParams("W"))
    # 위험 1 → T1 12 (1봉에 체결, 트림 1회) · +1R 뒤 트레일 켬 · 트림 뒤 3 ATR: 13 − 3 = 10 → 3봉 종가 9.9 < 10
    assert sim["trims"] == [("T1", 1)]
    assert (sim["exit_signal_bar"], sim["exit_bar"], sim["exit_why"]) == (3, 4, "Trail")
    assert sim["stop0"] == pytest.approx(9.0)


def test_simulate_position_open_below_plan_stop_resets_stop():
    aux = _aux(o=[10, 8.0, 8, 8], h=[10, 8.2, 8.2, 8.2], c=[10, 8.1, 7.0, 8])
    sim = e.simulate_position(aux, 1, 9.0, True, e.IsbParams("W"))    # 시가 8 < 계획 손절 9 → 8 − 0.5 ATR = 7.5
    assert sim["stop0"] == pytest.approx(7.5) and sim["exit_signal_bar"] == 2 and sim["exit_why"] == "Stop"


def test_bo_bottom_bars_paths_and_skip_while_holding():
    n = 10
    aux = _aux(o=[10] * n, h=[10.4] * n, c=[10, 10, 10, 10, 9, 10, 10, 10, 10, 10],
               n_btm_stop=[np.nan, 9.5, np.nan, 9.5] + [np.nan] * 6)
    eng = pd.DataFrame({"ev_buy": [False] * 6 + [True] + [False] * 3, "sig_type": [""] * 6 + ["BO"] + [""] * 3,
                        "btm": [False, True, False, True] + [False] * 6, "ev_stop": [np.nan] * 6 + [9.0] + [np.nan] * 3,
                        "stage": [1] * n}, index=aux["dates"])
    trades = [{"entry_bar": 7, "exit_bar": -1, "exit_why": ""}]          # ISB 실제 BO 거래 (기간 끝까지 보유)
    frame, user = e.bo_bottom_bars(eng, trades, aux, e.IsbParams("W"), "AAA")
    assert list(frame["pos"]) == [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]           # BOTTOM 1봉 → 2봉 진입, 4봉 손절 → 5봉 매도
    assert list(frame["hold_next"]) == [False, True, True, True, False, False, True, True, True, True]
    assert list(frame.index[frame["ev_buy"]]) == [aux["dates"][1], aux["dates"][6]]   # 3봉 BOTTOM 은 보유 중이라 무시
    assert frame["ev_exit"].sum() == 1 and frame["exit_why"].iloc[4] == "Stop"
    assert [t["sig_type"] for t in user] == ["BOTTOM", "BO"]
    assert user[0]["ret"] == pytest.approx(10 / 10 - 1 - (0.02 / 10 + 0.002)) and math.isnan(user[1]["ret"])
    assert (frame["rem"] == 1.0).all() and not frame["ev_t1"].any() and not frame["ev_top"].any()   # 트림 없음


def test_runner_definitions():
    assert bb.P_BONF == pytest.approx(0.025) and bb.TESTS == ("WBB-solo", "WBB-pt1")
    sp = bb.specs()
    assert sp["WBB-pt1"].tf == e.BB_KEY and sp["WBB-pt1"].gate == "isb" and sp["WBB-pt1"].sells == "isb"
    assert not sp["WBB-pt1"].t1                                           # T1 트림 없음
    assert sum(k.startswith("WBB-solo3-") for k in sp) == 20
    assert "BOTTOM" in bb.spec() and "EXIT 에서만 전량" in bb.spec()


@pytest.fixture(scope="module")
def data():
    return e.load_daily_prices(), e.load_macro()


@needs_data
@pytest.mark.parametrize("tf", ["D", "W"])
def test_simulator_reproduces_engine_trades(data, tf):
    prices, macro = data
    p = e.IsbParams(tf)
    mkt = e.build_market(prices["SPY"], macro, p)
    tickers = sorted(t for t in prices if t != "SPY")[::15]
    n_trades, bad = 0, []
    for t in tickers:
        aux: dict = {}
        eng, trades = e.run_engine(*e.prepare_bars(prices[t], mkt, p), p, t, aux_out=aux)
        n_trades += len(trades)
        bad += e.check_simulator(eng, trades, aux, p)
    assert n_trades > 50 and bad == []


@needs_data
@pytest.mark.parametrize("ticker", ["AAPL", "CSCO", "JPM", "XOM", "CAT", "NVDA"])
def test_bo_bottom_bars_invariants(data, ticker):
    prices, macro = data
    p = e.IsbParams("W")
    mkt = e.build_market(prices["SPY"], macro, p)
    aux: dict = {}
    eng, trades = e.run_engine(*e.prepare_bars(prices[ticker], mkt, p), p, ticker, aux_out=aux)
    frame, user = e.bo_bottom_bars(eng, trades, aux, p, ticker)
    pos, hold = frame["pos"].to_numpy(), frame["hold_next"].to_numpy()
    assert (pos[1:] == hold[:-1].astype(int)).all()
    buys = frame.index[frame["ev_buy"]]
    for d in buys:                                                        # 매수는 BO 또는 BOTTOM 봉에서만
        assert (eng.at[d, "ev_buy"] and eng.at[d, "sig_type"] == "BO") or eng.at[d, "btm"]
    std = {t["entry_bar"] - 1: t for t in trades}
    for t in user:                                                        # BO 는 ISB 실제 거래의 EXIT 그대로
        if t["sig_type"] == "BO":
            assert t["exit_bar"] == std[t["entry_bar"] - 1]["exit_bar"]
    spans = [(t["entry_bar"], t["exit_bar"] if t["exit_bar"] >= 0 else len(frame)) for t in user]
    assert all(a2 >= b1 for (_, b1), (a2, _) in zip(spans, spans[1:]))   # 겹치는 거래 없음
    ev = e.daily_events(frame, prices[ticker], p)
    assert not ev["trim_ev"].any() and ev["t1_fill"].isna().all() and (ev["rem"] == 1.0).all()
