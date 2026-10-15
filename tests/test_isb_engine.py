"""scripts/isb_engine.py 테스트 — ta 함수 손계산, 동점 규칙, 상태 기계 불변식, 룩어헤드 없음, 주봉 → 일봉 매핑."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import isb_engine as e  # noqa: E402

HAS_DATA = e.PRICE_CACHE.exists() and e.MACRO_PATH.exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="가격 캐시·VIX/DXY 파일 없음 (로컬 전용)")


# ── ta 함수 ───────────────────────────────────────────────
def test_ema_and_rma_start_from_sma():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    ema = e.ta_ema(x, 3)  # alpha 0.5
    assert np.isnan(ema[:2]).all()
    assert ema[2] == pytest.approx(2.0) and ema[3] == pytest.approx(3.0) and ema[4] == pytest.approx(4.0)
    rma = e.ta_rma(x, 2)  # alpha 0.5, 첫 값 = (1+2)/2
    assert np.isnan(rma[0]) and rma[1] == pytest.approx(1.5) and rma[2] == pytest.approx(2.25)


def test_ema_restarts_after_nan():
    ema = e.ta_ema([np.nan, 1.0, 3.0, 5.0], 2)
    assert np.isnan(ema[:2]).all() and ema[2] == pytest.approx(2.0)
    assert ema[3] == pytest.approx(2.0 / 3 * 5 + 1.0 / 3 * 2.0)


def test_rsi_wilder_hand_computed():
    rsi = e.ta_rsi([1.0, 2.0, 3.0, 2.0, 4.0], 2)
    assert np.isnan(rsi[:2]).all()
    assert rsi[2] == pytest.approx(100.0)        # 하락 평균 0, 상승 있음
    assert rsi[3] == pytest.approx(50.0)         # up 0.5 / down 0.5
    assert rsi[4] == pytest.approx(100 - 100 / 6)  # up 1.25 / down 0.25


def test_atr_first_true_range_is_high_minus_low():
    h, l, c = [10.0, 12.0, 11.0], [8.0, 9.0, 7.0], [9.0, 11.0, 8.0]
    tr = e.ta_tr(h, l, c)
    assert list(tr) == pytest.approx([2.0, 3.0, 4.0])
    atr = e.ta_atr(h, l, c, 2)
    assert np.isnan(atr[0]) and atr[1] == pytest.approx(2.5) and atr[2] == pytest.approx(3.25)


def test_stdev_population_and_percentrank():
    assert e.ta_stdev([1.0, 2.0, 3.0], 3)[2] == pytest.approx(math.sqrt(2 / 3))
    pr = e.ta_percentrank([1.0, 2.0, 3.0, 2.0, 5.0], 2)
    assert np.isnan(pr[:2]).all()
    assert list(pr[2:]) == pytest.approx([100.0, 50.0, 100.0])  # 직전 2개 중 현재값 이하 비율 (같은 값 포함)


def test_lowest_offset_ties_pick_most_recent():
    off = e.lowest_offset([3.0, 1.0, 2.0, 1.0], 4)
    assert np.isnan(off[:3]).all() and off[3] == 0.0
    assert e.lowest_offset([3.0, 1.0, 2.0, 4.0], 4)[3] == 2.0


def test_pivot_confirmed_late_and_tie_rule():
    ph = e.pivot_high([1.0, 3.0, 2.0], 1, 1)
    assert np.isnan(ph[:2]).all() and ph[2] == 3.0          # 오른쪽 1봉 뒤 확정
    tie = e.pivot_high([1.0, 3.0, 3.0, 1.0], 1, 1)
    assert np.isnan(tie[2]) and tie[3] == 3.0               # 같은 고가 둘 → 뒤 봉이 피벗 (왼쪽 이상 · 오른쪽 초과)
    low_tie = e.pivot_low([3.0, 1.0, 1.0, 3.0], 1, 1)
    assert np.isnan(low_tie[2]) and low_tie[3] == 1.0


def test_barssince_valuewhen_crossover():
    cond = [False, True, False, False, True]
    assert list(np.nan_to_num(e.barssince(cond), nan=-1)) == [-1, 0, 1, 2, 0]
    assert list(np.nan_to_num(e.valuewhen(cond, [10, 11, 12, 13, 14]), nan=-1)) == [-1, 11, 11, 11, 14]
    assert list(e.crossover([40.0, 49.0, 51.0, 52.0, 48.0, 50.5], 50)) == [False, False, True, False, False, True]


def test_stop_level_caps_and_log_fallback():
    assert e.stop_level(10.0, 0.25, 11.0, 1.0, 0.05, 3.0) == pytest.approx(9.75)
    assert e.stop_level(5.0, 0.25, 11.0, 1.0, 0.05, 3.0) == pytest.approx(8.0)    # 최대 3 ATR
    assert e.stop_level(10.9, 0.25, 11.0, 1.0, 0.05, 3.0) == pytest.approx(10.5)  # 최소 0.5 ATR
    s = e.stop_level(0.9, 0.25, 1.0, 2.0, 0.3, 3.0)                              # ATR > 가격 → 로그 ATR
    assert 0 < s < 1.0 and s == pytest.approx(min(max(0.9 * math.exp(-0.075), math.exp(-0.9)), math.exp(-0.15)))
    assert math.isnan(e.stop_level(float("nan"), 0.25, 11.0, 1.0, 0.05, 3.0))


def test_regime_rising_then_falling():
    c = np.r_[np.linspace(100, 200, 60), np.linspace(200, 100, 60)]
    atr = np.full(c.size, 1.0)
    stage, ma, slope, _ = e.f_regime(c, atr, 10, 3, 0.25, 0.25, 1.0)
    assert (stage[:12] == 0).all()           # MA·기울기 준비 전
    assert (stage[15:58] == 2).all()
    assert stage[75] == 4 and stage[-1] == 4


def test_dividend_factors_undo_yahoo_adjustment():
    # 실제 종가 100, 99, 100 · 둘째 날 배당락 1.0 → Yahoo 조정: 첫날 × (1 − 1/100)
    adj = [99.0, 99.0, 100.0]
    fac = e.dividend_factors(adj, [0.0, 1.0, 0.0])
    assert list(fac) == pytest.approx([0.99, 1.0, 1.0])
    assert adj[0] / fac[0] == pytest.approx(100.0)
    assert list(e.dividend_factors([10.0, 11.0], [0.0, 0.0])) == [1.0, 1.0]


# ── 주봉 만들기 · 주봉 → 일봉 매핑 ─────────────────────────
def _days(dates: list[str]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(dates)
    n = len(idx)
    return pd.DataFrame({"Open": np.arange(n) + 10.0, "High": np.arange(n) + 11.0, "Low": np.arange(n) + 9.0,
                         "Close": np.arange(n) + 10.5, "Volume": np.full(n, 1e6)}, index=idx)


def test_to_weekly_holiday_friday_and_first_day():
    # 2020-07-03(금) 휴장, 2020-09-07(월) 휴장
    d = _days(["2020-06-29", "2020-06-30", "2020-07-01", "2020-07-02", "2020-09-08", "2020-09-11"])
    wk = e.to_weekly(d)
    assert list(wk.index.strftime("%Y-%m-%d")) == ["2020-07-02", "2020-09-11"]
    assert list(pd.DatetimeIndex(wk["first_day"]).strftime("%Y-%m-%d")) == ["2020-06-29", "2020-09-08"]
    assert wk["Open"].iloc[0] == 10.0 and wk["Close"].iloc[0] == 13.5 and wk["High"].iloc[0] == 14.0


def test_weekly_to_daily_mapping():
    d = _days(["2020-06-29", "2020-06-30", "2020-07-01", "2020-07-02",
               "2020-07-06", "2020-07-07", "2020-07-08", "2020-07-09", "2020-07-10"])
    d.loc["2020-07-08", "High"] = 30.0                         # 둘째 주 수요일에 T1(20) 도달
    d["adj_factor"] = 0.98                                     # 엔진(배당 조정) 가격 = TradingView 가격 × 0.98
    wk = e.to_weekly(d)
    eng = pd.DataFrame({
        "pos": [0, 1], "rem": [1.0, 1.0], "hold_next": [True, False],
        "ev_buy": [True, False], "ev_exit": [False, True], "ev_top": [False, False], "ev_topwarn": [False, False],
        "ev_break": [False, False], "ev_t1": [False, True], "t1_level": [np.nan, 20.0], "t1_fill": [np.nan, 20.0],
        "sig_type": ["PB", ""], "grade": [3, 0], "ev_stop": [9.0, np.nan], "exit_why": ["", "Trail"],
        "btm": [False, False], "stage": [2, 2], "period": wk["period"].to_numpy(),
    }, index=wk.index)
    ev = e.daily_events(eng, d, e.IsbParams("W"))
    # 첫 주: 마지막 거래일(07-02, 금요일 휴장)에만 BUY · hold_next
    assert list(ev["buy_ev"]) == [False, False, False, True] + [False] * 5
    assert list(ev["hold_next"][:4]) == [False, False, False, True]
    # 둘째 주: 주중엔 보유(pos=1), 마지막 날 종가에 EXIT → hold_next False
    assert list(ev["hold_next"][4:]) == [True, True, True, True, False]
    assert list(ev["exit_full"][4:]) == [False, False, False, False, True]
    # T1: 그 주 처음으로 고가가 20 에 닿은 수요일, max(시가 16, 20) = 20 → 엔진 가격 20 × 0.98. 남은 비율은 다음 날부터
    assert ev.loc["2020-07-08", "t1_fill"] == pytest.approx(20.0 * 0.98)
    assert ev["t1_fill"].notna().sum() == 1
    assert list(ev["rem"][4:]) == pytest.approx([1.0, 1.0, 1.0, 0.67, 0.67])


# ── 실제 데이터: 상태 기계 불변식 · 룩어헤드 없음 ───────────
@pytest.fixture(scope="module")
def data():
    prices = e.load_daily_prices()
    return prices, e.load_macro()


@needs_data
@pytest.mark.parametrize("tf", ["D", "W"])
@pytest.mark.parametrize("ticker", ["AAPL", "JPM", "XOM", "NVDA"])
def test_state_machine_invariants(data, tf, ticker):
    prices, macro = data
    p = e.IsbParams(tf)
    mkt = e.build_market(prices["SPY"], macro, p)
    bars, inp = e.prepare_bars(prices[ticker], mkt, p)
    eng, trades = e.run_engine(bars, inp, p, ticker)
    pos, hold = eng["pos"].to_numpy(), eng["hold_next"].to_numpy()
    assert (pos[1:] == hold[:-1].astype(int)).all()                 # 다음 시가 보유 = 다음 봉 포지션
    assert not (eng["ev_buy"] & (eng["pos"] == 1)).any()             # 보유 중 추가 매수 없음
    assert not (eng["ev_break"] & (eng["pos"] == 1)).any()           # BREAKDOWN 은 미보유일 때만
    tw = eng["ev_topwarn"]
    assert (eng.loc[tw, "pos"] == 0).all() and eng.loc[tw, "stage"].isin([2, 3]).all()
    assert eng["rem"].min() >= 0.34 - 1e-9
    if p.daily:
        assert not eng["ev_t1"].any()                                # Trend Hold 는 T1 없음
    first_day = pd.DatetimeIndex(bars["first_day"]) if "first_day" in bars.columns else bars.index
    for t in trades:
        i = t["entry_bar"]
        assert t["entry_px"] == pytest.approx(bars["Open"].iloc[i])  # 다음 봉 시가 체결
        assert t["entry_date"] == first_day[i] and eng["ev_buy"].iloc[i - 1]
        assert len(t["trims"]) <= 2
        if not p.daily:
            assert all(pd.Timestamp(d) > bars.index[i] for kind, d, _ in t["trims"] if kind == "T1")  # 진입 봉엔 T1 없음
        if t["exit_bar"] >= 0:
            assert eng["ev_exit"].iloc[t["exit_bar"] - 1] and t["exit_px"] == pytest.approx(bars["Open"].iloc[t["exit_bar"]])


@needs_data
@pytest.mark.parametrize("tf,cut", [("D", "2019-06-28"), ("D", "2022-03-15"), ("W", "2021-04-09")])
def test_no_lookahead_truncation(data, tf, cut):
    """데이터를 날짜 cut 에서 잘라 계산해도 cut 까지의 결과가 같다."""
    prices, macro = data
    p = e.IsbParams(tf)
    cut_ts = pd.Timestamp(cut)
    for ticker in ("AAPL", "MSFT", "CAT"):
        full, _ = e.run_engine(*e.prepare_bars(prices[ticker], e.build_market(prices["SPY"], macro, p), p), p)
        spy_cut, tk_cut = prices["SPY"].loc[:cut_ts], prices[ticker].loc[:cut_ts]
        part, _ = e.run_engine(*e.prepare_bars(tk_cut, e.build_market(spy_cut, macro.loc[:cut_ts], p), p), p)
        common = part.index[part.index <= cut_ts]
        cols = [c for c in e.ENGINE_COLS if c not in ("exit_why",)]
        pd.testing.assert_frame_equal(full.loc[common, cols], part.loc[common, cols], check_dtype=False)


@needs_data
def test_tv_basis_matches_tradingview_abt(data):
    """2026-10-12 사용자 TradingView 화면 (ABT 일봉, ADJ 꺼짐) 과 같은 가격·라벨."""
    prices, macro = data
    bar = prices["ABT"].loc["2024-11-14"]
    assert [round(bar[k], 2) for k in ("Open", "High", "Low", "Close")] == [115.98, 116.09, 114.93, 115.03]
    assert bar["Volume"] == pytest.approx(3812300)
    p = e.IsbParams("D")
    eng, _, _ = e.compute_ticker("ABT", prices["ABT"], e.build_market(prices["SPY"], macro, p), p)
    labels = dict(e.event_lines(eng, p, "2023-11-01", "2025-09-30"))
    assert labels["2023-11-15"].startswith("BUY·EARLY")
    assert labels["2024-07-16"] == "EXIT · Stage"
    assert labels["2024-11-14"].startswith("BUY·S2 ·1/4")
    assert labels["2024-12-12"] == "EXIT · Stop"
    assert labels["2025-01-21"].startswith("BUY·BO 🏆4/4") and labels["2025-01-24"] == "TRIM·TOP"
