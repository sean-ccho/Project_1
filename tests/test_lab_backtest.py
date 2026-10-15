"""연구소 백테스터 테스트 — 결정일, 체결 시점, 비용·현금·레버리지, 매매 폭, 비중 표류, idea_batch 재현, PBO."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lab_backtest as lb  # noqa: E402
from paper_trading.evaluation import cagr, max_drawdown, sharpe  # noqa: E402
from research_utils import RESEARCH_DIR, pbo_cscv  # noqa: E402


def test_decision_mask_month_week_day():
    idx = pd.DatetimeIndex(["2024-03-26", "2024-03-27", "2024-03-28",          # 3/29 성금요일 휴장
                            "2024-04-01", "2024-04-02", "2024-04-05", "2024-04-08"])
    m = lb.decision_mask(idx, "M")
    assert list(idx[m].strftime("%m-%d")) == ["03-28", "04-08"]              # 마지막 행은 항상 결정일
    w = lb.decision_mask(idx, "W")
    assert list(idx[w].strftime("%m-%d")) == ["03-28", "04-05", "04-08"]     # 휴장 금요일 → 목요일
    assert lb.decision_mask(idx, "D").all()


def _one(values, idx):
    return pd.DataFrame({"X": values}, index=idx)


def test_close_and_open_execution_timing():
    idx = pd.bdate_range("2024-01-01", periods=6)
    ret = _one([0.01] * 6, idx)
    tgt = _one([np.nan, 1.0, np.nan, np.nan, np.nan, np.nan], idx)          # t=1 에 결정
    c = lb.simulate(tgt, ret, freq="D", cost=0.0)
    assert list(c.weights["X"]) == [0, 0, 0, 1, 1, 1]                         # 다음 날(2) 종가 체결 → 3 부터 수익
    o = lb.simulate(tgt, ret, freq="D", cost=0.0, exec_mode="open")
    assert list(o.weights["X"]) == [0, 0, 1, 1, 1, 1]                         # 다음 날(2) 시가 체결 → 2 의 시가→시가
    dl = lb.simulate(tgt, ret, freq="D", cost=0.0, delay=1)
    assert list(dl.weights["X"]) == [0, 0, 0, 0, 1, 1]


def test_costs_cash_interest_and_borrowing():
    idx = pd.bdate_range("2024-01-01", periods=4)
    ret = _one([0.0, 0.0, 0.0, 0.02], idx)
    rf = pd.Series(0.0001, index=idx)
    tgt = _one([0.5, np.nan, np.nan, np.nan], idx)
    s = lb.simulate(tgt, ret, rf, freq="D", cost=0.001, drift=False)
    assert s.ret.iloc[1] == pytest.approx(1.0 * 0.0001)                       # 현금 100% 이자
    assert s.ret.iloc[2] == pytest.approx(0.5 * 0.0001 - 0.5 * 0.001)         # 체결일: 반 현금 이자 − 매매 비용
    assert s.ret.iloc[3] == pytest.approx(0.5 * 0.02 + 0.5 * 0.0001)
    lev = lb.simulate(_one([1.5, np.nan, np.nan, np.nan], idx), ret, rf, freq="D", cost=0.0, drift=False,
                      borrow_spread=0.0252)
    assert lev.ret.iloc[3] == pytest.approx(1.5 * 0.02 - 0.5 * (0.0001 + 0.0001))   # 빌린 0.5 에 무위험 + 2.52%/252


def test_band_suppresses_small_rebalances_and_drift_moves_weights():
    idx = pd.bdate_range("2024-01-01", periods=8)
    ret = pd.DataFrame({"A": [0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0], "B": 0.0}, index=idx)
    tgt = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)
    s = lb.simulate(tgt, ret, freq="D", band=0.05, cost=0.0)
    a = s.weights["A"]
    assert a.iloc[4] == pytest.approx(0.55 / 1.05)                            # 표류: 0.5 × 1.1 / 1.05
    assert s.trades == 1                                                      # 표류 2.4%p 는 폭 5%p 안 → 첫 매수뿐
    s2 = lb.simulate(tgt, ret, freq="D", band=0.0, cost=0.0)
    assert s2.weights["A"].iloc[-1] == pytest.approx(0.5)                     # 폭 없으면 다시 맞춘다


def test_band_rule_counts():
    idx = pd.bdate_range("2024-01-01", periods=8)
    ret = pd.DataFrame({"A": [0.0, 0.0, 0.0, 0.30, 0.0, 0.0, 0.0, 0.0], "B": 0.0}, index=idx)
    tgt = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)
    small = lb.simulate(tgt, ret, freq="D", band=0.10, cost=0.0)            # 0.65/1.15=0.565 → 차이 6.5%p < 10%p
    big = lb.simulate(tgt, ret, freq="D", band=0.05, cost=0.0)              # 6.5%p > 5%p → 다시 맞춤
    assert small.trades == 1 and big.trades == 2
    assert big.weights["A"].iloc[-1] == pytest.approx(0.5)


def test_monthly_schedule_trades_only_at_month_end():
    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    ret = _one(np.full(len(idx), 0.001), idx)
    tgt = _one(np.where(idx.month == 2, 0.3, 0.8), idx)
    s = lb.simulate(tgt, ret, freq="M", cost=0.0, drift=False)
    changes = s.weights["X"].diff().fillna(s.weights["X"]).ne(0)
    days = list(s.weights.index[changes].strftime("%m-%d"))
    assert days == ["02-02", "03-04"]                                         # 1월 말 결정(0.8) → 2/2, 2월 말(0.3) → 3/4


def test_perf_matches_evaluation_module():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0004, 0.01, 600), index=pd.bdate_range("2020-01-01", periods=600))
    p = lb.perf(r)
    assert p["Sharpe_raw"] == pytest.approx(sharpe(r.tolist()))
    assert p["CAGR"] == pytest.approx(cagr(r.tolist()))
    assert p["MDD"] == pytest.approx(max_drawdown(r.tolist()))
    rf = pd.Series(0.0001, index=r.index)
    assert lb.perf(r, rf)["Sharpe"] == pytest.approx(sharpe((r - rf).tolist()))
    assert lb.score({"Sharpe": 0.8, "MDD": -0.40}) == pytest.approx(0.6)


def test_vol_overlay_scales_and_caps():
    idx = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(2)
    ret = pd.DataFrame({"X": rng.normal(0, 0.01, 100)}, index=idx)
    tgt = _one(np.ones(100), idx)
    out = lb.vol_overlay(tgt, ret, 0.08, freq="D", lookback=63, cap=1.5)
    i = 80
    sig = ret["X"].iloc[i - 62:i + 1].std() * math.sqrt(252)
    assert out["X"].iloc[i] == pytest.approx(min(1.5, 0.08 / sig))
    assert out["X"].iloc[:62].isna().all()
    hi = lb.vol_overlay(tgt, ret, 5.0, freq="D", lookback=63, cap=1.5)
    assert hi["X"].iloc[90] == pytest.approx(1.5)


def test_synthetic_2x():
    idx = pd.bdate_range("2020-01-01", periods=3)
    r = pd.Series([0.01, -0.02, 0.0], index=idx)
    rf = pd.Series(0.0002, index=idx)
    out = lb.synthetic_2x(r, rf, spread=0.0252)
    assert list(out.round(6)) == [round(0.02 - 0.0003, 6), round(-0.04 - 0.0003, 6), round(-0.0003, 6)]


@pytest.mark.skipif(not (RESEARCH_DIR / "etf_ohlcv.parquet").exists(), reason="idea_batch 자료 없음")
def test_reproduces_idea_batch_vol_managed_spy():
    from baseline_compare import timing_returns
    from idea_batch_research import START, vol_managed_weights
    from research_utils import load_ohlcv
    spy = load_ohlcv(RESEARCH_DIR / "etf_ohlcv.parquet")["SPY"]
    tw = vol_managed_weights(spy["Close"])
    ref = timing_returns(spy["Open"], spy["Close"], tw)
    r_oo = (spy["Open"].shift(-1) / spy["Open"] - 1.0).to_frame("SPY")
    mine = lb.simulate(tw.to_frame("SPY"), r_oo, freq="D", exec_mode="open", drift=False).ret.iloc[:-1]
    assert (mine - ref.reindex(mine.index)).abs().max() == 0.0
    assert sharpe(mine.loc[START:].tolist()) == pytest.approx(0.9211624988398148, abs=1e-12)


def test_pbo_noise_vs_edge():
    rng = np.random.default_rng(0)
    vals = [pbo_cscv(rng.normal(0, 0.01, (2000, 60)))["PBO"] for _ in range(6)]
    assert 0.3 < float(np.mean(vals)) < 0.7                                   # 운만 있으면 0.5 근처
    x = rng.normal(0, 0.01, (2000, 60))
    x[:, 5] += 0.002
    assert pbo_cscv(x)["PBO"] < 0.05                                          # 진짜 실력 하나 → 거의 0
    assert math.isnan(pbo_cscv(x[:20])["PBO"])
