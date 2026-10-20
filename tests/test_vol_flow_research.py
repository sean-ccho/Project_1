"""vol_flow_research — 거래량·수급 지표 계산, 미래 데이터 미사용, 사건 쿨다운, FINRA 파일 파싱."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ta.volume import ChaikinMoneyFlowIndicator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import vol_flow_research as vf  # noqa: E402


def _ohlcv(n: int = 260, seed: int = 0, cols: tuple[str, ...] = ("A", "B")) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.015, (n, len(cols))), axis=0)), idx, list(cols))
    high = close * (1 + rng.uniform(0.001, 0.02, close.shape))
    low = close * (1 - rng.uniform(0.001, 0.02, close.shape))
    vol = pd.DataFrame(rng.uniform(1e6, 2e6, close.shape), idx, list(cols))
    return {"Close": close, "High": high, "Low": low, "Volume": vol}


def test_udv_ratio_matches_hand_count():
    """오른 날 거래량 합 / 내린 날 거래량 합 (보합일 제외)."""
    idx = pd.bdate_range("2021-01-01", periods=6)
    close = pd.DataFrame({"A": [10, 11, 11, 10, 12, 11]}, idx, dtype=float)
    vol = pd.DataFrame({"A": [5, 100, 7, 30, 200, 20]}, idx, dtype=float)
    out = vf.udv_ratio(close, vol, n=5, min_days=4)
    # 마지막 날 5일 창 = 2..6일: 위 100·200, 아래 30·20, 보합 7(제외) → 300/50
    assert np.isclose(out["A"].iloc[-1], 6.0)
    assert vf.udv_ratio(close, vol, n=5, min_days=5)["A"].isna().all()  # 유효일 4 < 5


def test_cmf_matches_ta_library():
    d = _ohlcv(120, cols=("A",))
    ours = vf.cmf(d["High"], d["Low"], d["Close"], d["Volume"])["A"]
    ref = ChaikinMoneyFlowIndicator(d["High"]["A"], d["Low"]["A"], d["Close"]["A"], d["Volume"]["A"], window=20).chaikin_money_flow()
    both = pd.concat([ours, ref], axis=1).dropna().iloc[5:]  # ta 는 처음 19일을 다르게 채운다
    assert len(both) > 80 and np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-9)


def test_signals_do_not_use_future_data():
    """t 이후 값을 바꿔도 t 까지의 신호는 그대로."""
    d = _ohlcv(300)
    cut = 250
    e = {k: v.copy() for k, v in d.items()}
    for k in e:
        e[k].iloc[cut + 1:] *= 3.0
    pairs = [
        (vf.udv_ratio(d["Close"], d["Volume"]), vf.udv_ratio(e["Close"], e["Volume"])),
        (vf.cmf(d["High"], d["Low"], d["Close"], d["Volume"]), vf.cmf(e["High"], e["Low"], e["Close"], e["Volume"])),
        (vf.drybo_flags(d["Close"], d["Volume"]), vf.drybo_flags(e["Close"], e["Volume"])),
        (vf.bdiv_flags(d["High"], d["Low"], d["Close"], d["Volume"]), vf.bdiv_flags(e["High"], e["Low"], e["Close"], e["Volume"])),
    ]
    for a, b in pairs:
        pd.testing.assert_frame_equal(a.iloc[:cut + 1], b.iloc[:cut + 1])
    ha, hb = vf.hvp_flags(d["Close"], d["Volume"]), vf.hvp_flags(e["Close"], e["Volume"])
    last_full = ha.index[ha.index <= d["Close"].index[cut]][:-1]  # cut 이 든 주는 뒤가 바뀌므로 그 전 주까지
    pd.testing.assert_frame_equal(ha.loc[last_full], hb.loc[last_full])


def test_hvp_flags_volume_spike_with_flat_price():
    idx = pd.bdate_range("2021-01-04", periods=5 * 14)  # 월요일부터 14주
    close = pd.DataFrame({"A": 100.0, "B": 100.0}, idx)
    base = 1e6 * (1 - 0.001 * np.arange(len(idx)))  # 조금씩 줄어드는 거래량 (동점 없음 — 실제 거래량처럼)
    vol = pd.DataFrame({"A": base, "B": base}, idx)
    wk12 = idx[5 * 12:5 * 13]
    vol.loc[wk12, ["A", "B"]] = 5e6
    close.loc[wk12[-1]:, "B"] = 110.0  # B 는 그 주 +10% → 가격이 움직였으니 제외
    f = vf.hvp_flags(close, vol)
    assert bool(f.loc[wk12[-1], "A"]) and not bool(f.loc[wk12[-1], "B"])
    assert int(f["A"].sum()) == 1  # 앞 10주가 없는 초반 주·평범한 주는 아님


def test_drybo_needs_dry_spell_then_breakout():
    n = 120
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = pd.Series(np.linspace(100, 110, n), idx)  # 천천히 오르는 추세 (종가 > SMA50)
    close.iloc[-1] = 120.0  # 마지막 날 20일 고점 돌파
    vol = pd.Series(1e6, idx)
    vol.iloc[-8:-1] = 3e5  # 직전 며칠 거래량 고갈
    vol.iloc[-1] = 3e6
    f = vf.drybo_flags(close.to_frame("A"), vol.to_frame("A"))["A"]
    assert bool(f.iloc[-1])
    vol2 = vol.copy()
    vol2.iloc[-8:-1] = 1e6  # 고갈 없음 → 사건 아님
    assert not bool(vf.drybo_flags(close.to_frame("A"), vol2.to_frame("A"))["A"].iloc[-1])


def test_bdiv_near_low_below_sma200_with_buying_flow():
    n = 260
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(np.linspace(200, 100, n), idx)  # 내리막 → SMA200 아래, 60일 저점 근처
    high = close * 1.01
    low = close * 0.99
    vol = pd.Series(1e6, idx)
    up = close.copy()
    up.iloc[-20:] = high.iloc[-20:] * 0.999  # 마지막 20일 종가가 고가 근처 → CMF 양(+)
    f = vf.bdiv_flags(high.to_frame("A"), low.to_frame("A"), up.to_frame("A"), vol.to_frame("A"))["A"]
    assert bool(f.iloc[-1])
    dn = close.copy()
    dn.iloc[-20:] = low.iloc[-20:] * 1.001  # 종가가 저가 근처 → 매집 아님, 대조군 쪽
    assert not bool(vf.bdiv_flags(high.to_frame("A"), low.to_frame("A"), dn.to_frame("A"), vol.to_frame("A"))["A"].iloc[-1])
    assert bool(vf.bdiv_flags(high.to_frame("A"), low.to_frame("A"), dn.to_frame("A"), vol.to_frame("A"), sign=-1)["A"].iloc[-1])


def test_flag_events_cooldown():
    idx = pd.bdate_range("2021-01-01", periods=30)
    flags = pd.DataFrame(False, idx, ["A"])
    flags.iloc[[0, 3, 10, 12, 25], 0] = True
    ev = vf.flag_events(flags, 10)
    assert list(ev["date"]) == [idx[0], idx[10], idx[25]]


def test_parse_finra_drops_trailer_and_normalizes_symbols():
    text = ("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
            "20180801|A|127212|14|658064|Q,N\n"
            "20180801|BRK/B|1000|0|4000|Q,N\n"
            "20180801|ZZZZ|5|0|10|Q\n"
            "3\n")
    out = vf.parse_finra(text, keep={"A", "BRK-B"}).set_index("티커")
    assert list(out.index) == ["A", "BRK-B"]
    assert out.loc["BRK-B", "sv"] == 1000 and out.loc["BRK-B", "tv"] == 4000
    assert out["date"].iloc[0] == pd.Timestamp("2018-08-01")


def test_svr_signals_ratio_of_sums():
    idx = pd.bdate_range("2021-01-01", periods=70)
    sv = pd.DataFrame({"A": 40.0}, idx)
    tv = pd.DataFrame({"A": 100.0}, idx)
    sv.iloc[-5:] = 80.0
    svr20, dsvr = vf.svr_signals(sv, tv)
    assert np.isclose(svr20["A"].iloc[-1], (15 * 40 + 5 * 80) / (20 * 100))
    assert np.isclose(dsvr["A"].iloc[-1], 0.8 - (55 * 40 + 5 * 80) / 6000)


def test_month_ends_last_trading_day():
    idx = pd.DatetimeIndex(["2021-01-28", "2021-01-29", "2021-02-01", "2021-02-26"])
    assert list(vf.month_ends(idx)) == [pd.Timestamp("2021-01-29"), pd.Timestamp("2021-02-26")]
