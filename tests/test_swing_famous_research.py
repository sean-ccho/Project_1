"""swing_famous_research: 지표(Pine·ta 와 같은 값) · 미래 정보 없음 · 체결 시점·비용 회계 · 무작위 시점·부트스트랩 점검."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import swing_famous_research as sf  # noqa: E402


def _walk(n: int = 900, seed: int = 0) -> sf.Px:
    """무작위 걸음 가격 (고가 ≥ 종가·시가 ≥ 저가), 총수익 시가 = 시가 × 1.0001^t (배당 흉내)."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    open_ = close * np.exp(rng.normal(0.0, 0.005, n))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.02, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.02, n))
    dates = pd.bdate_range("2000-01-03", periods=n)
    return sf.Px("TEST", dates, high, low, close, open_ * 1.0001 ** np.arange(n))


def test_rma_matches_pine_definition() -> None:
    v = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 10.0, 0.0])
    out = sf.rma(v, 3)
    exp = np.full(7, np.nan)
    exp[3] = 2.0                                   # 첫 3개 (1, 2, 3) 평균
    for i in range(4, 7):
        exp[i] = (exp[i - 1] * 2 + v[i]) / 3
    assert np.allclose(out, exp, equal_nan=True)


def test_rsi_matches_ta_library_after_warmup() -> None:
    x = _walk()
    ours = sf.rsi(x.close, 14)
    theirs = ta.momentum.RSIIndicator(pd.Series(x.close), window=14).rsi().to_numpy()
    assert np.isnan(ours[13]) and not np.isnan(ours[14])     # Pine: 14번째 변화(봉 14)부터
    assert np.allclose(ours[300:], theirs[300:], atol=1e-6)  # 시작값 차이는 (13/14)^k 로 사라진다


def test_bollinger_matches_ta_library() -> None:
    x = _walk()
    lo, hi = sf.bollinger(x.close, 20, 2.0)
    bb = ta.volatility.BollingerBands(pd.Series(x.close), window=20, window_dev=2)
    assert np.allclose(lo[19:], bb.bollinger_lband().to_numpy()[19:], atol=1e-9)
    assert np.allclose(hi[19:], bb.bollinger_hband().to_numpy()[19:], atol=1e-9)


def _pine_supertrend_reference(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int,
                               mult: float) -> np.ndarray:
    """Pine pine_supertrend 를 한 줄씩 그대로 (supertrend 값 비교 포함) — 구현 대조용."""
    tr = np.empty(len(close))
    for i in range(len(close)):
        tr[i] = high[i] - low[i] if i == 0 else max(high[i] - low[i], abs(high[i] - close[i - 1]),
                                                    abs(low[i] - close[i - 1]))
    atr = np.full(len(close), np.nan)
    for i in range(len(close)):
        if i == n - 1:
            atr[i] = tr[:n].mean()
        elif i >= n:
            atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    lower = np.full(len(close), np.nan)
    upper = np.full(len(close), np.nan)
    st = np.full(len(close), np.nan)
    out = np.zeros(len(close))
    for i in range(len(close)):
        if np.isnan(atr[i]):
            continue
        src = (high[i] + low[i]) / 2
        ub, lb = src + mult * atr[i], src - mult * atr[i]
        plb = 0.0 if i == 0 or np.isnan(lower[i - 1]) else lower[i - 1]
        pub = 0.0 if i == 0 or np.isnan(upper[i - 1]) else upper[i - 1]
        c1 = close[i - 1] if i > 0 else np.nan
        lb = lb if (lb > plb or c1 < plb) else plb
        ub = ub if (ub < pub or c1 > pub) else pub
        prev_st = st[i - 1] if i > 0 else np.nan
        if i == 0 or np.isnan(atr[i - 1]):
            d = 1
        elif prev_st == pub:
            d = -1 if close[i] > ub else 1
        else:
            d = 1 if close[i] < lb else -1
        lower[i], upper[i] = lb, ub
        st[i] = lb if d == -1 else ub
        out[i] = d
    return out


def test_supertrend_matches_pine_reference() -> None:
    for seed in range(3):
        x = _walk(seed=seed)
        for n, m in ((10, 3.0), (7, 2.0), (14, 4.0)):
            ours = sf.supertrend_dir(x.high, x.low, x.close, n, m)
            ref = _pine_supertrend_reference(x.high, x.low, x.close, n, m)
            assert np.array_equal(ours, ref)
            assert set(np.unique(ours[n:])) <= {-1.0, 1.0}


def test_supertrend_flips_with_trend() -> None:
    up = np.linspace(100, 200, 80)
    down = np.linspace(200, 120, 80)
    close = np.concatenate([up, down])
    x = sf.supertrend_dir(close * 1.005, close * 0.995, close, 10, 3.0)
    assert x[9] == 1                               # 첫 ATR 봉은 하락(빨강)에서 시작 (Pine)
    assert (x[40:80] == -1).all()                  # 꾸준히 오르면 초록
    assert (x[120:] == 1).all()                    # 꾸준히 내리면 빨강


def test_signals_use_only_past_bars() -> None:
    x = _walk(seed=5)
    k = 600
    rng = np.random.default_rng(9)
    f = np.ones(len(x.close))
    f[k + 1:] = np.exp(np.cumsum(rng.normal(0, 0.05, len(x.close) - k - 1)))
    y = sf.Px("TEST", x.dates, x.high * f, x.low * f, x.close * f, x.open_tr * f)
    for rule in sf.RULES:
        for _, _, p in rule.grid():
            e1, x1 = sf.signals(rule.code, p, x)
            e2, x2 = sf.signals(rule.code, p, y)
            assert np.array_equal(e1[:k + 1], e2[:k + 1]), (rule.code, p)
            assert np.array_equal(x1[:k + 1], x2[:k + 1]), (rule.code, p)


def test_hold_state_semantics() -> None:
    e = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0], dtype=bool)
    x = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0], dtype=bool)
    st = sf.hold_state(e, x, start=0)
    # 0: 들고 있지 않아 매도 신호 무시 · 1: 매수 · 2: 들고 있어 매수 무시, 매도 → 0 · 3: 매도 무시 · 5: 매수(같은 봉 매도 무시)
    # 6: 매수 무시 (매도 없음) · 8: 매도 → 0 · 9: 그대로 0
    assert st.tolist() == [0, 1, 0, 0, 0, 1, 1, 1, 0, 0]
    assert sf.hold_state(e, x, start=3).tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]


def test_sleeve_timing_costs_and_cash() -> None:
    n = 12
    dates = pd.bdate_range("2001-01-01", periods=n)
    otr = 100.0 * 1.01 ** np.arange(n)            # 매일 +1% (시가→시가)
    px = sf.Px("T", dates, otr, otr, otr, otr)
    st = np.zeros(n, dtype=np.int8)
    st[4:7] = 1                                    # 봉 4 종가 매수 신호 → 봉 5 시가 체결 · 봉 7 종가 매도 신호 → 봉 8 시가
    rf = np.full(n, 0.0001)
    s = sf.make_sleeve(px, st, 2, rf, cost=0.001)
    assert s.dates[0] == dates[2] and len(s.pos) == n - 3      # 봉 2 ~ 봉 n−2
    assert s.pos.tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0]
    assert np.isclose(s.ret[0], 0.0001)                          # 쉬는 칸 = 국채
    assert np.isclose(s.ret[3], 0.01 - 0.001)                    # 산 날: 수익 − 편도 비용
    assert np.isclose(s.ret[4], 0.01)
    assert np.isclose(s.ret[6], 0.0001 - 0.001)                  # 판 날: 국채 − 편도 비용
    lag = sf.make_sleeve(px, st, 2, rf, cost=0.001, delay=1)
    assert lag.pos.tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0]       # 하루 늦은 체결
    h = sf.hold_sleeve(px, 2, rf, cost=0.001)
    assert np.isclose(np.prod(1 + h.ret), (1.01 - 0.001) * 1.01 ** (n - 4))
    tr = sf.trade_list(s, 0.001)
    assert len(tr) == 1 and tr["보유일"].iloc[0] == 3 and not tr["열림"].iloc[0]
    assert np.isclose(tr["수익"].iloc[0], 1.01 ** 3 * 0.999 ** 2 - 1)


def test_trade_list_marks_open_trade() -> None:
    n = 8
    dates = pd.bdate_range("2001-01-01", periods=n)
    otr = np.linspace(10, 17, n)
    px = sf.Px("T", dates, otr, otr, otr, otr)
    st = np.zeros(n, dtype=np.int8)
    st[3:] = 1
    s = sf.make_sleeve(px, st, 1, np.zeros(n), cost=0.0)
    tr = sf.trade_list(s, 0.0)
    assert len(tr) == 1 and bool(tr["열림"].iloc[0])
    assert np.isclose(tr["수익"].iloc[0], otr[-1] / otr[4] - 1)  # 마지막 시가로 평가


def test_placebo_zero_shift_equals_real_group() -> None:
    rf = pd.Series(0.0001, index=pd.bdate_range("2000-01-03", periods=900))
    sleeves = []
    for seed in range(3):
        x = _walk(seed=seed)
        e, xx = sf.signals("A2", (20, 2.0), x)
        st = sf.hold_state(e, xx, 281 + 30 * seed - 1)
        sleeves.append(sf.make_sleeve(sf.Px(f"T{seed}", x.dates, x.high, x.low, x.close, x.open_tr),
                                      st, 281 + 30 * seed, rf.to_numpy(), 0.001))
    g = sf.group_returns(sleeves)
    real = sf.sharpe_ex((g - rf.reindex(g.index)).to_numpy())
    plc = sf.placebo_sharpes(sleeves, rf, 0.001, shifts=np.array([0, 100]))
    assert np.isclose(plc[0], real)
    assert not np.isclose(plc[1], real)


def test_boot_dsharpe_identical_and_dominant() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0.0005, 0.01, 3000)
    same = sf.boot_dsharpe(a, a, n_boot=500)
    assert same["dS"] == 0 and same["p"] == 1.0 and same["lo"] == 0 and same["hi"] == 0
    better = sf.boot_dsharpe(a + 0.002, a, n_boot=500)
    assert better["lo"] > 0 and better["p"] == 0.0


def test_control_ranking_rules() -> None:
    dates = pd.bdate_range("2010-01-01", "2010-12-31")
    cols = pd.MultiIndex.from_product([["Close", "Volume"], ["SPY", "GOOG", "GOOGL", "AAA", "NEW"]],
                                      names=["Price", "Ticker"])
    df = pd.DataFrame(1.0, index=dates, columns=cols)
    df[("Volume", "SPY")] = 1e6                     # ETF 는 빠진다
    df[("Volume", "GOOG")] = 9e5                    # 두 번째 종류는 빠진다
    df[("Volume", "GOOGL")] = 5e5
    df[("Volume", "AAA")] = 1e5
    df[("Volume", "NEW")] = 8e5
    df.loc[dates[:-50], ("Close", "NEW")] = np.nan  # 2010 거래일 < 200 → 빠진다
    rank = sf.control_ranking(df)
    assert list(rank.index) == ["GOOGL", "AAA"]


def test_registered_rules_and_groups() -> None:
    assert [r.center for r in sf.RULES] == [(14, 30, 70), (20, 2.0), (25, 2.0), (10, 5), (10, 3.0)]
    assert all(len(r.grid()) == 9 and r.grid()[4][2] == r.center for r in sf.RULES)
    assert {g: len(v) for g, v in sf.GROUPS.items()} == {"G": 10, "D": 10, "C": 10}
    assert sf.N_TESTS == len(sf.RULES) * len(sf.TEST_GROUPS) and np.isclose(sf.P_MAX, 0.005)
