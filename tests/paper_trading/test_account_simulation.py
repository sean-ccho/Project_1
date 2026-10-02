"""PT-2/PT-3 전략 + 계좌 엔진 통합 스모크 테스트 (합성 시세, pandas 불필요).

수익성은 보지 않는다. 수백 일을 돌려도 예외가 없고 장부가 맞는지만 확인한다.
"""

import random
from datetime import date, timedelta

from paper_trading.account_engine import AccountState, BarSeries, process_bar
from paper_trading.accounts import get_profile
from paper_trading.indicators import atr_series, compute_indicators, ema_series

N_DAYS = 420
WARMUP = 220


def _make_market(seed: int = 7) -> dict[str, BarSeries]:
    rng = random.Random(seed)
    start = date(2024, 1, 1)
    dates = [str(start + timedelta(days=i)) for i in range(N_DAYS)]
    market = {}
    for t, drift in [("SPY", 0.0004)] + [(f"T{i:02d}", rng.uniform(-0.001, 0.002)) for i in range(12)]:
        price, bars = 50.0 + rng.random() * 100, []
        for d in dates:
            o = price * (1 + rng.gauss(0, 0.004))
            c = o * (1 + drift + rng.gauss(0, 0.02))
            h, lo = max(o, c) * (1 + abs(rng.gauss(0, 0.006))), min(o, c) * (1 - abs(rng.gauss(0, 0.006)))
            bars.append({"date": d, "open": o, "high": h, "low": lo, "close": c, "volume": rng.uniform(5e5, 3e6)})
            price = c
        market[t] = BarSeries(bars)
    return market


def _row(ticker: str, bars: list[dict], sector: str) -> dict:
    closes = [b["close"] for b in bars]
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    ind = compute_indicators(bars)
    e20, e50, e200 = ema_series(closes, 20), ema_series(closes, 50), ema_series(closes, 200)
    atr = atr_series(highs, lows, closes, 14)
    atr_pct = [a / c for a, c in zip(atr[-252:], closes[-252:]) if a == a]
    crossed = any(e20[-k - 1] < e50[-k - 1] and e20[-k] >= e50[-k] for k in range(1, 6))
    gap = e20[-1] / e50[-1] - 1
    vols = [b["volume"] for b in bars[-20:]]
    lo52, hi52 = min(closes[-252:]), max(closes[-252:])
    return {
        "티커": ticker, "섹터": sector, "close": closes[-1], "현재가격": closes[-1],
        "ema20": e20[-1], "ema50": e50[-1], "ema200": e200[-1], "atr_value": atr[-1],
        "최근20일평균거래대금": 50e6, "RSI": ind["rsi"], "bollinger_pband": ind["bb_pband"], "adx": ind["adx"],
        "5일수익률": closes[-1] / closes[-6] - 1, "20일수익률": ind["ret_20d"],
        "52주포지션": (closes[-1] - lo52) / (hi52 - lo52) if hi52 > lo52 else 0.5,
        "ema_gap_50_200": e50[-1] / e200[-1] - 1, "ema_gap_20_50": gap,
        "거래량돌파배수": bars[-1]["volume"] / (sum(vols) / len(vols)),
        "변동성압축": atr_pct[-1] / sorted(atr_pct)[len(atr_pct) // 2],
        "일봉패턴": "골든크로스" if crossed and 0 <= gap <= 0.05 else "",
        "주봉패턴": "", "월봉패턴": "", "주봉_MA갭": 0.0, "월봉_MA갭": 0.0,
        "저점반전캔들": 0, "매수적합도_표시": "★★★",
    }


def _simulate(key: str) -> tuple[AccountState, list[dict]]:
    market = _make_market()
    profile = get_profile(key)
    state = AccountState.new(float(profile.params["initial_capital"]))
    sectors = ["Tech", "Health", "Energy", "Financials"]
    results = []
    dates = market["SPY"].dates
    for i in range(WARMUP, N_DAYS):
        d = dates[i]
        rows = [_row(t, s.upto(d, 260), sectors[n % 4]) for n, (t, s) in enumerate(market.items())]
        results.append(process_bar(state, profile, d, market, rows=rows))
        assert state.cash > -1e-6
        assert len(state.positions) <= profile.params["max_positions"]
        assert len({p["ticker"] for p in state.positions}) == len(state.positions)
    return state, results


def _check_ledger(state: AccountState, key: str) -> None:
    cost = get_profile(key).params["cost_per_side"]
    invested = sum(p["shares"] * p["entry_price"] * (1 + cost) for p in state.positions)
    realized = sum(t["dollar_pnl"] for t in state.trades)
    # 현금 + 보유 원가 = 초기자본 + 실현손익 (반올림 오차 허용)
    assert abs(state.cash + invested - (state.initial_capital + realized)) < 0.5
    for t in state.trades:
        assert t["exit_date"] >= t["entry_date"]
        assert t["exit_reason"]


def test_pt2_simulation_runs_and_ledger_balances():
    state, results = _simulate("pt2")
    assert len(state.equity_history) == N_DAYS - WARMUP
    _check_ledger(state, "pt2")
    assert any(r["orders"] for r in results)


def test_pt3_simulation_runs_and_ledger_balances():
    state, results = _simulate("pt3")
    assert len(state.equity_history) == N_DAYS - WARMUP
    _check_ledger(state, "pt3")
    for t in state.trades:
        assert t["holding_bars"] <= get_profile("pt3").params["max_hold_bars"]
