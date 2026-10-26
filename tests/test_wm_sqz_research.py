"""wm_sqz_research: 주봉 신호 계좌 — 매수·매도 시점, 주봉 매도, 손절, 교체, 배당 포함 평가 점검."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import wm_sqz_research as ws  # noqa: E402


def _mkt(n: int, tickers: list[str], close: dict[str, list[float]] | None = None, factor: float = 1.0,
         atr: float = 0.05) -> ws.Mkt:
    idx = pd.bdate_range("2022-01-03", periods=n)
    c = np.array([(close or {}).get(t, [10.0] * n) for t in tickers], dtype=float).T
    return ws.Mkt(idx, tickers, c.copy(), c.copy(), c * factor, np.full_like(c, factor), np.full_like(c, atr),
                  np.full_like(c, 1e8))


def _grid(mk: ws.Mkt) -> ws.Grid:
    shape = (len(mk.dates), len(mk.tickers))
    return ws.Grid(*(np.zeros(shape, dtype=bool) for _ in range(6)), np.zeros(shape, dtype=np.int16),
                   np.zeros(shape, dtype=np.float32))


def _run(mk, g, cand, **kw):
    return ws.simulate(mk, g, cand, start=mk.dates[0], end=mk.dates[-1], capital=1000.0, **kw)


def test_buy_next_open_and_sell_after_two_weak_weeks() -> None:
    mk = _mkt(12, ["AAA"], close={"AAA": [10, 10, 10, 11, 12, 12, 12, 12, 12, 13, 13, 13]})
    g = _grid(mk)
    g.w_golden[:] = True
    for d in (4, 9):                       # 주봉 마지막 날 두 번, 둘 다 막대 약해짐
        g.week_end[d, 0] = True
        g.falling[d, 0] = True
    cand = np.zeros_like(g.fire)
    cand[2, 0] = True
    eq, tr = _run(mk, g, cand)
    assert tr[0]["entry_date"] == mk.dates[3]                       # 신호 다음 날 시가
    assert tr[0]["reason"] == "모멘텀 2주 약해짐" and tr[0]["exit_date"] == mk.dates[10]
    assert tr[0]["return_pct"] == pytest.approx(13 * 0.999 / (11 * 1.001) - 1)
    assert eq.iloc[3] == pytest.approx(800 + 200 / 1.001)            # 진입일 종가 = 시가 → 비용만 빠짐


def test_weekly_dead_cross_and_daily_stop() -> None:
    mk = _mkt(10, ["AAA", "BBB"], close={"AAA": [10] * 10, "BBB": [10, 10, 10, 10, 9.2, 8.9, 8.9, 8.9, 8.9, 8.9]})
    g = _grid(mk)
    g.w_golden[:, 1] = True
    g.week_end[5, 0] = True                # AAA: 주봉 EMA10 < EMA20 (w_golden False) → 다음 날 매도
    cand = np.zeros_like(g.fire)
    cand[1, [0, 1]] = True
    _, tr = _run(mk, g, cand)
    by = {t["ticker"]: t for t in tr}
    assert by["AAA"]["reason"] == "주봉 데드크로스" and by["AAA"]["exit_date"] == mk.dates[6]
    # BBB: 손절선 = 10 × (1 − 2 × 5%) = 9.0 → 5일 종가 8.9 < 9.0 → 6일 시가
    assert by["BBB"]["reason"] == "손절" and by["BBB"]["exit_date"] == mk.dates[6]


def test_replacement_swaps_weakest_weakening_holding() -> None:
    names = ["A1", "A2", "A3", "A4", "A5", "NEW"]
    closes = {t: [10.0] * 10 for t in names}
    closes["A3"] = [10, 10, 10, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6]   # 가장 약함 (손절선 9.0 위)
    mk = _mkt(10, names, close=closes)
    g = _grid(mk)
    g.w_golden[:] = True
    g.week_end[4, :5] = True
    g.falling[4, [2, 3]] = True             # A3·A4 가 약해짐 → 그중 수익 낮은 A3 교체
    cand = np.zeros_like(g.fire)
    cand[1, :5] = True
    cand[5, 5] = True
    _, tr = _run(mk, g, cand)
    sold = [t for t in tr if t["reason"] == "교체"]
    assert len(sold) == 1 and sold[0]["ticker"] == "A3" and sold[0]["exit_date"] == mk.dates[6]
    assert any(t["ticker"] == "NEW" and t["entry_date"] == mk.dates[6] for t in tr)


def test_no_replacement_when_nothing_weak_and_full() -> None:
    names = ["A1", "A2", "A3", "A4", "A5", "NEW"]
    mk = _mkt(8, names)
    g = _grid(mk)
    g.w_golden[:] = True
    cand = np.zeros_like(g.fire)
    cand[1, :5] = True
    cand[4, 5] = True
    _, tr = _run(mk, g, cand)
    assert "NEW" not in {t["ticker"] for t in tr}


def test_dividend_factor_counts_in_value_and_trade_return() -> None:
    mk = _mkt(6, ["AAA"])
    mk.factor[3:, 0] = 1.02                 # 3일째부터 배당 반영 비율이 2% 커짐 (배당 재투자 효과)
    mk.adj[:, 0] = mk.close[:, 0] * mk.factor[:, 0]
    g = _grid(mk)
    cand = np.zeros_like(g.fire)
    cand[0, 0] = True
    eq, tr = _run(mk, g, cand)
    assert tr[0]["reason"] == "기간 끝"
    assert tr[0]["return_pct"] == pytest.approx(1.02 * 0.999 / 1.001 - 1)
    assert eq.iloc[-1] == pytest.approx(800 + 200 / 1.001 * 1.02)


def test_entry_mask_kinds() -> None:
    mk = _mkt(3, ["AAA", "BBB", "CCC"])
    g = _grid(mk)
    g.week_end[1, :] = True
    g.fire[1, [0, 1]] = True
    g.w_golden[1, [0, 2]] = True
    g.m_golden[1, :] = True
    g.w_gc[1, 2] = True
    uni = np.ones_like(g.fire)
    assert np.flatnonzero(ws.entry_mask(g, mk, uni, "main")[1]).tolist() == [0]
    assert np.flatnonzero(ws.entry_mask(g, mk, uni, "fire")[1]).tolist() == [0, 1]
    assert np.flatnonzero(ws.entry_mask(g, mk, uni, "gold")[1]).tolist() == [2]
    mk.close[1, 0] = 4.0                    # $5 아래 → 빠짐
    assert ws.entry_mask(g, mk, uni, "main")[1].sum() == 0
