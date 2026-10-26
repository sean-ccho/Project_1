"""my_style_research: 5칸 계좌 시뮬레이터 — 매수·매도 시점, 손절, 비용, 순서, 조건 점검."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import my_style_research as ms  # noqa: E402


def _market(closes: dict[str, list[float]], opens: dict[str, list[float]] | None = None,
            ema: dict[str, list[float]] | None = None, atr: float = 0.02) -> ms.Market:
    idx = pd.bdate_range("2020-01-01", periods=len(next(iter(closes.values()))))
    tk = list(closes)
    c = np.array([closes[t] for t in tk], dtype=float).T
    o = np.array([(opens or closes)[t] for t in tk], dtype=float).T
    e = np.array([(ema or {}).get(t, [0.0] * len(idx)) for t in tk], dtype=float).T
    return ms.Market(idx, tk, o, c, e, np.full_like(c, atr))


def test_buys_next_open_by_score_and_sells_after_hold(monkeypatch) -> None:
    monkeypatch.setattr(ms, "HOLD", {"X1": 3})
    mk = _market({"AAA": [10, 10, 11, 12, 13, 14], "BBB": [20, 20, 20, 20, 20, 20]})
    cands = {0: [(5.0, "AAA"), (9.0, "BBB")]}
    eq, tr = ms.simulate(mk, cands, "X1", start=mk.dates[0], end=mk.dates[-1], capital=1000.0)
    by = {t["ticker"]: t for t in tr}
    # 0일 신호 → 1일 시가 진입, 진입일 포함 3거래일째(3일) 종가로 판단 → 4일 시가 청산
    assert by["AAA"]["entry_date"] == mk.dates[1] and by["AAA"]["exit_date"] == mk.dates[4]
    assert by["AAA"]["reason"] == "3일" and by["AAA"]["exit_price"] == 13
    assert by["AAA"]["return_pct"] == pytest.approx(13 * 0.999 / (10 * 1.001) - 1)
    # 칸 = 자본 ÷ 5 → 두 종목에 각 200 (★ 높은 BBB 먼저)
    assert eq.iloc[0] == pytest.approx(1000.0)
    assert eq.iloc[1] == pytest.approx(600 + 2 * 200 / 1.001)   # 진입일 종가 = 시가라 비용만 빠짐


def test_stop_loss_uses_atr_times_three_and_comes_first(monkeypatch) -> None:
    monkeypatch.setattr(ms, "HOLD", {"X1": 50})
    mk = _market({"AAA": [10, 10, 9.5, 9.3, 9.0, 9.0]}, atr=0.02)   # 손절선 = 10 × (1 − 0.06) = 9.4
    eq, tr = ms.simulate(mk, {0: [(5.0, "AAA")]}, "X1", start=mk.dates[0], end=mk.dates[-1], capital=1000.0)
    assert len(tr) == 1 and tr[0]["reason"] == "손절"
    assert tr[0]["exit_date"] == mk.dates[4] and tr[0]["exit_price"] == 9.0   # 3일 종가 9.3 < 9.4 → 4일 시가


def test_ema_exit_and_end_of_period_close() -> None:
    mk = _market({"AAA": [10, 10, 10.5, 10.2, 10.1, 10.3], "BBB": [5, 5, 5.2, 5.4, 5.5, 5.6]},
                 ema={"AAA": [9, 9, 9.5, 10.4, 10.0, 10.0], "BBB": [4, 4, 4, 4, 4, 4]})
    eq, tr = ms.simulate(mk, {0: [(5.0, "AAA"), (4.0, "BBB")]}, "X3", start=mk.dates[0], end=mk.dates[-1],
                         capital=1000.0)
    by = {t["ticker"]: t for t in tr}
    assert by["AAA"]["reason"] == "EMA50 이탈" and by["AAA"]["exit_date"] == mk.dates[4]   # 3일 10.2 < 10.4
    assert by["BBB"]["reason"] == "기간 끝" and by["BBB"]["exit_price"] == 5.6


def test_slots_cap_skips_held_and_same_issuer() -> None:
    names = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GOOG", "GOOGL"]
    mk = _market({t: [10.0] * 4 for t in names})
    cands = {0: [(9.0, "GOOGL"), (8.5, "GOOG"), *[(float(i), t) for i, t in enumerate(names[:6])]]}
    _, tr = ms.simulate(mk, cands, "X1", start=mk.dates[0], end=mk.dates[-1], capital=1000.0)
    bought = [t["ticker"] for t in tr]
    assert len(bought) == ms.SLOTS and "GOOG" not in bought                  # 같은 회사 두 클래스는 하나만
    assert bought[0] == "GOOGL" and set(bought[1:]) == {"FFF", "EEE", "DDD", "CCC"}   # ★ 높은 순


def test_eligible_revenue_and_insider() -> None:
    pool = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"] * 4), "ticker": ["A", "B", "C", "D"],
                         "rev_yoy": [0.1, -0.05, np.nan, 0.2], "buyers": [0, 3, 2, 1], "score": [1, 2, 3, 4]})
    assert ms.eligible(pool, "S1")["ticker"].tolist() == ["A", "D"]          # 매출 모르면 불통과
    assert ms.eligible(pool, "S2")["ticker"].tolist() == ["D"]
    with pytest.raises(ValueError):
        ms.eligible(pool, "S9")


def test_atr_pct_simple_true_range_mean() -> None:
    idx = pd.bdate_range("2020-01-01", periods=3)
    h = pd.DataFrame({"A": [11.0, 12.0, 12.0]}, index=idx)
    lo = pd.DataFrame({"A": [9.0, 10.0, 11.0]}, index=idx)
    c = pd.DataFrame({"A": [10.0, 11.0, 11.5]}, index=idx)
    a = ms.atr_pct(h, lo, c, n=2)["A"]
    # TR: 2(첫날 범위), max(2, |12−10|, |10−10|)=2, max(1, |12−11|, |11−11|)=1 → 2일 평균 2.0 · 1.5
    assert np.isnan(a.iloc[0]) and a.iloc[1] == pytest.approx(2.0 / 11.0) and a.iloc[2] == pytest.approx(1.5 / 11.5)


def test_exit_is_booked_on_the_sell_day_not_the_decision_day(monkeypatch) -> None:
    monkeypatch.setattr(ms, "HOLD", {"X1": 2})
    # 2일 종가 12 로 판단 → 3일 시가 15 에 판다. 2일 자본은 12 로, 3일 자본은 15 로 잡혀야 한다 (시가를 하루 앞당기지 않음)
    mk = _market({"AAA": [10, 10, 12, 15, 15]}, opens={"AAA": [10, 10, 12, 15, 15]})
    eq, tr = ms.simulate(mk, {0: [(5.0, "AAA")]}, "X1", start=mk.dates[0], end=mk.dates[-1], capital=1000.0)
    sh = 200 / (10 * 1.001)
    assert eq.iloc[2] == pytest.approx(800 + sh * 12)
    assert eq.iloc[3] == pytest.approx(800 + sh * 15 * 0.999)
    assert tr[0]["exit_date"] == mk.dates[3]


def test_x3b_waits_until_price_was_above_ema() -> None:
    # 진입 때 이미 EMA 아래 → X3 는 바로 팔지만 X3b 는 EMA 위로 한 번 올라선 뒤 내려올 때 판다
    mk = _market({"AAA": [10, 10, 9.9, 10.6, 10.1, 10.1, 10.1]},
                 ema={"AAA": [10.5, 10.5, 10.4, 10.4, 10.3, 10.3, 10.3]})
    _, tr3 = ms.simulate(mk, {0: [(5.0, "AAA")]}, "X3", start=mk.dates[0], end=mk.dates[-1], capital=1000.0)
    _, tr3b = ms.simulate(mk, {0: [(5.0, "AAA")]}, "X3b", start=mk.dates[0], end=mk.dates[-1], capital=1000.0)
    assert tr3[0]["exit_date"] == mk.dates[2]                       # 1일 종가 10 < 10.5 → 2일 시가
    assert tr3b[0]["reason"] == "EMA50 이탈" and tr3b[0]["exit_date"] == mk.dates[5]   # 3일 위 → 4일 아래 → 5일 시가
