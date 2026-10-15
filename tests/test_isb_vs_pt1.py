"""scripts/isb_vs_pt1.py 테스트 — 매수 게이트 순서, 트림·매도 결정, 달력시간 수익, 판정 (백테스트 없이)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import isb_vs_pt1 as m  # noqa: E402
from paper_trading.backtest import BtPosition  # noqa: E402


def _s(t: str, ccs: float) -> dict:
    return {"ticker": t, "ccs": ccs, "idx": t}


def test_pick_gated_order_and_counts():
    first = {"ticker": "B"}
    top5 = [_s("A", 0.9), _s("B", 0.8), _s("C", 0.7), _s("D", 0.6), _s("E", 0.5), _s("F", 0.4)]
    item, n_eval, n_pass = m.pick_gated(first, top5, lambda t: t in {"B"})
    assert item is first and (n_eval, n_pass) == (1, 1)              # PT-1 선택이 먼저
    item, n_eval, n_pass = m.pick_gated(first, top5, lambda t: t in {"C", "A"})
    assert item["ticker"] == "A" and (n_eval, n_pass) == (2, 1)      # 그다음 CCS 순 (B 는 한 번만)
    item, n_eval, n_pass = m.pick_gated(first, top5, lambda t: t == "F")
    assert item is None and (n_eval, n_pass) == (5, 0)               # top5 밖(F)은 안 봄 → 그날 안 삼


def test_trim_allowed_matches_isb_cantrim():
    assert m.trim_allowed(0, 100.0, 100.0)
    assert m.trim_allowed(1, 67.0, 100.0)
    assert not m.trim_allowed(2, 34.0, 100.0)       # 최대 2회
    assert not m.trim_allowed(1, 40.0, 100.0)       # 트림 뒤 15% 미만


def test_apply_trim_keeps_position_and_returns_proceeds():
    ctx = m.RunCtx(m.Overlay("X"), None)
    pos = BtPosition(ticker="AAA", entry_price=100.0, entry_date="2020-01-02", strategy="s", star_rating="",
                     ccs_score=0.5, sector="", shares=10.0)
    got = m.apply_trim(ctx, pos, 110.0, "2020-02-03", "TOP")
    assert got == pytest.approx(3.3 * 110.0 * 0.999) and pos.shares == pytest.approx(6.7)
    assert m.apply_trim(ctx, pos, 120.0, "2020-03-02", "T1") == pytest.approx(3.3 * 120.0 * 0.999)
    assert pos.shares == pytest.approx(3.4)
    assert m.apply_trim(ctx, pos, 130.0, "2020-04-01", "TOP") == 0.0 and pos.shares == pytest.approx(3.4)
    assert [t["kind"] for t in ctx.trims] == ["TOP", "T1"]


def _boom():
    raise AssertionError("PT-1 규칙을 부르면 안 된다")


def test_sell_decision_modes_and_priority():
    pt1_sell = lambda: (True, "손절(-10.0%)")  # noqa: E731
    pt1_hold = lambda: (False, "")  # noqa: E731
    # full (isb): 상태 기반, PT-1 규칙 안 씀
    assert m.sell_decision("isb", _boom, hold_next=False, exit_full=False, trim=True, trim_ok=True)[0]
    assert m.sell_decision("isb", _boom, True, False, True, True) == (True, "ISB트림·TOP")
    assert m.sell_decision("isb", _boom, True, True, False, True) == (False, "")
    # exit 추가 (both): ISB 전량 > PT-1 > ISB 트림
    ok, why = m.sell_decision("both", _boom, True, True, True, True)
    assert ok and why.startswith("ISB매도")
    assert m.sell_decision("both", pt1_sell, True, False, True, True) == (True, "손절(-10.0%)")
    assert m.sell_decision("both", pt1_hold, True, False, True, True) == (True, "ISB트림·TOP")
    assert m.sell_decision("both", pt1_hold, True, False, True, False) == (False, "")
    assert m.sell_decision("pt1", pt1_sell, False, True, True, True) == (True, "손절(-10.0%)")
    # ISB 이유는 익절 보류(목표가·시간익절)를 타지 않는다
    for reason in ("ISB매도(보유 끝)", "ISB매도(EXIT·BREAKDOWN)", "ISB트림·TOP", "ISB트림·T1"):
        assert not reason.startswith(("목표가", "시간익절"))


def test_weighted_returns_with_turnover_cost():
    idx = pd.date_range("2020-01-01", periods=2)
    w = pd.DataFrame({"A": [0.5, 0.5], "B": [0.0, 0.5]}, index=idx)
    r = pd.DataFrame({"A": [0.1, 0.0], "B": [0.2, -0.1]}, index=idx)
    got = m.weighted_returns(w, r, cost=0.001)
    assert list(got) == pytest.approx([0.05 - 0.0005, -0.05 - 0.0005])


class _Mem:
    def __init__(self, names):
        self.names = frozenset(names)

    def on(self, d):
        return self.names


def test_solo_returns_isb_vs_equal_weight():
    idx = pd.date_range("2016-01-04", periods=4, freq="B")
    cols = pd.MultiIndex.from_product([["AAA", "BBB", "SPY"], ["Open", "High", "Low", "Close", "Volume"]],
                                      names=["Ticker", "Price"])
    raw = pd.DataFrame(1.0, index=idx, columns=cols)
    raw[("AAA", "Open")] = [10.0, 11.0, 12.1, 12.1]
    raw[("BBB", "Open")] = [20.0, 20.0, 18.0, 18.0]
    for t in ("AAA", "BBB"):
        raw[(t, "High")], raw[(t, "Low")], raw[(t, "Volume")] = 30.0, 5.0, 1e6
    ev = pd.DataFrame({"date": list(idx) * 2, "ticker": ["AAA"] * 4 + ["BBB"] * 4,
                       "pos": [1, 1, 1, 1, 0, 0, 0, 0], "rem": [1.0, 1.0, 0.67, 0.67, 1.0, 1.0, 1.0, 1.0]})
    r_isb, r_ew, wsum = m.solo_returns(ev, raw, _Mem({"AAA", "BBB"}))
    # 마지막 날(다음 시가 없음) 제외 → 3일. AAA 수익 +10%, +10%, 0 / BBB 0, −10%, 0
    assert len(r_isb) == 3
    assert r_ew.iloc[1] == pytest.approx(0.5 * 0.10 + 0.5 * -0.10)
    assert r_isb.iloc[1] == pytest.approx(0.5 * 0.10)                       # AAA 만 1/N 비중
    assert wsum.iloc[2] == pytest.approx(0.5 * 0.67)                        # 트림 뒤 남은 비율
    assert r_isb.iloc[0] == pytest.approx(0.5 * 0.10 - 0.5 * 0.001)         # 첫날 매수 비용


def test_judge_needs_verdict_and_bonferroni_p():
    assert m.P_BONF == pytest.approx(0.05 / 8)
    st = {"ΔSharpe": 0.3, "ΔSharpe_하한": 0.05, "ΔSharpe_상한": 0.5, "p_개선아님": 0.004, "구간_개선": 3, "구간_수": 3}
    assert m.judge(st, "채택 후보") == "채택 후보"
    assert m.judge({**st, "p_개선아님": 0.01}, "채택 후보") == "관찰"
    assert m.judge({**st, "ΔSharpe_하한": -0.1}, "운과 구분 안 됨") == "탈락"
    assert m.judge({}, "") == "탈락"


def test_overlay_t1_only_for_weekly_isb_sells():
    specs = m.overlay_specs()
    assert specs["W-full"].t1 and specs["W-exit"].t1
    assert not specs["W-entry"].t1 and not specs["D-full"].t1 and not specs["PROBE"].t1
    assert all(o.t1 for o in m.solo3_specs() if o.tf == "W")
    assert set(m.TESTS) == {"D-full", "W-full", "D-entry", "W-entry", "D-exit", "W-exit", "D-solo", "W-solo"}


def test_stable_uniform_is_deterministic():
    a = m.stable_uniform(3, "2020-01-02", "AAPL")
    assert a == m.stable_uniform(3, "2020-01-02", "AAPL") and 0 <= a < 1
    assert a != m.stable_uniform(4, "2020-01-02", "AAPL")


def test_solo_select_ranks_grade_then_seed():
    class Book:
        def get(self, name, tk, day, default):
            table = {"buy_ev": {"AAA": True, "BBB": True, "CCC": True, "SPY": True},
                     "grade": {"AAA": 2, "BBB": 4, "CCC": 4}}
            return table.get(name, {}).get(tk, default)
    ctx = m.RunCtx(m.Overlay("x", tf="D", selector="solo", seed=1), Book(), date=pd.Timestamp("2020-01-02"))
    df = pd.DataFrame({"티커": ["AAA", "BBB", "CCC", "SPY"], "close": [1.0, 2.0, 3.0, 4.0], "섹터": ["a", "b", "c", ""],
                       "현재가격": [1.0, 2.0, 3.0, 4.0], "ema50": [1.0] * 4, "ema200": [1.0] * 4})
    picks, debug = m._solo_select(ctx, df, holdings=[{"ticker": "CCC"}], k=1)
    assert [p["ticker"] for p in picks] == ["BBB"] and picks[0]["ccs_score"] == 4.0   # 보유 중 CCC · SPY 제외
    assert "regime" in debug
    picks, _ = m._solo_select(ctx, df, holdings=[], k=1)
    assert picks[0]["ticker"] in {"BBB", "CCC"}                                        # 등급 4 동점 → 시드
