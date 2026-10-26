"""my_picks_research: 골든 봉 읽기 · 규칙 · 고르는 순서 · 제외 · 이벤트 행렬 점검."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import my_picks_research as mp  # noqa: E402


def _df(rows: list[dict]) -> pd.DataFrame:
    base = {"일봉패턴": "", "주봉패턴": "", "월봉패턴": "", "20일수익률": 0.05, "매수적합도": 5.0, "전략구분": "📈 모멘텀",
            "매수적합도_표시": "★★★ (5.0)", "섹터": "Technology", "close": 100.0, "회사": ""}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_gold_tfs_reads_golden_and_imminent_tokens_only() -> None:
    row = {"일봉패턴": "이중바닥, 골든크로스임박", "주봉패턴": "정배열(형성중)", "월봉패턴": "골든크로스, 차트반전"}
    assert mp.gold_tfs(row) == "DM"                       # 정배열(형성중)은 골든 라벨이 아니다
    assert mp.gold_tfs({"일봉패턴": "데드크로스임박", "주봉패턴": np.nan, "월봉패턴": None}) == ""
    assert mp.gold_tfs({"일봉패턴": "골든크로스임박도지"}) == ""   # 토큰 단위로만 (부분 문자열 아님)


def test_eligible_rules() -> None:
    assert mp.eligible("MY1", "D", 0.01)
    assert not mp.eligible("MY1", "", 0.10)
    assert not mp.eligible("MY1", "DWM", 0.0)             # 20일 수익 0 은 안 됨
    assert not mp.eligible("MY1", "DWM", float("nan"))
    assert not mp.eligible("MY2", "D", 0.10)              # 일봉만 골든이면 MY2 아님
    assert mp.eligible("MY2", "M", 0.10) and mp.eligible("MY2", "DW", 0.10)
    with pytest.raises(ValueError):
        mp.eligible("MY9", "D", 0.1)


def test_pick_orders_by_score_and_excludes_index_held_and_same_issuer() -> None:
    df = _df([
        {"티커": "SPY", "일봉패턴": "골든크로스", "매수적합도": 99.0},               # 지수 ETF
        {"티커": "AAA", "일봉패턴": "골든크로스", "매수적합도": 9.0},
        {"티커": "BBB", "주봉패턴": "골든크로스임박", "매수적합도": 12.0},           # 보유 중
        {"티커": "GOOG", "월봉패턴": "골든크로스", "매수적합도": 11.0},              # GOOGL 보유 → 같은 회사
        {"티커": "CCC", "일봉패턴": "골든크로스", "매수적합도": 10.0, "20일수익률": -0.02},  # 20일 하락
        {"티커": "DDD", "일봉패턴": "이중바닥", "매수적합도": 20.0},                 # 골든 없음
        {"티커": "EEE", "월봉패턴": "골든크로스", "매수적합도": 9.0},                # AAA 와 동점 → 티커순
    ])
    held = [{"ticker": "BBB"}, {"ticker": "GOOGL"}]
    picks = mp.pick(df, held, "MY1", k=3)
    assert [p["ticker"] for p in picks] == ["AAA", "EEE"]
    assert picks[0]["strategy"] == "📈 모멘텀" and picks[0]["ccs_score"] == 0.0 and picks[0]["entry_price"] == 100.0
    assert [p["ticker"] for p in mp.pick(df, held, "MY2", k=3)] == ["EEE"]


def test_pick_seed_order_is_deterministic_and_differs_from_score_order() -> None:
    df = _df([{"티커": f"T{i:02d}", "일봉패턴": "골든크로스", "매수적합도": float(i)} for i in range(30)])
    day = pd.Timestamp("2020-01-02")
    a = [p["ticker"] for p in mp.pick(df, [], "MY1", k=5, seed=3, day=day)]
    b = [p["ticker"] for p in mp.pick(df, [], "MY1", k=5, seed=3, day=day)]
    assert a == b
    assert a != ["T29", "T28", "T27", "T26", "T25"]
    assert [p["ticker"] for p in mp.pick(df, [], "MY1", k=2)] == ["T29", "T28"]


def test_pick_skips_second_share_class_on_same_day_and_missing_strategy_defaults() -> None:
    df = _df([
        {"티커": "GOOGL", "일봉패턴": "골든크로스", "매수적합도": 9.0, "전략구분": ""},
        {"티커": "GOOG", "일봉패턴": "골든크로스", "매수적합도": 8.0},
        {"티커": "ZZZ", "일봉패턴": "골든크로스", "매수적합도": 1.0},
    ])
    picks = mp.pick(df, [], "MY1", k=2)
    assert [p["ticker"] for p in picks] == ["GOOGL", "ZZZ"]
    assert picks[0]["strategy"] == "골든크로스"


def test_qualify_matrix_and_new_entries() -> None:
    days = pd.bdate_range("2021-03-01", periods=5)
    lists = pd.DataFrame({
        "date": [days[0], days[1], days[3], days[1], days[2]],
        "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB"],
        "tfs": ["D", "D", "W", "D", "D"],
        "ret20": [0.1, 0.1, 0.1, 0.1, -0.1],
    })
    cols = pd.Index(["AAA", "BBB"])
    q1 = mp.qualify_matrix(lists, "MY1", days, cols)
    assert q1["AAA"].tolist() == [True, True, False, True, False]
    assert q1["BBB"].tolist() == [False, True, False, False, False]   # 20일 하락인 날은 빠짐
    ent = mp.new_entries(q1)
    assert list(zip(ent["signal_date"], ent["티커"])) == [(days[0], "AAA"), (days[1], "BBB"), (days[3], "AAA")]
    assert ent["entry_pos"].tolist() == [1, 2, 4]
    q2 = mp.qualify_matrix(lists, "MY2", days, cols)
    assert q2.to_numpy().sum() == 1 and bool(q2.at[days[3], "AAA"])


def test_luck_table_binomial() -> None:
    idx = pd.bdate_range("2016-01-04", periods=4)
    cols = pd.Index([f"S{i}" for i in range(4)])
    # 첫날 → 다음날: 4종목 중 3개 상승 (p = 0.75), 둘째날 → 셋째날: 모두 하락 (p = 0)
    closes = pd.DataFrame([[10, 10, 10, 10], [11, 11, 11, 9], [10, 10, 10, 8], [10, 10, 10, 8]],
                          index=idx, columns=cols, dtype=float)
    member = pd.DataFrame(True, index=idx, columns=cols)
    spy = pd.Series([100, 101, 99, 99], index=idx, dtype=float)
    out = mp.luck_table(closes, member, spy, horizons=(1,)).iloc[0]
    p_first = 5 * 0.75 ** 4 * 0.25 + 0.75 ** 5
    assert out["날짜 수"] == 3
    assert out["SPY 오른 구간"] == pytest.approx(p_first)
    assert out["전체"] == pytest.approx(p_first / 3)


def test_sleeve_returns_timing_weights_and_costs() -> None:
    idx = pd.bdate_range("2021-03-01", periods=5)
    r_oo = pd.DataFrame({"AAA": [0.10, 0.20, 0.30, 0.40, np.nan]}, index=idx)
    spy = pd.Series(0.0, index=idx)
    picks = pd.DataFrame({"date": [idx[0]], "ticker": ["AAA"], "w": [1.0]})
    r = mp.sleeve_returns(picks, r_oo, spy, hold=2, cost=0.001)
    # 다음날(1) 시가에 사서 2거래일: 칸 비중 1/2 → 1·2일째 수익의 절반, 사고판 날 비용 (종목 0.5 + SPY 0.5) × 0.1%
    assert r.iloc[1] == pytest.approx(0.5 * 0.20 - 0.001)
    assert r.iloc[2] == pytest.approx(0.5 * 0.30)
    assert r.iloc[3] == pytest.approx(-0.001)
    assert r.iloc[4] == pytest.approx(0.0)


def test_daily_picks_top1_pool_and_rand() -> None:
    d0, d1 = pd.Timestamp("2021-03-01"), pd.Timestamp("2021-03-02")
    lists = pd.DataFrame({
        "date": [d0, d0, d0, d1, d1],
        "ticker": ["AAA", "BBB", "CCC", "AAA", "DDD"],
        "tfs": ["W", "M", "D", "WM", "W"],                 # CCC 는 일봉만 → MY2 아님
        "ret20": [0.1, 0.1, 0.1, 0.1, 0.1],
        "score": [5.0, 7.0, 9.0, 3.0, 3.0],
    })
    top = mp.daily_picks(lists, "MY2", "top1")
    assert list(zip(top["date"], top["ticker"])) == [(d0, "BBB"), (d1, "AAA")]   # 동점은 티커순
    pool = mp.daily_picks(lists, "MY2", "pool")
    assert pool.groupby("date")["w"].sum().tolist() == [1.0, 1.0] and "CCC" not in set(pool["ticker"])
    r1, r2 = mp.daily_picks(lists, "MY2", "rand", seed=4), mp.daily_picks(lists, "MY2", "rand", seed=4)
    assert r1["ticker"].tolist() == r2["ticker"].tolist() and len(r1) == 2
