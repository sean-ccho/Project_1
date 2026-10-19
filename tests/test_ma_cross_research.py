"""ma_cross_research · pt1_cross 의 작은 함수들: 쿨다운, 상태 비중(미래 미사용), 패턴 문자열 토큰 교체."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from ma_cross_research import _cooldown, _gc_describe, state_weights
from pt1_cross import rewrite


def test_cooldown_keeps_first_in_window() -> None:
    pos = np.array([0, 3, 9, 10, 25, 30])
    assert _cooldown(pos, 10).tolist() == [True, False, False, True, True, False]  # 0 · 10 · 25 만 (30 은 25 와 5봉)


def test_state_weights_use_previous_bar_only() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    cols = pd.Index(["A", "B"])
    # 주봉 상태: 1/5(금) 봉에서 A 골든 → 다음 거래일(1/8) 시가부터 보유
    state = pd.DataFrame({"A": [True], "B": [False]}, index=[pd.Timestamp("2024-01-05")])
    mem = pd.DataFrame(True, index=idx, columns=cols)
    tradable = pd.DataFrame(True, index=idx, columns=cols)
    w = state_weights(state, idx, cols, mem, tradable)
    assert w.loc[:"2024-01-05"].sum().sum() == 0.0
    assert w.loc["2024-01-08", "A"] == 1.0 and w.loc["2024-01-08", "B"] == 0.0


def test_state_weights_respect_membership() -> None:
    idx = pd.bdate_range("2024-01-01", periods=4)
    cols = pd.Index(["A", "B"])
    state = pd.DataFrame({"A": [True] * 4, "B": [True] * 4}, index=idx)
    mem = pd.DataFrame({"A": [True] * 4, "B": [True, True, False, False]}, index=idx)
    w = state_weights(state, idx, cols, mem, pd.DataFrame(True, index=idx, columns=cols))
    assert w.loc[idx[1]].tolist() == [0.5, 0.5]
    assert w.loc[idx[2]].tolist() == [1.0, 0.0]


def test_gc_describe_whipsaw_and_lag() -> None:
    idx = pd.bdate_range("2024-01-01", periods=8)
    closes = np.array([10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 10.5, 10.0])
    f = pd.DataFrame({"close": closes}, index=idx)
    pos = np.arange(8)
    dc_pos = np.array([0, 6])
    d = _gc_describe(f, 4, pos, dc_pos, closes, "D")
    assert d["lag_bars"] == 2            # 최저 8.0 (봉 2) → 교차 봉 4
    assert d["missed"] == np.float64(10.0 / 8.0 - 1)
    assert d["whipsaw"]                  # 봉 6 데드크로스 = 2봉 뒤 (일봉 10봉 안)


def test_rewrite_tokens() -> None:
    assert rewrite("골든크로스임박, 강세잉걸핑", None, "D") == "강세잉걸핑"
    assert rewrite("이중바닥", "골든크로스", "W") == "골든크로스, 이중바닥"
    assert rewrite("정배열(완전)", "골든크로스", "W") == "정배열(완전)"     # 앱 규칙: 이미 정배열이면 안 붙임
    assert rewrite("정배열(완전)", "골든크로스", "D") == "골든크로스, 정배열(완전)"
    assert rewrite(np.nan, "골든크로스임박", "M") == "골든크로스임박"
    assert rewrite("데드크로스", None, "D") == "데드크로스"
