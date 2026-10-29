"""rotation_research: Sahm 규칙 · 모멘텀 비중(미래 정보 없음) · 월간 매매 회계 점검."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rotation_research as rr  # noqa: E402


def _months(n: int, start: str = "2000-01") -> pd.PeriodIndex:
    return pd.period_range(start, periods=n, freq="M")


def test_sahm_indicator_matches_definition() -> None:
    u = pd.Series([4.0] * 15 + [4.3, 4.6, 4.9], index=_months(18))
    s = rr.sahm_indicator(u)
    # 마지막 달: 3개월 평균 4.6 − 직전 12개월 3개월 평균 최저 4.0 = 0.6
    assert abs(s.iloc[-1] - 0.6) < 1e-12
    assert np.isnan(s.iloc[10])                    # 3개월 평균 + 12개월 최저가 다 쌓이기 전
    assert abs(s.iloc[14]) < 1e-12


def test_momentum_weights_use_only_past_months() -> None:
    idx = _months(10)
    r = pd.DataFrame({"A": [0.0] * 10, "B": [0.0] * 10, "C": [0.0] * 10}, index=idx)
    r.loc[idx[6], "C"] = 0.50                      # 7번째 달에 C 가 크게 오른다
    w = rr.momentum_weights(r, lookback=3, top=1)
    assert w.loc[idx[6], "C"] == 0.0               # 그달 수익은 그달 비중을 정할 때 모른다
    assert w.loc[idx[7], "C"] == 1.0               # 다음 달부터 반영
    assert w.iloc[:3].isna().all().all()           # 직전 3개월이 다 쌓이기 전에는 결정 안 함


def test_momentum_weights_skip_assets_without_history() -> None:
    idx = _months(8)
    r = pd.DataFrame({"A": [0.01] * 8, "B": [0.02] * 8, "NEW": [np.nan] * 5 + [0.30] * 3}, index=idx)
    w = rr.momentum_weights(r, lookback=3, top=1, min_assets=2)
    assert w.loc[idx[5], "NEW"] == 0.0 and w.loc[idx[5], "B"] == 1.0   # NEW 는 3개월이 쌓이기 전
    assert w.loc[idx[7], "NEW"] == 0.0                                    # idx[7] 직전 3개월 = idx[4..6], idx[4] 가 NaN
    assert rr.momentum_weights(r, lookback=3, top=1, min_assets=3).iloc[:8].isna().all().all()


def test_simulate_turnover_and_cost() -> None:
    idx = _months(3)
    a = pd.DataFrame({"X": [0.10, 0.0, 0.0], "Y": [0.0, 0.0, 0.0]}, index=idx)
    w = pd.DataFrame({"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 1.0]}, index=idx)
    out = rr.simulate(w, a, cost=0.01)
    assert abs(out["turnover"].iloc[0] - 1.0) < 1e-12          # 처음 매수
    assert abs(out["ret"].iloc[0] - (0.10 - 0.01)) < 1e-12
    assert abs(out["turnover"].iloc[1] - 2.0) < 1e-12          # X 전부 팔고 Y 전부 사기
    assert abs(out["turnover"].iloc[2]) < 1e-12                # 그대로 두면 매매 없음
