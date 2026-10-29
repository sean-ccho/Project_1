"""pt1_slots: 평균 보유 종목 수 · 판정 규칙 점검."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pt1_slots as ps  # noqa: E402


def test_avg_holdings_counts_open_positions_and_skips_zero_share() -> None:
    idx = pd.bdate_range("2024-01-01", periods=4)
    trades = [
        {"entry_date": str(idx[0].date()), "exit_date": str(idx[2].date()), "shares": 5},   # 0·1일째 보유
        {"entry_date": str(idx[1].date()), "exit_date": str(idx[3].date()), "shares": 2},   # 1·2일째 보유
        {"entry_date": str(idx[0].date()), "exit_date": str(idx[3].date()), "shares": 0},   # 0주 매수는 안 셈
    ]
    assert ps.avg_holdings(trades, idx) == (1 + 2 + 1 + 0) / 4


def test_judge_order() -> None:
    win = {"ΔSharpe_하한": 0.05, "p_개선아님": 0.001}
    lose = {"ΔSharpe_하한": -0.20, "p_개선아님": 0.40}
    assert ps.judge(win, lose) == "SPY 이김"
    assert ps.judge(lose, win) == "개선"
    assert ps.judge(lose, lose) == "변화 없음"
    assert ps.judge({"ΔSharpe_하한": 0.01, "p_개선아님": 0.02}, lose) == "변화 없음"   # p 가 0.05/3 보다 큼
