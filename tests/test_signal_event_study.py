"""scripts/signal_event_study.py 테스트 — 이벤트 시작일·쿨다운·판정."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from signal_event_study import grade, onsets


def test_onsets_start_and_cooldown():
    dates = pd.bdate_range("2024-01-01", periods=30)
    flag_days = {0, 1, 2, 5, 15, 16}  # 0 시작, 5는 쿨다운(10일) 안, 15 시작
    sig = pd.DataFrame({"date": dates, "티커": "AAA"})
    flag = pd.Series([i in flag_days for i in range(30)])
    di = {d: i for i, d in enumerate(dates)}
    out = onsets(sig, flag, di, cooldown=10)
    assert list(out[out].index) == [0, 15]


def test_onsets_reentry_after_gap_counts():
    dates = pd.bdate_range("2024-01-01", periods=30)
    sig = pd.DataFrame({"date": dates[[0, 1, 20]], "티커": "AAA"})  # 1 → 20 사이 구성종목 이탈
    flag = pd.Series([True, True, True])
    di = {d: i for i, d in enumerate(dates)}
    assert onsets(sig, flag, di, cooldown=10).tolist() == [True, False, True]


def _r(t, net=0.01, halves=(0.01, 0.01), yp=3, ny=4):
    return {"t_nw": t, "mean_net": net, "first_half": halves[0], "second_half": halves[1],
            "years_pos": yp, "n_years": ny}


def test_grade():
    assert grade(_r(4.0), 3.5) == "후보"
    assert grade(_r(4.0, yp=2), 3.5) == "관찰"   # 연도 2/3 미달
    assert grade(_r(-4.0), 3.5) == "역신호"
    assert grade(_r(2.5, halves=(0.01, -0.01)), 3.5) == "탈락"
    assert grade({}, 3.5) == "표본 없음"
