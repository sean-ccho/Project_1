"""idea_batch_v2 — 판정 뒤집기(피하기)·매출 가속 신호·이벤트 진입일."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import idea_batch_v2 as ib  # noqa: E402


def _r(t, mean, first, second, yp, ny=8):
    return {"t_nw": t, "mean_ex": mean, "mean_net": mean - 0.002, "first_half": first, "second_half": second,
            "years_pos": yp, "n_years": ny}


def test_judge_avoid_flips_sign_without_cost():
    """피하기 검정: 음(−)이 좋은 쪽. 비용은 빼지 않는다."""
    assert ib.judge(_r(-3.0, -0.001, -0.001, -0.002, 1), 2.87, "avoid") == "피하기 후보"
    assert ib.judge(_r(-2.4, -0.005, -0.001, -0.01, 2), 2.87, "avoid") == "피하기 관찰"
    assert ib.judge(_r(3.0, 0.01, 0.01, 0.01, 7), 2.87, "avoid") == "역신호(오히려 +)"
    assert ib.judge(_r(3.0, 0.01, 0.01, 0.01, 7), 2.87, "buy") == "후보"
    assert ib.judge({}, 2.87, "buy") == "표본 없음"


def test_revenue_signals_uses_only_quarters_known_before(monkeypatch, tmp_path):
    """그날 전에 공시된 분기만 쓰고, 2분기 연속 가속 + YoY ≥ 15% 면 RUNNER."""
    rev = pd.DataFrame({
        "티커": ["A"] * 4,
        "end": pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]),
        "known": pd.to_datetime(["2024-05-01", "2024-08-01", "2024-11-01", "2025-02-10"]),
        "yoy": [0.05, 0.10, 0.20, 0.30],
    })
    path = tmp_path / "rev.parquet"
    rev.to_parquet(path)
    monkeypatch.setattr(ib, "REV_PATH", path)
    rebal = pd.DataFrame({"date": pd.to_datetime(["2024-11-29", "2025-01-31", "2025-02-28"]), "티커": "A"})
    out = ib.revenue_signals(rebal).set_index("date")
    assert np.isclose(out.loc["2024-11-29", "REVACC"], 0.10) and out.loc["2024-11-29", "RUNNER"] == 1.0
    assert np.isclose(out.loc["2025-01-31", "yoy0"], 0.20)            # 12월 분기는 아직 공시 전
    assert np.isclose(out.loc["2025-02-28", "REVACC"], 0.10) and out.loc["2025-02-28", "RUNNER"] == 1.0


def test_event_entries_enter_day_after_calendar_date():
    idx = pd.bdate_range("2024-01-01", periods=10)
    opens = pd.DataFrame(1.0, index=idx, columns=["A", "B"])
    an = pd.DataFrame({"티커": ["A", "A", "C"], "action": ["up", "up", "up"],
                       "ts": pd.to_datetime(["2024-01-02 23:00", "2024-01-02 09:00", "2024-01-03 10:00"])})
    e = ib.event_entries(an, "up", opens, "2024-01-01")
    assert list(e["티커"]) == ["A"]                                    # 같은 날 두 건은 하나, 가격 없는 C 는 뺀다
    assert idx[int(e["entry_pos"].iloc[0])] == pd.Timestamp("2024-01-03")  # 이벤트 달력일 다음 거래일
