"""SEC 재무·실적일 연구 도구 테스트 (네트워크 없음)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_sec_data import _first_filed, reaction_sessions  # noqa: E402
from fundamental_research import _split_after  # noqa: E402

SESSIONS = pd.DatetimeIndex(pd.bdate_range("2024-01-01", "2024-12-31"))


def test_reaction_session_after_close_is_next_day_dst_aware() -> None:
    # 여름(EDT): 20:16Z = 16:16 ET → 다음 거래일. 겨울(EST): 21:16Z = 16:16 ET → 다음 거래일
    out = reaction_sessions(["2024-08-06T20:16:00.000Z", "2024-02-06T21:16:00.000Z"], SESSIONS)
    assert [d for _, d in out] == [pd.Timestamp("2024-02-07"), pd.Timestamp("2024-08-07")]
    assert out[1][0].hour == 16


def test_reaction_session_premarket_same_day_and_weekend_rolls() -> None:
    # 11:36Z = 07:36 ET → 그날. 금요일 장 마감 후 → 월요일
    out = reaction_sessions(["2024-07-16T11:36:00.000Z", "2024-07-19T20:30:00.000Z"], SESSIONS)
    assert [d for _, d in out] == [pd.Timestamp("2024-07-16"), pd.Timestamp("2024-07-22")]


def test_first_filed_keeps_original_not_restated_and_annual_only() -> None:
    units = [
        {"start": "2022-01-01", "end": "2022-12-31", "val": 100, "form": "10-K", "filed": "2023-02-10"},
        {"start": "2022-01-01", "end": "2022-12-31", "val": 90, "form": "10-K", "filed": "2024-02-10"},  # 정정 비교치
        {"start": "2022-10-01", "end": "2022-12-31", "val": 30, "form": "10-K", "filed": "2023-02-10"},  # 분기
        {"start": "2023-01-01", "end": "2023-12-31", "val": 120, "form": "10-Q", "filed": "2024-02-10"},  # 10-Q 제외
    ]
    v = _first_filed(units, annual_flow=True)
    assert v["val"].tolist() == [100]
    assert v["filed"].iloc[0] == pd.Timestamp("2023-02-10")


def test_split_after_multiplies_only_later_splits() -> None:
    sp = pd.DataFrame({"date": pd.to_datetime(["2020-08-31", "2024-06-10"]), "티커": ["X", "X"], "ratio": [4.0, 10.0]})
    g = _split_after(sp)
    assert g("X", pd.Timestamp("2019-12-31")) == 40.0
    assert g("X", pd.Timestamp("2021-01-01")) == 10.0
    assert g("X", pd.Timestamp("2024-06-10")) == 1.0
    assert g("Y", pd.Timestamp("2019-12-31")) == 1.0


def test_latest_fiscal_year_ignores_late_amendment_of_old_year() -> None:
    from fundamental_research import latest_fiscal_year
    rebal = pd.DataFrame({"date": pd.to_datetime(["2024-06-28"]), "티커": ["X"]})
    fa = pd.DataFrame({"티커": ["X", "X", "X"],
                       "fy_end": pd.to_datetime(["2022-12-31", "2023-12-31", "2019-12-31"]),
                       "filed": pd.to_datetime(["2023-02-15", "2024-02-15", "2024-05-01"]),  # 2019 연도 늦은 정정
                       "assets": [1.0, 2.0, 9.0]})
    out = latest_fiscal_year(rebal, fa)
    assert out["assets"].tolist() == [2.0]
    # 아직 공개 전인 연도는 쓰지 않는다
    early = latest_fiscal_year(pd.DataFrame({"date": pd.to_datetime(["2024-02-15"]), "티커": ["X"]}), fa)
    assert early["assets"].tolist() == [1.0]
