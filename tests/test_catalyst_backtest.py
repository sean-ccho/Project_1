"""scripts/catalyst_backtest.py — 13D · 스핀오프 사건 찾기 · 월간 자본곡선 (2026-11-16)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import catalyst_backtest as cb  # noqa: E402


def test_activist_names():
    assert cb.activist_of("Legion Partners Asset Management, LLC") == "Legion"
    assert cb.activist_of("Starboard Value LP") == "Starboard"
    assert cb.activist_of("ELLIOTT ASSOCIATES, L.P.") == "Elliott"
    assert cb.activist_of("ICAHN CARL C") == "Icahn"
    assert cb.activist_of("Trian Fund Management, L.P.") == "Trian"
    assert cb.activist_of("THIRD POINT LLC") == "Third Point"
    assert cb.activist_of("THIRD POINT REINSURANCE LTD.") is None      # 보험사
    assert cb.activist_of("GAMCO INVESTORS, INC. ET AL") is None         # 수동형 13D 는 뺐다
    assert cb.activist_of("ANCORA PHARMACEUTICALS INC") is None
    assert cb.activist_of("Elliott Opportunity II Corp") is None        # 이름만 비슷한 회사


def test_parse_master_and_act_pairs():
    text = "\n".join([
        "CIK|Company Name|Form Type|Date Filed|Filename",
        "1005286|LANDEC CORP \\CA\\|SC 13D|2020-01-13|edgar/data/1005286/0000921895-20-000125.txt",
        "1560207|Legion Partners Asset Management, LLC|SC 13D|2020-01-13|edgar/data/1560207/0000921895-20-000125.txt",
        "1001115|GEOSPACE TECHNOLOGIES CORP|SC 13D|2020-01-14|edgar/data/1001115/0001445866-20-000141.txt",
        "1710368|SANSONE CAPITAL MANAGEMENT, LLC|SC 13D|2020-01-14|edgar/data/1710368/0001445866-20-000141.txt",
        "1517137|Starboard Value LP|SC 13D/A|2020-01-15|edgar/data/1517137/0000921895-20-000200.txt",
        "1781335|Otis Worldwide Corp|10-12B|2020-02-07|edgar/data/1781335/0001140361-20-002547.txt",
        # 공동 신고 (세 줄) — 우리 목록에 있는 쪽이 대상
        "111|ACME CORP|SC 13D|2021-03-01|edgar/data/111/0000000000-21-000001.txt",
        "222|SMITH JOHN|SC 13D|2021-03-01|edgar/data/222/0000000000-21-000001.txt",
        "333|Starboard Value LP|SC 13D|2021-03-01|edgar/data/333/0000000000-21-000001.txt",
    ])
    rows = cb.parse_master(text, cb.FORMS_13D + cb.FORMS_SPIN)
    assert len(rows) == 8                                 # 13D/A 는 안 받는다
    ev = cb.act_events(rows, known={1005286, 111})
    assert [(e["cik"], e["activist"]) for e in ev] == [(1005286, "Legion"), (111, "Starboard")]
    assert len(cb.act_events(rows, known=None)) == 1      # 목록 없으면 공동 신고는 대상을 못 정해 뺀다
    sp = cb.spin_filings(rows)
    assert sp[1781335]["date"] == "2020-02-07"


def test_entry_month():
    assert cb.entry_month_act("2020-02-13") == pd.Timestamp("2020-02-29")
    assert cb.entry_month_act("2020-03-31") == pd.Timestamp("2020-04-30")   # 말일 공시는 다음 달 말


def test_monthly_returns_cash_after_delisting():
    idx = pd.date_range("2020-01-31", "2020-06-30", freq="ME")
    df = pd.DataFrame({"adj": [10.0, 11.0, 12.1, np.nan, np.nan, np.nan], "close": 1.0}, index=idx)
    r = cb.monthly_returns(df, list(idx))
    assert r[0] == 0 and r[1] == pytest.approx(0.1) and r[2] == pytest.approx(0.1) and (r[3:] == 0).all()


def test_calendar_portfolio_equal_weight_spy_fill_and_cost():
    months = list(pd.date_range("2020-01-31", "2020-05-31", freq="ME"))
    rets = {"A": np.array([0.10, 0.10, 0.10, 0.10, 0.10]), "B": np.array([0.0, -0.10, 0.0, 0.0, 0.0])}
    spy = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    pos = [{"ticker": "A", "entry": "2020-01-31", "hold": 2}, {"ticker": "B", "entry": "2020-01-31", "hold": 1},
           {"ticker": "A", "entry": "2020-02-29", "hold": 1}]             # 겹치는 A 는 한 번만
    r, n = cb.calendar_portfolio(pos, rets, months, spy, cost=0.0)
    assert list(n) == [0, 2, 1, 0, 0]
    assert r[0] == pytest.approx(0.01)                                    # 보유 없음 → SPY
    assert r[1] == pytest.approx((0.10 - 0.10) / 2)
    assert r[2] == pytest.approx(0.10)
    r_c, _ = cb.calendar_portfolio(pos, rets, months, spy, cost=0.01)
    assert r_c[1] == pytest.approx((1 + 0.0) * (1 - 0.01 * 2.0) - 1)      # SPY 1 → A ½ · B ½ = 매매 2
