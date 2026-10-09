"""PT-1 $5,000 계좌 테스트 — 배분·현금 장부, 날짜별 평가액, SPY 비교, 표시 (네트워크 없이)."""
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from paper_trading.pt1_account import (  # noqa: E402
    Lot,
    allocate,
    build_dollar_account,
    daily_equity,
    section_html,
    sheet_rows,
    summary,
)

DATES = pd.bdate_range("2026-09-01", periods=5)  # 9/1 ~ 9/7


def _trade(t, e, ep, x, xp):
    return {"ticker": t, "entry_date": e, "entry_price": ep, "exit_date": x, "exit_price": xp}


def _frame(closes: dict, spy_open: list[float], spy_close: list[float]) -> pd.DataFrame:
    cols = {(t, "Close"): v for t, v in closes.items()}
    cols.update({(t, "Open"): v for t, v in closes.items()})
    cols[("SPY", "Open")], cols[("SPY", "Close")] = spy_open, spy_close
    return pd.DataFrame(cols, index=DATES)


def test_allocate_splits_cash_over_empty_slots_and_sells_first():
    lots = [Lot("AAA", "2026-09-01", 10.0, "2026-09-03", 12.0), Lot("BBB", "2026-09-02", 20.0),
            Lot("CCC", "2026-09-03", 5.0)]
    cash = allocate(lots, max_positions=2, initial=1000.0, cost=0.0)
    a, b, c = lots
    assert (a.cost, a.shares) == (500.0, 50.0)          # 빈 자리 2개 → 절반
    assert (b.cost, b.shares) == (500.0, 25.0)          # 남은 자리 1개 → 남은 현금 전부
    assert a.proceeds == 600.0                          # 9/3 AAA 판 돈으로
    assert (c.cost, c.shares) == (600.0, 120.0)         # 같은 날 CCC 매수
    assert cash == 0.0


def test_allocate_same_day_round_trip_buys_before_selling():
    lots = [Lot("AAA", "2026-09-01", 10.0, "2026-09-01", 9.0)]
    cash = allocate(lots, max_positions=3, initial=900.0, cost=0.0)
    assert lots[0].shares == 30.0 and cash == 900.0 - 300.0 + 270.0


def test_daily_equity_marks_holdings_and_falls_back_to_entry_price():
    lots = [Lot("AAA", "2026-09-01", 10.0, "2026-09-03", 12.0), Lot("BBB", "2026-09-02", 20.0)]
    allocate(lots, max_positions=2, initial=1000.0, cost=0.0)
    closes = pd.DataFrame({"AAA": [11.0, 11.5, 12.0, 12.5, 13.0], "BBB": [float("nan")] * 3 + [22.0, float("nan")]},
                          index=DATES)
    curve = daily_equity(lots, 1000.0, closes)
    # 9/1: 현금 500 + AAA 50주×11 / 9/2: 현금 0 + 575 + BBB 종가 없음 → 매수가 25주×20
    assert curve.iloc[0] == 1050.0 and curve.iloc[1] == 575.0 + 500.0
    assert curve.iloc[2] == 600.0 + 500.0               # 9/3 AAA 매도 → 현금
    assert curve.iloc[3] == 600.0 + 25 * 22.0           # 9/4 BBB 종가
    assert curve.iloc[4] == 600.0 + 25 * 22.0           # 9/7 종가 없음 → 마지막 종가


def test_build_account_compares_with_spy_bought_same_day():
    trades = [_trade("AAA", "2026-09-01", 10.0, "2026-09-03", 12.0)]
    positions = [{"ticker": "BBB", "entry_date": "2026-09-02", "entry_price": 20.0}]
    frame = _frame({"AAA": [11.0, 11.5, 12.0, 12.5, 13.0], "BBB": [20.0, 20.0, 21.0, 22.0, 24.0]},
                   spy_open=[100.0] * 5, spy_close=[100.0, 101.0, 102.0, 103.0, 110.0])
    acct = build_dollar_account(trades, positions, frame=frame, initial=1000.0, max_positions=2, cost=0.0)
    s = summary(acct)
    assert s["start"] == "2026-09-01" and s["holdings"] == 1 and s["n_closed"] == 1
    assert math.isclose(s["equity"], 600.0 + 25 * 24.0) and math.isclose(s["ret"], 0.2)
    assert math.isclose(s["spy_equity"], 1100.0) and math.isclose(s["spy_ret"], 0.1)
    assert math.isclose(s["realized"], 100.0) and math.isclose(s["unrealized"], 100.0)

    html = section_html([("PT-1", acct), ("PT-1S (S&P 500만)", build_dollar_account([], [], frame=frame))])
    assert "$1,200" in html and "+10.0%p" in html and "아직 거래 없음" in html
    rows = sheet_rows(acct, "PT-1")
    assert rows[3][:4] == ["2026-09-01", "$1,000", "$1,200", "+20.0%"]
    assert ["날짜", "평가액", "같은 날 SPY 샀으면"] in rows and rows[-1] == ["2026-09-07", 1200.0, 1100.0]
