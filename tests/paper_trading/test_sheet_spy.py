"""시트 거래로그·성과요약의 SPY 같은 기간 비교 열 테스트 (가짜 워크시트, 네트워크 없음)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import paper_trading.sheet_sync as ss
from paper_trading.benchmarks import spy_window_return


class FakeWS:
    def __init__(self, rows=None):
        self.rows = rows or []

    def get_all_values(self):
        return self.rows

    def insert_row(self, row, index=1, value_input_option=None):
        self.rows.insert(index - 1, row)

    def append_row(self, row, value_input_option=None):
        self.rows.append(row)

    def update(self, rows, *args, **kwargs):
        if args and args[0] == "A1":
            self.rows[0] = rows[0]
        else:
            self.rows = rows

    def clear(self):
        self.rows = []


SPY = pd.Series([100.0, 101.0, 102.0, 105.0],
                index=pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-06"]))


def test_spy_window_return_uses_last_close_on_or_before():
    assert abs(spy_window_return(SPY, "2026-04-01", "2026-04-06") - 0.05) < 1e-12
    assert abs(spy_window_return(SPY, "2026-04-02", "2026-04-05") - (102 / 101 - 1)) < 1e-12  # 4/5 휴장 → 4/3 종가
    assert spy_window_return(SPY, "2026-03-01", "2026-04-06") is None
    assert spy_window_return(pd.Series(dtype=float), "2026-04-01", "2026-04-06") is None


def _patch(monkeypatch, ws):
    monkeypatch.setattr(ss, "_open_sheet", lambda: object())
    monkeypatch.setattr(ss, "_get_or_create_worksheet", lambda sheet, name, **k: ws)
    monkeypatch.setattr(ss, "_SPY_CLOSE", SPY)
    import screener.exporter as ex
    monkeypatch.setattr(ex, "_resize_worksheet_to_data", lambda ws, rows: None, raising=False)


def test_sell_row_has_spy_columns_and_old_header_is_extended(monkeypatch):
    old_header = ss._LOG_HEADERS[:12]
    ws = FakeWS([old_header])
    _patch(monkeypatch, ws)
    trade = {"ticker": "AAA", "entry_date": "2026-04-01", "exit_date": "2026-04-06", "entry_price": 10,
             "exit_price": 11, "return_pct": 0.10, "holding_days": 3, "exit_reason": "목표가"}
    assert ss.sync_trade_log(trade, action="SELL")
    assert ws.rows[0] == ss._LOG_HEADERS
    assert ws.rows[-1][-2:] == ["+5.0%", "+5.0"]
    assert ss.sync_trade_log({"ticker": "BBB", "entry_date": "2026-04-06"}, action="BUY")
    assert len(ws.rows[-1]) == len(ss._LOG_HEADERS)


def test_summary_spy_columns(monkeypatch):
    ws = FakeWS()
    _patch(monkeypatch, ws)
    trades = [
        {"entry_date": "2026-04-01", "exit_date": "2026-04-06", "return_pct": 0.10, "strategy": "📈 모멘텀"},
        {"entry_date": "2026-04-01", "exit_date": "2026-04-03", "return_pct": -0.01, "strategy": "📈 모멘텀"},
    ]
    assert ss.sync_summary(trades)
    header, total = ws.rows[0], ws.rows[1]
    assert header[7:10] == ["SPY동기간평균", "SPY대비평균", "SPY이긴비율"]
    assert total[0] == "전체" and total[7] == "+3.5%" and total[8] == "+1.0%p" and total[9] == "50%"


def test_spy_comparison():
    from paper_trading.benchmarks import spy_comparison
    trades = [{"entry_date": "2026-04-01", "exit_date": "2026-04-06", "return_pct": 0.10},
              {"entry_date": "2026-04-01", "exit_date": "2026-04-03", "return_pct": -0.01},
              {"entry_date": "2025-01-01", "exit_date": "2025-01-05", "return_pct": 0.5}]  # SPY 데이터 밖 → 제외
    c = spy_comparison(trades, SPY)
    assert c["n"] == 2 and abs(c["diff"] - 0.01) < 1e-12 and c["beat"] == 0.5
    assert spy_comparison([], SPY) is None
