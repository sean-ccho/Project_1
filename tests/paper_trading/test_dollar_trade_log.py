"""거래로그 탭 = $5,000 계좌 기준으로 통째로 다시 쓰기 (가짜 시트, 네트워크 없음)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import paper_trading.sheet_sync as ss
from paper_trading.pt1_account import allocate, allocation_events, lots_from_records

TRADES = [
    {"ticker": "A", "entry_date": "2026-10-01", "exit_date": "2026-10-08", "entry_price": 10.0, "exit_price": 11.0,
     "return_pct": 0.10, "holding_days": 7, "strategy": "📈 모멘텀", "star_rating": "★★★★★ (12.0)", "ccs_score": 0.6,
     "sector": "Tech", "exit_reason": "목표가"},
    {"ticker": "B", "entry_date": "2026-10-02", "exit_date": "2026-10-08", "entry_price": 4000.0, "exit_price": 3900.0,
     "return_pct": -0.025, "holding_days": 6, "strategy": "📉 바닥반등", "star_rating": "★★★★★ (7.0)", "exit_reason": "손절"},
]
POSITIONS = [{"ticker": "C", "entry_date": "2026-10-08", "entry_price": 50.0, "strategy": "📉 바닥반등"}]


class FakeWS:
    id, row_count, col_count, title = 7, 1000, 20, "페이퍼_거래로그"

    def __init__(self):
        self.rows, self.cleared = [["옛 머리글"], ["옛 줄"]], False

    def clear(self):
        self.rows, self.cleared = [], True

    def update(self, rows, *a, **k):
        self.rows = rows

    def resize(self, rows=None, cols=None):
        self.row_count, self.col_count = rows or self.row_count, cols or self.col_count


class FakeSheet:
    def __init__(self):
        self.requests = []

    def batch_update(self, body):
        self.requests += body["requests"]


def test_allocate_records_cash_after_each_event():
    lots = lots_from_records(TRADES, POSITIONS)
    cash = allocate(lots, max_positions=3, initial=5000.0, cost=0.001)
    a, b, c = lots
    assert a.shares == 166 and math.isclose(a.cash_after_buy, 5000 - a.cost)
    assert b.shares == 0 and b.cost == 0                         # $1,667 로는 $4,000 짜리 1주도 못 산다
    assert math.isclose(c.cash_after_buy, cash)                  # 마지막 사건 뒤 현금 = 남은 현금
    order = [(d, s, lots[i].ticker) for d, s, i in allocation_events(lots)]
    assert order[-3:] == [("2026-10-08", "sell", "A"), ("2026-10-08", "sell", "B"), ("2026-10-08", "buy", "C")]


def test_rows_header_values_zero_share_and_spy():
    rows = ss.dollar_log_rows(TRADES, POSITIONS, max_positions=3, spy_fn=lambda t: 0.01)
    hdr, body = rows[0], rows[1:]
    assert hdr == ss.DOLLAR_LOG_HEADERS and len(body) == 5
    col = {h: i for i, h in enumerate(hdr)}
    sell_a = next(r for r in body if r[col["종목"]] == "A" and r[col["액션"]] == "🔴 SELL")
    assert sell_a[col["주식 수"]] == 166 and sell_a[col["손익($)"]] > 0
    assert sell_a[col["수익률"]].startswith("+") and sell_a[col["SPY동기간%"]] == "+1.0%"
    assert sell_a[col["SPY대비%p"]].startswith("+")
    buy_b = next(r for r in body if r[col["종목"]] == "B" and r[col["액션"]] == "🟢 BUY")
    assert buy_b[col["주식 수"]] == 0 and buy_b[col["사유"]].startswith("0주")


def test_rebuild_rewrites_same_tab_and_resets_formats(monkeypatch):
    ws, sheet = FakeWS(), FakeSheet()
    monkeypatch.setattr(ss, "_open_sheet", lambda: sheet)
    monkeypatch.setattr(ss, "_get_or_create_worksheet", lambda s, name, **k: ws)
    monkeypatch.setattr(ss, "_spy_return", lambda t: None)
    assert ss.rebuild_trade_log(TRADES, POSITIONS, worksheet="페이퍼_거래로그")
    assert ws.cleared and ws.rows[0] == ss.DOLLAR_LOG_HEADERS and len(ws.rows) == 6
    assert sheet.requests == [{"updateCells": {"range": {"sheetId": 7}, "fields": "userEnteredFormat"}}]


def test_sync_all_rebuilds_log_and_summary_uses_dollar_returns(monkeypatch):
    calls = {}
    monkeypatch.setattr(ss, "rebuild_trade_log", lambda t, p, worksheet: calls.setdefault("log", (len(t), len(p), worksheet)))
    monkeypatch.setattr(ss, "sync_trade_log", lambda *a, **k: calls.setdefault("append", True))
    monkeypatch.setattr(ss, "sync_positions", lambda *a, **k: True)
    monkeypatch.setattr(ss, "sync_summary", lambda t, worksheet, positions=None: calls.setdefault("summary", positions))
    ss.sync_all({"sells": [TRADES[0]], "buys": [POSITIONS[0]]}, POSITIONS, TRADES, {},
                tabs={"log": "L", "positions": "P", "summary": "S"})
    assert calls["log"] == (2, 1, "L") and "append" not in calls and calls["summary"] == POSITIONS
    dt = ss.dollar_trades(TRADES, POSITIONS)
    assert [t["ticker"] for t in dt] == ["A"]                    # 0주 거래(B)는 세지 않는다 ($5,000 계좌 요약과 같게)
    assert dt[0]["return_pct"] < 0.10                            # 비용 포함이라 원래 10% 보다 작다
