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

    def __init__(self, rows=None):
        self.rows, self.cleared = rows if rows is not None else [["옛 머리글"], ["옛 줄"]], False

    def get_all_values(self):
        return [list(r) for r in self.rows]

    def clear(self):
        self.rows, self.cleared = [], True

    def update(self, rows, *a, **k):
        self.rows = rows

    def resize(self, rows=None, cols=None):
        self.row_count, self.col_count = rows or self.row_count, cols or self.col_count


class FakeSheet:
    def __init__(self, titles=(), fail_copy=False):
        self.requests, self.copies, self.titles, self.fail_copy = [], [], list(titles), fail_copy

    def batch_update(self, body):
        self.requests += body["requests"]

    def worksheets(self):
        return [type("W", (), {"title": t})() for t in self.titles]

    def duplicate_sheet(self, sheet_id, new_sheet_name=None):
        if self.fail_copy:
            raise RuntimeError("quota")
        self.copies.append((sheet_id, new_sheet_name))
        self.titles.append(new_sheet_name)


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


def _patched(monkeypatch, ws, sheet):
    monkeypatch.setattr(ss, "_open_sheet", lambda: sheet)
    monkeypatch.setattr(ss, "_get_or_create_worksheet", lambda s, name, **k: ws)
    monkeypatch.setattr(ss, "_spy_return", lambda t: None)


def test_rebuild_refuses_when_records_shrank(monkeypatch):
    """상태 파일이 날아가 기록이 줄었으면 시트를 지우지 않는다."""
    old = [ss.DOLLAR_LOG_HEADERS] + [["2026-04-02", f"T{i}"] for i in range(10)]
    ws, sheet = FakeWS([list(r) for r in old]), FakeSheet()
    _patched(monkeypatch, ws, sheet)
    assert not ss.rebuild_trade_log(TRADES, POSITIONS, worksheet="페이퍼_거래로그")   # 새 5줄 < 시트 10줄
    assert not ws.cleared and ws.rows == old


def test_old_format_is_copied_once_before_rewrite(monkeypatch):
    old = [list(ss._LOG_HEADERS), ["2026-10-01", "A", "", "🟢 BUY"]]
    ws, sheet = FakeWS([list(r) for r in old]), FakeSheet()
    _patched(monkeypatch, ws, sheet)
    assert ss.rebuild_trade_log(TRADES, POSITIONS, worksheet="페이퍼_거래로그")
    assert sheet.copies == [(7, "페이퍼_거래로그_예전")] and ws.rows[0] == ss.DOLLAR_LOG_HEADERS
    # 다음 날은 새 모양이라 복사하지 않는다
    assert ss.rebuild_trade_log(TRADES, POSITIONS, worksheet="페이퍼_거래로그") and len(sheet.copies) == 1


def test_old_format_copy_failure_keeps_sheet(monkeypatch):
    old = [list(ss._LOG_HEADERS), ["2026-10-01", "A", "", "🟢 BUY"]]
    ws, sheet = FakeWS([list(r) for r in old]), FakeSheet(fail_copy=True)
    _patched(monkeypatch, ws, sheet)
    assert not ss.rebuild_trade_log(TRADES, POSITIONS, worksheet="페이퍼_거래로그")
    assert not ws.cleared and ws.rows == old
