"""시트 거래로그 append — 빈 시트(gspread 6 은 [[]] 반환)와 같은 일봉 재처리 중복 방지 (가짜 워크시트, 네트워크 없음)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import paper_trading.sheet_sync as ss


class GspreadLikeWS:
    """빈 시트에서 get_all_values() 가 [[]] 를 돌려주는 gspread 6.2 동작을 흉내 낸다."""

    def __init__(self, rows=None):
        self.rows = rows or []

    def get_all_values(self):
        return [list(r) for r in self.rows] or [[]]

    def insert_row(self, row, index=1, value_input_option=None):
        self.rows.insert(index - 1, list(row))

    def append_row(self, row, value_input_option=None):
        self.rows.append(list(row))

    def update(self, rows, *args, **kwargs):
        self.rows[0] = list(rows[0])


def _patch(monkeypatch, ws):
    monkeypatch.setattr(ss, "_open_sheet", lambda: object())
    monkeypatch.setattr(ss, "_get_or_create_worksheet", lambda sheet, name, **k: ws)


BUY = {"ticker": "MPWR", "entry_date": "2026-10-09", "entry_price": 1386.29, "sector": "IT",
       "strategy": "📈 모멘텀", "star_rating": "★★★★★ (12.0)", "ccs_score": 0.685}


def test_empty_sheet_gets_header_and_row(monkeypatch):
    """10-07 ~ 10-11 PT-1S 탭: 빈 시트에서 existing[0][0] 이 IndexError 라 한 줄도 못 썼다."""
    ws = GspreadLikeWS()
    _patch(monkeypatch, ws)
    assert ss.sync_trade_log(BUY, action="BUY")
    assert ws.rows[0] == ss._LOG_HEADERS
    assert ws.rows[1][:2] == ["2026-10-09", "MPWR"] and "BUY" in ws.rows[1][3]


def test_same_bar_reprocessed_does_not_duplicate(monkeypatch):
    ws = GspreadLikeWS()
    _patch(monkeypatch, ws)
    assert ss.sync_trade_log(BUY, action="BUY")
    assert ss.sync_trade_log(BUY, action="BUY")  # 상태가 저장 안 돼 같은 일봉을 다시 처리한 경우
    assert len(ws.rows) == 2


def test_dedupe_understands_sheet_date_format(monkeypatch):
    """USER_ENTERED 로 날짜가 된 셀은 '2026. 10. 9' 처럼 보일 수 있다."""
    row = ["2026. 10. 9", "MPWR", "IT", "🟢 BUY"] + [""] * (len(ss._LOG_HEADERS) - 4)
    ws = GspreadLikeWS([ss._LOG_HEADERS, row])
    _patch(monkeypatch, ws)
    assert ss.sync_trade_log(BUY, action="BUY")
    assert len(ws.rows) == 2


def test_sell_after_buy_and_later_rebuy_are_kept(monkeypatch):
    ws = GspreadLikeWS()
    _patch(monkeypatch, ws)
    monkeypatch.setattr(ss, "_spy_return", lambda trade: None)
    sell = dict(BUY, exit_date="2026-10-20", exit_price=1400.0, return_pct=0.01, holding_days=11, exit_reason="트레일링")
    assert ss.sync_trade_log(BUY, action="BUY")
    assert ss.sync_trade_log(sell, action="SELL")
    assert ss.sync_trade_log(dict(BUY, entry_date="2026-10-21"), action="BUY")
    assert [r[0] for r in ws.rows[1:]] == ["2026-10-09", "2026-10-20", "2026-10-21"]
