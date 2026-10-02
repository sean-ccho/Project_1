"""PT-2/PT-3 계좌 이메일 본문 생성 테스트 (발송 없이)."""

from types import SimpleNamespace

from paper_trading.account_email import build_account_email
from paper_trading.account_engine import AccountState
from paper_trading.accounts import AccountProfile


def _profile() -> AccountProfile:
    params = {"version": "pt3-v0", "data_dir": "unused", "max_positions": 5, "cost_per_side": 0.001}
    return AccountProfile("pt3", "PT-3 일봉 단타", params, {}, True, SimpleNamespace())


def test_build_email_with_activity():
    state = AccountState.new(5000.0)
    state.positions = [{
        "ticker": "AAA", "entry_date": "2026-01-06", "entry_price": 100.0, "last_close": 103.0,
        "stop_price": 100.0, "stop_kind": "본절", "target_price": 104.0, "bars_held": 2,
        "strategy": "단타-눌림목", "shares": 9.99,
    }]
    state.trades = [{"return_pct": 0.031}, {"return_pct": -0.02}]
    state.equity_history = [{"date": "2026-01-07", "equity": 5100.0}, {"date": "2026-01-08", "equity": 5050.0}]
    result = {
        "date": "2026-01-08",
        "buys": [{"ticker": "AAA", "entry_price": 100.0, "shares": 9.99, "stop_price": 97.0,
                  "target_price": 104.0, "strategy": "단타-눌림목"}],
        "sells": [{"ticker": "BBB", "entry_date": "2026-01-02", "entry_price": 50.0, "exit_price": 52.0,
                   "return_pct": 0.038, "holding_days": 6, "holding_bars": 4, "exit_reason": "목표가(+3.8%)"}],
        "orders": [{"side": "BUY", "ticker": "CCC", "signal_close": 20.0, "score": 0.81, "strategy": "단타-돌파"},
                   {"side": "SELL", "ticker": "AAA", "signal_close": 103.0, "reason": "셋업완료(눌림 회복)"}],
        "cancelled": [{"ticker": "DDD", "cancel_reason": "갭상승 +4.0%"}],
        "candidates": [{"ticker": "CCC", "score": 0.81, "strategy": "단타-돌파", "sector": "Tech", "star": "★★★"}],
        "debug": {"통과": 1},
    }
    subject, body = build_account_email(_profile(), result, state)
    assert subject == "[PT-3 일봉 단타] 2026-01-08 매매 알림"
    for text in ("AAA", "BBB", "CCC", "DDD", "갭상승 +4.0%", "셋업완료(눌림 회복)", "본절", "-1.0%"):
        assert text in body


def test_build_email_without_activity():
    state = AccountState.new(5000.0)
    result = {"date": "2026-01-08", "buys": [], "sells": [], "orders": [], "cancelled": [],
              "candidates": [], "debug": {}}
    subject, body = build_account_email(_profile(), result, state)
    assert subject.endswith("일일 리포트")
    assert "보유 종목 없음" in body and "조건을 만족한 후보 없음" in body
