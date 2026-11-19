"""보유 중 액면분할 (2026-11-18) — 매수가 · 고점을 분할 뒤 기준으로, 두 번 고치지 않기, $5,000 계좌 주식 수 (네트워크 없이)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from paper_trading.engine import check_sell_conditions  # noqa: E402
from paper_trading.portfolio import apply_splits, close_position  # noqa: E402
from paper_trading.pt1_account import Lot, allocate, lots_from_records  # noqa: E402


def _pos(**kw):
    p = {"ticker": "EMBJ", "entry_price": 80.0, "entry_date": "2026-11-10", "strategy": "📈 모멘텀",
         "highest_price": 81.0}
    p.update(kw)
    return p


def test_forward_split_rescales_entry_and_high():
    pos = [_pos()]
    notes = apply_splits(pos, {"EMBJ": [("2026-11-16", 4.0)]}, "2026-11-17")
    assert pos[0]["entry_price"] == 20.0 and pos[0]["highest_price"] == 20.25
    assert pos[0]["split_factor"] == 4.0 and pos[0]["splits"] == ["2026-11-16:4"]
    assert notes and "EMBJ 4:1" in notes[0]
    # 분할 뒤 가격 19.8 — 고치기 전이면 −75% 손절, 고친 뒤엔 −1% · 고점 대비 −2.2% 로 매도 아님
    assert check_sell_conditions(_pos(), 19.8, "2026-11-17")[0]
    assert not check_sell_conditions(pos[0], 19.8, "2026-11-17")[0]


def test_split_applied_once():
    pos = [_pos()]
    s = {"EMBJ": [("2026-11-16", 4.0)]}
    apply_splits(pos, s, "2026-11-17")
    assert apply_splits(pos, s, "2026-11-18") == []
    assert pos[0]["entry_price"] == 20.0 and pos[0]["split_factor"] == 4.0


def test_old_future_and_same_day_splits_are_ignored():
    s = {"EMBJ": [("2020-08-31", 4.0), ("2026-11-10", 2.0), ("2026-12-01", 3.0)]}
    pos = [_pos()]
    assert apply_splits(pos, s, "2026-11-17") == []     # 매수 전 · 매수일 당일(분할 뒤 가격에 샀음) · 아직 안 온 날
    assert pos[0]["entry_price"] == 80.0 and "split_factor" not in pos[0]


def test_reverse_split():
    pos = [_pos(entry_price=0.15, highest_price=0.16)]
    apply_splits(pos, {"EMBJ": [("2026-11-16", 1 / 30)]}, "2026-11-17")
    assert abs(pos[0]["entry_price"] - 4.5) < 1e-9 and abs(pos[0]["split_factor"] - 1 / 30) < 1e-12


def test_trade_record_keeps_split_factor():
    pos = [_pos()]
    apply_splits(pos, {"EMBJ": [("2026-11-16", 4.0)]}, "2026-11-17")
    trades: list = []
    t = close_position(pos, trades, ticker="EMBJ", exit_price=22.0, exit_date="2026-11-20", reason="익절")
    assert t["split_factor"] == 4.0 and abs(t["return_pct"] - 0.1) < 1e-9
    plain: list = []
    t2 = close_position([_pos()], plain, ticker="EMBJ", exit_price=88.0, exit_date="2026-11-20", reason="익절")
    assert "split_factor" not in t2


def test_dollar_account_buys_whole_shares_at_paid_price():
    # $5,000 · 자리 1 개: 그날 80 에 62주 (비용 0.1%) → 4:1 분할 뒤 248주, 매수금액은 그대로
    lots = lots_from_records([], [{"ticker": "EMBJ", "entry_date": "2026-11-10", "entry_price": 20.0,
                                    "split_factor": 4.0}])
    cash = allocate(lots, max_positions=1, initial=5000.0, cost=0.001)
    assert lots[0].shares == 62 * 4
    assert abs(lots[0].cost - 62 * 80.0 * 1.001) < 1e-9 and abs(cash - (5000 - 62 * 80.0 * 1.001)) < 1e-9
    no_split = [Lot("EMBJ", "2026-11-10", 80.0)]
    allocate(no_split, max_positions=1, initial=5000.0, cost=0.001)
    assert no_split[0].shares == 62
